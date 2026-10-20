"""
Type checking tests

The type hints are tested with pyright (used by VS Code / Pylance) and mypy:

- the ``ok_*.py`` files show correct usage, nothing may be reported in them
- in the ``errors*.py`` files every line marked with ``# E: <reason>`` must be reported, and nothing else

Run it from the project root: ``python tests/typecheck/check.py``
"""
from typing import Dict, List, Set, Tuple

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

MARKER = re.compile(r'#\s*E:')
MYPY_LINE = re.compile(r'^(?P<file>[^:]+):(?P<line>\d+):(?:\d+:)? (?P<severity>error|warning): (?P<message>.*)$')

# Diagnostics: file -> line -> messages
Diagnostics = Dict[str, Dict[int, List[str]]]


def run_pyright(files: List[Path]) -> Diagnostics:
    """ Run pyright, the files have a `# pyright: standard` comment, so the project settings are not used """
    proc = subprocess.run([sys.executable, '-m', 'pyright', '--outputjson', '--pythonpath', sys.executable,
                           *map(str, files)], cwd=ROOT, capture_output=True, text=True)
    try:
        output = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit(f"Pyright failed:\n{proc.stdout}\n{proc.stderr}")
    res: Diagnostics = {}
    for diag in output['generalDiagnostics']:
        if diag['severity'] not in ('error', 'warning'):
            continue
        line = diag['range']['start']['line'] + 1
        msg = f"{diag['message'].splitlines()[0]} ({diag.get('rule', '')})"
        res.setdefault(str(Path(diag['file']).resolve()), {}).setdefault(line, []).append(msg)
    return res


def run_mypy(files: List[Path]) -> Diagnostics:
    """ Run mypy, only the given files are reported, not the imported modules """
    proc = subprocess.run([sys.executable, '-m', 'mypy', '--follow-imports=silent', '--no-error-summary',
                           '--hide-error-context', '--no-pretty', '--show-error-codes', '--no-incremental',
                           *map(str, files)], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.exit(f"Mypy failed:\n{proc.stdout}\n{proc.stderr}")
    res: Diagnostics = {}
    for out_line in proc.stdout.splitlines():
        m = MYPY_LINE.match(out_line)
        if not m:
            continue
        file = str((ROOT / m['file']).resolve())
        res.setdefault(file, {}).setdefault(int(m['line']), []).append(m['message'])
    return res


def expected_lines(file: Path) -> Set[int]:
    """ The lines marked as errors """
    return {i for i, line in enumerate(file.read_text().splitlines(), 1) if MARKER.search(line)}


def check(tool: str, diags: Diagnostics, files: List[Path]) -> List[str]:
    """ Compare the reported and the expected lines """
    problems = []
    for file in files:
        lines = file.read_text().splitlines()
        reported = diags.get(str(file.resolve()), {})
        expected = expected_lines(file)
        rel = file.relative_to(ROOT)
        for line in sorted(set(reported) - expected):
            for msg in reported[line]:
                problems.append(f"{tool}: {rel}:{line}: unexpected: {msg}")
        for line in sorted(expected - set(reported)):
            problems.append(f"{tool}: {rel}:{line}: not reported: {lines[line - 1].strip()}")
    return problems


def check_copies() -> List[str]:
    """
    ESQueryClause and ESSubAgg are copies of ESQuery and ESAgg (see their docstrings), their keys must be the same
    """
    sys.path.insert(0, str(ROOT))
    from esorm.query import ESQuery, ESQueryClause
    from esorm.aggs import ESAgg, ESSubAgg

    problems = []
    for orig, copy in ((ESQuery, ESQueryClause), (ESAgg, ESSubAgg)):
        orig_keys = {k: str(v) for k, v in orig.__annotations__.items()}
        copy_keys = {k: str(v) for k, v in copy.__annotations__.items()}
        if orig_keys != copy_keys:
            diff = sorted(set(orig_keys.items()) ^ set(copy_keys.items()))
            problems.append(f"{copy.__name__} must have the same keys as {orig.__name__}, differences: {diff}")
    return problems


def main() -> int:
    files = sorted(HERE.glob('ok_*.py')) + sorted(HERE.glob('errors*.py'))
    problems: List[str] = check_copies()
    summary: List[Tuple[str, int]] = []
    for tool, runner in (('pyright', run_pyright), ('mypy', run_mypy)):
        diags = runner(files)
        tool_problems = check(tool, diags, files)
        problems += tool_problems
        summary.append((tool, len(tool_problems)))

    for problem in problems:
        print(problem)
    n_expected = sum(len(expected_lines(f)) for f in files)
    for tool, n in summary:
        print(f"{tool}: {'OK' if n == 0 else f'{n} problems'} ({len(files)} files, {n_expected} expected errors)")
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
