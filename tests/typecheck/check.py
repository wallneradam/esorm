"""
Type checking tests

The type hints are tested with pyright (used by VS Code / Pylance) and mypy:

- the ``ok_*.py`` files show correct usage, nothing may be reported in them
- in the ``errors*.py`` files every line marked with ``# E: <reason>`` must be reported, and nothing else
- the python examples of the query builder documentation are correct usage too, they are collected into a temporary
  ``ok_docs.py`` file (every example is the body of a function), the problems are reported at the markdown lines

Run it from the project root: ``python tests/typecheck/check.py``
"""
from typing import Dict, List, Optional, Set, Tuple

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

MARKER = re.compile(r'#\s*E:')
MYPY_LINE = re.compile(r'^(?P<file>[^:]+):(?P<line>\d+):(?:\d+:)? (?P<severity>error|warning): (?P<message>.*)$')

# Diagnostics: file -> line -> messages
Diagnostics = Dict[str, Dict[int, List[str]]]
# The origins of the lines of generated files: file -> line -> "markdown file:line"
Origins = Dict[str, Dict[int, str]]


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


def docs_examples(directory: Path) -> Tuple[Path, Origins]:
    """
    The python examples of the query builder documentation in one file, every example is the body of a function

    :return: The file, and the origins of its lines
    """
    sys.path.insert(0, str(ROOT))
    from tests.docs_examples import builder_blocks

    lines = ['# pyright: standard', '""" Generated from the documentation by check.py """', '']
    origins: Dict[int, str] = {}
    for n, block in enumerate(builder_blocks()):
        lines.append(f'async def _example_{n}() -> None:')
        for i, code_line in enumerate(block.code.splitlines()):
            lines.append(f'    {code_line}' if code_line else '')
            origins[len(lines)] = f"{block.file.relative_to(ROOT)}:{block.line + 1 + i}"
        lines.append('')
    file = directory / 'ok_docs.py'
    file.write_text('\n'.join(lines))
    return file, {str(file.resolve()): origins}


def check(tool: str, diags: Diagnostics, files: List[Path], origins: Optional[Origins] = None) -> List[str]:
    """ Compare the reported and the expected lines """
    problems = []
    for file in files:
        lines = file.read_text().splitlines()
        reported = diags.get(str(file.resolve()), {})
        expected = expected_lines(file)
        rel = file.relative_to(ROOT)
        origin = (origins or {}).get(str(file.resolve()), {})
        for line in sorted(set(reported) - expected):
            for msg in reported[line]:
                problems.append(f"{tool}: {origin.get(line, f'{rel}:{line}')}: unexpected: {msg}")
        for line in sorted(expected - set(reported)):
            problems.append(f"{tool}: {rel}:{line}: not reported: {lines[line - 1].strip()}")
    return problems


def check_stubs() -> List[str]:
    """
    The generated stubs must be up to date, and the parameters of the method descriptors in _model_methods.pyi must
    be the same as the parameters of the runtime methods
    """
    import ast
    import inspect
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'tools'))
    import generate_stubs
    from esorm import ESModel

    problems = []
    for module in generate_stubs.MODULES:
        stub_file = ROOT / 'esorm' / f'{module.name}.pyi'
        if not stub_file.exists() or stub_file.read_text() != generate_stubs.generate(module):
            problems.append(f"{stub_file.relative_to(ROOT)} is not up to date, run `python tools/generate_stubs.py`")

    stub = ast.parse((ROOT / 'esorm' / '_model_methods.pyi').read_text())
    classes = {n.name: n for n in stub.body if isinstance(n, ast.ClassDef)}
    for method, cls_name in (('search', '_Search'), ('search_one', '_SearchOne'), ('scan', '_Scan'),
                             ('scan_batches', '_ScanBatches')):
        runtime = [p for p in inspect.signature(getattr(ESModel, method)).parameters]
        for call in (n for n in classes[cls_name].body if isinstance(n, ast.FunctionDef) and n.name == '__call__'):
            args = call.args
            params = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs][1:]  # without self
            if args.kwarg:
                params.append(args.kwarg.arg)
            params = [p for p in params if p != '_pycharm_workaround']
            if params != runtime:
                problems.append(f"_model_methods.pyi: the parameters of {cls_name}.__call__ {params} are not the same "
                                f"as the parameters of ESModel.{method} {runtime}")
    return problems


def main() -> int:
    files = sorted(HERE.glob('ok_*.py')) + sorted(HERE.glob('errors*.py'))
    problems: List[str] = check_stubs()
    summary: List[Tuple[str, int]] = []
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        docs_file, origins = docs_examples(Path(tmp))
        files.append(docs_file)
        for tool, runner in (('pyright', run_pyright), ('mypy', run_mypy)):
            diags = runner(files)
            tool_problems = check(tool, diags, files, origins)
            problems += tool_problems
            summary.append((tool, len(tool_problems)))
        n_expected = sum(len(expected_lines(f)) for f in files)

    for problem in problems:
        print(problem)
    for tool, n in summary:
        print(f"{tool}: {'OK' if n == 0 else f'{n} problems'} ({len(files)} files, {n_expected} expected errors)")
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
