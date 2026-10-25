"""
Code blocks of the documentation, for the tests of the examples (tests/test_builder_docs.py, tests/typecheck/check.py)

The blocks after a ``<!-- no-check ... -->`` comment are skipped.
"""
from typing import List, NamedTuple

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# The documentation of the query builder
BUILDER_DOCS = [ROOT / 'docs' / 'query_builder.md', ROOT / 'README.md']

FENCE = re.compile(r'^(?P<indent> *)```(?P<lang>\w+)\s*$')


class CodeBlock(NamedTuple):
    """ A fenced code block """
    file: Path
    line: int
    """ The line of the opening fence """
    lang: str
    code: str
    skip: bool
    """ It is after a no-check comment """

    def __str__(self) -> str:
        return f"{self.file.relative_to(ROOT)}:{self.line}"


def code_blocks(file: Path) -> List[CodeBlock]:
    """ The fenced code blocks of a markdown file """
    lines = file.read_text().splitlines()
    blocks: List[CodeBlock] = []
    prev = ''  # The previous non-empty line
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if m:
            indent = len(m['indent'])
            end = i + 1
            while lines[end].strip() != '```':
                end += 1
            code = '\n'.join(line[indent:] for line in lines[i + 1:end])
            blocks.append(CodeBlock(file, i + 1, m['lang'], code, prev.strip().startswith('<!-- no-check')))
            prev = lines[end]
            i = end + 1
            continue
        if lines[i].strip():
            prev = lines[i]
        i += 1
    return blocks


def builder_blocks() -> List[CodeBlock]:
    """ The python blocks of the documentation which use the query builder (without the skipped ones) """
    return [b for file in BUILDER_DOCS for b in code_blocks(file)
            if b.lang == 'python' and 'esorm.builder' in b.code and not b.skip]


def builder_examples() -> List[CodeBlock]:
    """ The query builder examples: python blocks followed by a json block, which is the result of q.to_query() """
    res = []
    for file in BUILDER_DOCS:
        blocks = code_blocks(file)
        for block, following in zip(blocks, blocks[1:]):
            if block.lang == 'python' and following.lang == 'json' and not block.skip:
                res.append(block)
    return res


def expected_json(block: CodeBlock) -> str:
    """ The json block after an example """
    blocks = code_blocks(block.file)
    return blocks[blocks.index(block) + 1].code
