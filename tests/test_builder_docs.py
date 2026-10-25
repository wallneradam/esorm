"""
The examples of the query builder documentation must be correct

Every python code block in the documentation which is followed by a json code block is an example: the python block
is executed, and the result of ``q.to_query()`` must be the same as the json. The other python blocks must be valid
python. The blocks after a ``<!-- no-check ... -->`` comment are skipped.

The python blocks are type checked by ``tests/typecheck/check.py`` too.
"""
import json
import runpy
from pathlib import Path

import pytest

from .docs_examples import CodeBlock, builder_blocks, builder_examples, expected_json


@pytest.fixture(autouse=True)
def restore_models():
    """ The examples define models, they must not stay in the model registry """
    from esorm.model import _ESModelMeta
    models = dict(_ESModelMeta.__models__)
    yield
    _ESModelMeta.__models__.clear()
    _ESModelMeta.__models__.update(models)


def test_examples_found():
    examples = builder_examples()
    assert len([b for b in examples if b.file.name == 'query_builder.md']) >= 25
    assert any(b.file.name == 'README.md' for b in examples)


@pytest.mark.parametrize('block', builder_examples(), ids=str)
def test_example(block: CodeBlock, tmp_path: Path):
    # It runs from a file, because the models get their descriptions from their source
    file = tmp_path / 'docs_example.py'
    file.write_text(block.code)
    namespace = runpy.run_path(str(file))
    assert 'q' in namespace, "The example must build the query in the `q` variable"
    # Dates are serialized by the ES client, the json shows them as strings
    result = json.loads(json.dumps(namespace['q'].to_query(), default=str))
    assert result == json.loads(expected_json(block))


@pytest.mark.parametrize('block', builder_blocks(), ids=str)
def test_valid_python(block: CodeBlock):
    compile(block.code, str(block), 'exec')
