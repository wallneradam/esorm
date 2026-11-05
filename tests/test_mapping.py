"""
Tests of the mapping comparison (check_mappings) and the strict mode of setup_mappings
"""
from typing import List
import os

import pytest

from esorm.mapping import MappingDiff, diff_mappings, format_diffs

services = os.environ.get('ESORM_TEST_SERVICES', 'es8x,es9x,os2x,os3x').split(',')


def _kinds(diffs: List[MappingDiff]):
    return sorted((d.kind, d.field, d.param) for d in diffs)


class TestDiffMappings:
    """
    Unit tests of the mapping comparison (no server is needed)
    """

    def test_equal(self):
        expected = {'f_str': {'type': 'keyword'}, 'f_obj': {'properties': {'f_int': {'type': 'long'}}}}
        actual = {'f_str': {'type': 'keyword'}, 'f_obj': {'properties': {'f_int': {'type': 'long'}}}}
        assert diff_mappings('idx', expected, actual) == ([], {})

    def test_defaults_and_server_params(self):
        """ Default values are not returned by the server, and the server adds parameters the model doesn't set """
        expected = {
            'f_text': {'type': 'text', 'analyzer': 'standard', 'index': True, 'search_analyzer': 'standard'},
            'f_vector': {'type': 'dense_vector', 'index': 'true', 'similarity': 'cosine'},
        }
        actual = {
            'f_text': {'type': 'text'},
            'f_vector': {'type': 'dense_vector', 'dims': 3, 'index': True, 'similarity': 'cosine',
                         'index_options': {'type': 'int8_hnsw', 'm': 16}},
        }
        assert diff_mappings('idx', expected, actual) == ([], {})

    def test_opensearch_space_type_in_method(self):
        expected = {'f_vector': {'type': 'knn_vector', 'dimension': 3, 'space_type': 'cosinesimil'}}
        actual = {'f_vector': {'type': 'knn_vector', 'dimension': 3,
                               'method': {'engine': 'faiss', 'space_type': 'cosinesimil', 'name': 'hnsw'}}}
        assert diff_mappings('idx', expected, actual) == ([], {})

    def test_new_fields(self):
        expected = {
            'f_str': {'type': 'keyword'},
            'f_new': {'type': 'long'},
            'f_obj': {'properties': {'f_int': {'type': 'long'}, 'f_new': {'type': 'keyword'}}},
            'f_text': {'type': 'text', 'analyzer': 'english', 'fields': {'raw': {'type': 'keyword'}}},
        }
        actual = {
            'f_str': {'type': 'keyword'},
            'f_obj': {'properties': {'f_int': {'type': 'long'}}},
            'f_text': {'type': 'text', 'analyzer': 'english'},
        }
        diffs, patch = diff_mappings('idx', expected, actual)
        assert _kinds(diffs) == [('new_field', 'f_new', None), ('new_field', 'f_obj.f_new', None),
                                 ('new_field', 'f_text.raw', None)]
        # Only the changes are sent, changed fields with all of their parameters
        assert patch == {
            'f_new': {'type': 'long'},
            'f_obj': {'properties': {'f_new': {'type': 'keyword'}}},
            'f_text': {'type': 'text', 'analyzer': 'english', 'fields': {'raw': {'type': 'keyword'}}},
        }

    def test_update(self):
        expected = {'f_text': {'type': 'text', 'analyzer': 'english', 'search_analyzer': 'simple'}}
        actual = {'f_text': {'type': 'text', 'analyzer': 'english'}}
        diffs, patch = diff_mappings('idx', expected, actual)
        assert diffs == [MappingDiff('idx', 'update', 'f_text', 'search_analyzer', 'simple', 'english')]
        assert patch == expected

    def test_conflicts(self):
        expected = {
            'f_type': {'type': 'text'},
            'f_object': {'type': 'nested', 'properties': {'f_int': {'type': 'long'}}},
            'f_index': {'type': 'keyword', 'index': False},
            'f_analyzer': {'type': 'text', 'analyzer': 'english'},
            'f_sub': {'type': 'text', 'fields': {'raw': {'type': 'keyword'}}},
        }
        actual = {
            'f_type': {'type': 'keyword'},
            'f_object': {'properties': {'f_int': {'type': 'long'}}},
            'f_index': {'type': 'keyword'},
            'f_analyzer': {'type': 'text'},
            'f_sub': {'type': 'text', 'fields': {'raw': {'type': 'text'}}},
        }
        diffs, patch = diff_mappings('idx', expected, actual)
        assert _kinds(diffs) == [('conflict', 'f_analyzer', 'analyzer'), ('conflict', 'f_index', 'index'),
                                 ('conflict', 'f_object', 'type'), ('conflict', 'f_sub.raw', 'type'),
                                 ('conflict', 'f_type', 'type')]
        assert all(d.is_conflict for d in diffs)
        assert patch == {}
        analyzer_diff = next(d for d in diffs if d.field == 'f_analyzer')
        assert analyzer_diff.expected == 'english' and analyzer_diff.actual == 'standard'

    def test_conflict_with_compatible_changes(self):
        """ Compatible changes of a field are applied, the conflicting parameters are kept as in the index """
        expected = {'f_text': {'type': 'text', 'analyzer': 'english', 'fields': {'raw': {'type': 'keyword'}}}}
        actual = {'f_text': {'type': 'text', 'analyzer': 'standard'}}
        diffs, patch = diff_mappings('idx', expected, actual)
        assert _kinds(diffs) == [('conflict', 'f_text', 'analyzer'), ('new_field', 'f_text.raw', None)]
        assert patch == {'f_text': {'type': 'text', 'analyzer': 'standard', 'fields': {'raw': {'type': 'keyword'}}}}

    def test_extra(self):
        expected = {'f_str': {'type': 'keyword'}, 'f_text': {'type': 'text'}}
        actual = {'f_str': {'type': 'keyword'}, 'f_old': {'type': 'long'},
                  'f_text': {'type': 'text', 'fields': {'keyword': {'type': 'keyword', 'ignore_above': 256}}}}
        diffs, patch = diff_mappings('idx', expected, actual)
        assert _kinds(diffs) == [('extra', 'f_old', None), ('extra', 'f_text.keyword', None)]
        assert patch == {}

    def test_format(self):
        diffs = [MappingDiff('idx', 'new_index'),
                 MappingDiff('idx', 'new_field', 'f_new', expected={'type': 'long'}),
                 MappingDiff('idx', 'update', 'f_text', 'search_analyzer', 'simple', 'standard'),
                 MappingDiff('idx', 'conflict', 'f_str', 'type', 'text', 'keyword'),
                 MappingDiff('idx', 'extra', 'f_old', actual={'type': 'long'})]
        assert format_diffs(diffs).splitlines() == [
            "  - [new_index] `idx`: the index does not exist",
            "  - [new_field] `idx` field `f_new`: new field (long)",
            "  - [update] `idx` field `f_text`: `search_analyzer` can be updated: 'standard' -> 'simple'",
            "  - [conflict] `idx` field `f_str`: `type` is 'keyword' in the index, but 'text' in the model "
            "(needs a new index and reindexing)",
            "  - [extra] `idx` field `f_old`: the field is in the index, but not in the model",
        ]


@pytest.mark.parametrize('service', services, scope='class')
class TestMappingCheck:
    """
    Tests of check_mappings and setup_mappings with existing indices
    """

    @pytest.fixture
    async def cleanup(self, es, esorm):
        """ Remove the models and indices created by the test, so they don't affect other tests """
        indices = []
        yield indices
        for index in indices:
            esorm.model._ESModelMeta.__models__.pop(index, None)
            for name in [index, index + '_v1']:
                try:
                    await es.indices.delete(index=name)
                except Exception:
                    pass

    async def test_matching_mappings(self, es, esorm, cleanup):
        """
        The mappings created from the models match the models, even with default and server added parameters
        """
        from enum import Enum
        from datetime import datetime
        from typing import Optional, List

        class Color(Enum):
            RED = 'red'

        class Sub(esorm.ESBaseModel):
            f_str: str
            f_int: int

        class MatchingModel(esorm.ESModel):
            class ESConfig:
                index = 'esorm_mapping_matching'

            f_str: str
            f_text: esorm.fields.text = esorm.fields.TextField(..., keyword=True, analyzer='standard',
                                                               search_analyzer='standard')
            f_english: esorm.fields.text = esorm.Field(..., analyzer='english', fields={
                'raw': {'type': 'keyword', 'ignore_above': 256}})
            f_not_indexed: Optional[esorm.fields.keyword] = esorm.Field(None, index=False)
            f_binary: Optional[esorm.fields.binary] = esorm.Field(None, index=False)
            f_datetime: datetime
            f_color: Color
            f_sub: Sub
            f_sub_list: List[Sub] = []
            f_vector: esorm.fields.dense_vector = esorm.DenseVectorField(..., dims=3, similarity='l2_norm')
            f_geo: Optional[esorm.fields.geo_point] = None

        cleanup.append('esorm_mapping_matching')

        assert _kinds(await esorm.check_mappings()) == [('new_index', None, None)]
        await esorm.setup_mappings()
        assert await esorm.check_mappings() == []

        # Saving a document doesn't change it
        await MatchingModel(f_str='a', f_text='b', f_english='c', f_datetime=datetime.now(), f_color=Color.RED,
                            f_sub=Sub(f_str='d', f_int=1), f_vector=[1.0, 2.0, 3.0]).save(refresh=True)
        assert await esorm.check_mappings() == []
        await esorm.setup_mappings()

    async def test_dense_vector_without_dims(self, es, esorm, cleanup):
        """
        The dimensions of a dense vector are set by ElasticSearch from the first document, it is not a conflict
        """
        if esorm.is_opensearch():
            pytest.skip("OpenSearch needs the dimensions")

        class VectorModel(esorm.ESModel):
            class ESConfig:
                index = 'esorm_mapping_vector'

            f_vector: esorm.fields.dense_vector

        cleanup.append('esorm_mapping_vector')
        await esorm.setup_mappings()
        await VectorModel(f_vector=[1.0, 2.0, 3.0]).save(refresh=True)
        assert await esorm.check_mappings() == []
        await esorm.setup_mappings()

    async def test_new_fields(self, es, esorm, cleanup):
        """
        New fields and subfields are added to the existing index
        """
        index = 'esorm_mapping_new_fields'
        cleanup.append(index)

        class Sub(esorm.ESBaseModel):
            f_int: int

        class NewFieldsV1(esorm.ESModel):
            class ESConfig:
                index = 'esorm_mapping_new_fields'

            f_text: esorm.fields.text = esorm.Field(..., analyzer='english')
            f_sub: Sub

        await esorm.setup_mappings()

        class Sub2(esorm.ESBaseModel):
            f_int: int
            f_str: str

        class NewFieldsV2(esorm.ESModel):
            class ESConfig:
                index = 'esorm_mapping_new_fields'

            f_text: esorm.fields.text = esorm.Field(..., analyzer='english', search_analyzer='simple',
                                                    fields={'raw': {'type': 'keyword'}})
            f_sub: Sub2
            f_new: int = 0

        assert _kinds(await esorm.check_mappings()) == [
            ('new_field', 'f_new', None), ('new_field', 'f_sub.f_str', None), ('new_field', 'f_text.raw', None),
            ('update', 'f_text', 'search_analyzer'),
        ]
        await esorm.setup_mappings()
        assert await esorm.check_mappings() == []

        properties = (await es.indices.get_mapping(index=index))[index]['mappings']['properties']
        assert properties['f_new'] == {'type': 'long'}
        assert properties['f_sub']['properties']['f_str'] == {'type': 'keyword'}
        assert properties['f_text']['fields'] == {'raw': {'type': 'keyword'}}
        assert properties['f_text']['search_analyzer'] == 'simple'
        assert properties['f_text']['analyzer'] == 'english'

    async def test_conflict(self, es, esorm, cleanup):
        """
        Incompatible changes raise an error in strict mode, before any change, the compatible changes are applied
        in non-strict mode
        """
        index = 'esorm_mapping_conflict'
        cleanup.append(index)

        class ConflictV1(esorm.ESModel):
            class ESConfig:
                index = 'esorm_mapping_conflict'

            f_str: str

        await esorm.setup_mappings()

        class ConflictV2(esorm.ESModel):
            class ESConfig:
                index = 'esorm_mapping_conflict'

            f_str: esorm.fields.text
            f_new: int = 0

        diffs = await esorm.check_mappings()
        assert _kinds(diffs) == [('conflict', 'f_str', 'type'), ('new_field', 'f_new', None)]

        with pytest.raises(esorm.MappingConflictError) as exc_info:
            await esorm.setup_mappings()
        assert exc_info.value.diffs == [d for d in diffs if d.is_conflict]
        assert "`esorm_mapping_conflict` field `f_str`: `type` is 'keyword' in the index, but 'text'" \
               in str(exc_info.value)
        assert isinstance(exc_info.value, esorm.MappingError)
        # Nothing is changed in strict mode
        properties = (await es.indices.get_mapping(index=index))[index]['mappings']['properties']
        assert 'f_new' not in properties

        # The compatible changes are applied in non-strict mode
        await esorm.setup_mappings(strict=False)
        properties = (await es.indices.get_mapping(index=index))[index]['mappings']['properties']
        assert properties['f_new'] == {'type': 'long'}
        assert properties['f_str'] == {'type': 'keyword'}
        assert _kinds(await esorm.check_mappings()) == [('conflict', 'f_str', 'type')]

    async def test_mapping_error(self, es, esorm, cleanup):
        """
        Errors from the server raise MappingError in strict mode
        """
        index = 'esorm_mapping_error'
        cleanup.append(index)

        class ErrorModel(esorm.ESModel):
            class ESConfig:
                index = 'esorm_mapping_error'

            f_text: esorm.fields.text = esorm.Field(..., analyzer='no_such_analyzer')

        with pytest.raises(esorm.MappingError, match='Could not create the index `esorm_mapping_error`'):
            await esorm.setup_mappings()

        await esorm.setup_mappings(strict=False)
        assert not await es.indices.exists(index=index)

    async def test_alias(self, es, esorm, cleanup):
        """
        The index of the model can be an alias
        """
        index = 'esorm_mapping_alias'
        cleanup.append(index)
        await es.indices.create(index=index + '_v1', mappings={'properties': {'f_str': {'type': 'keyword'}}},
                                aliases={index: {}})

        class AliasIndexModel(esorm.ESModel):
            class ESConfig:
                index = 'esorm_mapping_alias'

            f_str: str
            f_new: int = 0

        assert _kinds(await esorm.check_mappings()) == [('new_field', 'f_new', None)]
        assert (await esorm.check_mappings())[0].index == index + '_v1'
        await esorm.setup_mappings()
        assert await esorm.check_mappings() == []

        doc_id = await AliasIndexModel(f_str='a', f_new=1).save(refresh=True)
        assert (await AliasIndexModel.get(doc_id)).f_new == 1
