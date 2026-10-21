"""
Unit tests of the OpenSearch request conversion (no server is needed)
"""
import pytest

from esorm.opensearch import (_to_body, _SEARCH_BODY, _convert_search_body, _convert_index_body,
                              _convert_mapping_kwargs, _convert_template_kwargs)


class TestOpenSearchConversion:
    """
    OpenSearch request conversion tests
    """

    def test_search_body(self):
        kwargs = _to_body(dict(index='idx', query={'match_all': {}}, from_=10, size=None, sort=[{'f': 'asc'}],
                               source=['f'], _source=None, routing='r', seq_no_primary_term=True), _SEARCH_BODY)
        assert kwargs == {
            'index': 'idx',
            'routing': 'r',
            '_source': None,
            'body': {'query': {'match_all': {}}, 'from': 10, 'sort': [{'f': 'asc'}], '_source': ['f'],
                     'seq_no_primary_term': True},
        }

    def test_whole_body_param(self):
        assert _to_body(dict(index='idx', document={'f': 1}, refresh=True), 'document') == \
               {'index': 'idx', 'body': {'f': 1}, 'refresh': True}
        # The body parameter of the opensearch-py client works too
        assert _to_body(dict(index='idx', body={'f': 1}), 'document') == {'index': 'idx', 'body': {'f': 1}}

    def test_knn_query(self):
        kwargs = {'body': {'query': {'bool': {'must': [
            {'knn': {'field': 'v', 'query_vector': [1.0, 2.0], 'k': 5, 'num_candidates': 50,
                     'filter': [{'term': {'f': 1}}]}},
        ]}}}}
        _convert_search_body(kwargs)
        assert kwargs['body']['query'] == {'bool': {'must': [
            {'knn': {'v': {'vector': [1.0, 2.0], 'k': 5, 'filter': {'bool': {'filter': [{'term': {'f': 1}}]}}}}},
        ]}}

    def test_knn_query_default_k(self):
        kwargs = {'body': {'query': {'knn': {'field': 'v', 'query_vector': [1.0], 'num_candidates': 20}}}}
        _convert_search_body(kwargs)
        assert kwargs['body']['query'] == {'knn': {'v': {'vector': [1.0], 'k': 20}}}

    def test_opensearch_knn_query_untouched(self):
        query = {'knn': {'v': {'vector': [1.0], 'k': 1}}}
        kwargs = {'body': {'query': query}}
        _convert_search_body(kwargs)
        assert kwargs['body']['query'] == query

    def test_top_level_knn(self):
        knn = {'field': 'v', 'query_vector': [1.0], 'k': 3}
        kwargs = {'body': {'knn': knn}}
        _convert_search_body(kwargs)
        assert kwargs['body'] == {'query': {'knn': {'v': {'vector': [1.0], 'k': 3}}}}

        kwargs = {'body': {'knn': [knn], 'query': {'match_all': {}}}}
        _convert_search_body(kwargs)
        assert kwargs['body'] == {'query': {'bool': {'should': [
            {'match_all': {}}, {'knn': {'v': {'vector': [1.0], 'k': 3}}}
        ]}}}

    def test_dense_vector_mapping(self):
        body = {
            'mappings': {'properties': {
                'v': {'type': 'dense_vector', 'dims': 3, 'similarity': 'dot_product', 'index': True},
                'nested': {'properties': {'v2': {'type': 'dense_vector', 'dims': 2}}},
                'f': {'type': 'keyword'},
            }},
            'settings': {'number_of_shards': 1},
        }
        converted = _convert_index_body(body)
        assert converted == {
            'mappings': {'properties': {
                'v': {'type': 'knn_vector', 'dimension': 3, 'space_type': 'innerproduct'},
                'nested': {'properties': {'v2': {'type': 'knn_vector', 'dimension': 2, 'space_type': 'cosinesimil'}}},
                'f': {'type': 'keyword'},
            }},
            'settings': {'number_of_shards': 1, 'index.knn': True},
        }
        # The original body should not be modified
        assert body['mappings']['properties']['v']['type'] == 'dense_vector'
        assert 'index.knn' not in body['settings']

    def test_knn_setting_not_overwritten(self):
        body = {'mappings': {'properties': {'v': {'type': 'dense_vector', 'dims': 3}}},
                'settings': {'index': {'knn': False}}}
        assert _convert_index_body(body)['settings'] == {'index': {'knn': False}}

    def test_no_vectors(self):
        body = {'mappings': {'properties': {'f': {'type': 'keyword'}}}}
        assert _convert_index_body(body) == body

    def test_dense_vector_without_dims(self):
        with pytest.raises(ValueError):
            _convert_index_body({'mappings': {'properties': {'v': {'type': 'dense_vector'}}}})

    def test_put_mapping(self):
        kwargs = {'body': {'properties': {'v': {'type': 'dense_vector', 'dims': 3, 'similarity': 'l2_norm'}}}}
        _convert_mapping_kwargs(kwargs)
        assert kwargs['body'] == {'properties': {'v': {'type': 'knn_vector', 'dimension': 3, 'space_type': 'l2'}}}

    def test_index_template(self):
        kwargs = {'body': {'index_patterns': ['p*'], 'template': {
            'mappings': {'properties': {'v': {'type': 'dense_vector', 'dims': 3}}}
        }}}
        _convert_template_kwargs(kwargs)
        assert kwargs['body']['template']['mappings']['properties']['v']['type'] == 'knn_vector'
        assert kwargs['body']['template']['settings'] == {'index.knn': True}
