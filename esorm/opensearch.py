"""
OpenSearch support

OpenSearch is supported through the official `opensearch-py` client (`pip install pyesorm[opensearch]`).

The client is wrapped, so the same code works with both ElasticSearch and OpenSearch:

- the most common API methods accept the keyword arguments of the elasticsearch 9.x client (e.g. `query`, `sort`,
  `document`, `operations`, `mappings`) instead of `body`,
- `dense_vector` mappings are converted to `knn_vector` mappings and ElasticSearch `knn` queries to OpenSearch
  `knn` queries,
- the exceptions of the elasticsearch client are raised (e.g. `elasticsearch.NotFoundError`,
  `elasticsearch.ConflictError`).

Other methods are passed to the opensearch-py client as they are.
"""
from typing import Any, Dict, Tuple, Type, Union, cast

import copy
from functools import wraps

import elastic_transport
import elasticsearch
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig

try:
    import opensearchpy
    from opensearchpy import AsyncOpenSearch, AsyncTransport
    from opensearchpy._async.client.indices import IndicesClient
except ImportError as _e:  # pragma: no cover
    raise ImportError("OpenSearch support requires the opensearch-py package, "
                      "please install it with `pip install pyesorm[opensearch]`!") from _e

__all__ = ['AsyncOpenSearchCompat']

#
# Request conversion
#

# Body parameters of the methods: either the name of the parameter which is the whole body,
# or the names of the body fields and their aliases
_BodyParams = Union[str, Tuple[Tuple[str, ...], Dict[str, str]]]

_SEARCH_BODY: _BodyParams = (
    ('aggregations', 'aggs', 'collapse', 'docvalue_fields', 'explain', 'ext', 'fields', 'from_', 'highlight',
     'indices_boost', 'knn', 'min_score', 'pit', 'post_filter', 'profile', 'query', 'rescore', 'runtime_mappings',
     'script_fields', 'search_after', 'seq_no_primary_term', 'size', 'slice', 'sort', 'source', 'stats',
     'stored_fields', 'suggest', 'terminate_after', 'timeout', 'track_scores', 'track_total_hits', 'version'),
    {'from_': 'from', 'source': '_source'}
)

_SPACE_TYPES = {
    'cosine': 'cosinesimil',
    'dot_product': 'innerproduct',
    'max_inner_product': 'innerproduct',
    'l2_norm': 'l2',
}

_DATA_TYPES = {
    'bit': 'binary',
}


def _to_body(kwargs: Dict[str, Any], body_params: _BodyParams) -> Dict[str, Any]:
    """
    Move body parameters of the elasticsearch client into the body

    :param kwargs: Keyword arguments in elasticsearch client format
    :param body_params: The body parameters of the method
    :return: Keyword arguments in opensearch-py client format
    """
    if isinstance(body_params, str):
        if body_params in kwargs:
            kwargs['body'] = kwargs.pop(body_params)
        return kwargs

    fields, aliases = body_params
    body = dict(kwargs.pop('body', None) or {})
    for name in fields:
        value = kwargs.pop(name, None)
        if value is not None:
            body[aliases.get(name, name)] = value
    if body:
        kwargs['body'] = body
    return kwargs


def _convert_knn(knn: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an ElasticSearch knn query to an OpenSearch one

    :param knn: ElasticSearch knn query, e.g. `{'field': 'v', 'query_vector': [...], 'k': 10}`
    :return: OpenSearch knn query, e.g. `{'v': {'vector': [...], 'k': 10}}`
    """
    knn = dict(knn)
    field = knn.pop('field')
    params: Dict[str, Any] = {'vector': knn.pop('query_vector')}
    num_candidates = knn.pop('num_candidates', None)
    if 'k' not in knn and 'min_score' not in knn and 'max_distance' not in knn:
        params['k'] = num_candidates or 10
    if 'filter' in knn:
        knn_filter = _convert_query(knn.pop('filter'))
        params['filter'] = {'bool': {'filter': knn_filter}} if isinstance(knn_filter, list) else knn_filter
    params.update(knn)
    return {field: params}


def _convert_query(query: Any) -> Any:
    """
    Convert ElasticSearch knn queries to OpenSearch ones recursively

    :param query: The query
    :return: The converted query
    """
    if isinstance(query, list):
        return [_convert_query(q) for q in query]
    if not isinstance(query, dict):
        return query
    return {k: _convert_knn(v) if k == 'knn' and isinstance(v, dict) and 'field' in v and 'query_vector' in v
            else _convert_query(v)
            for k, v in query.items()}


def _convert_search_body(kwargs: Dict[str, Any]):
    """
    Convert the queries in the body of search-like methods

    :param kwargs: Keyword arguments in opensearch-py client format
    """
    body = kwargs.get('body')
    if not isinstance(body, dict):
        return
    body = kwargs['body'] = dict(body)
    for key in ('query', 'post_filter'):
        if key in body:
            body[key] = _convert_query(body[key])
    # OpenSearch has no top level knn search option, so it is added to the query
    if 'knn' in body:
        knn = body.pop('knn')
        queries = [body['query']] if 'query' in body else []
        queries += [{'knn': _convert_knn(k)} for k in (knn if isinstance(knn, list) else [knn])]
        body['query'] = queries[0] if len(queries) == 1 else {'bool': {'should': queries}}


def _convert_properties(properties: Dict[str, Any]) -> bool:
    """
    Convert dense_vector fields to knn_vector fields recursively (in place)

    :param properties: Mapping properties
    :return: True if there are knn_vector fields
    """
    has_knn = False
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        if prop.get('type') == 'dense_vector':
            if 'dims' not in prop:
                raise ValueError(f"OpenSearch needs the dimensions of the `{name}` dense_vector field, "
                                 f"please set it with `DenseVectorField(..., dims=...)`!")
            prop['type'] = 'knn_vector'
            prop['dimension'] = prop.pop('dims')
            similarity = prop.pop('similarity', 'cosine')
            prop['space_type'] = _SPACE_TYPES.get(similarity, similarity)
            if 'element_type' in prop:
                element_type = prop.pop('element_type')
                prop['data_type'] = _DATA_TYPES.get(element_type, element_type)
            # These are not supported by knn_vector
            prop.pop('index', None)
            prop.pop('index_options', None)
        if prop.get('type') == 'knn_vector':
            has_knn = True
        for key in ('properties', 'fields'):
            if isinstance(prop.get(key), dict):
                has_knn = _convert_properties(prop[key]) or has_knn
    return has_knn


def _has_setting(settings: Dict[str, Any], key: str) -> bool:
    """ Check if a setting is specified either in flat ('a.b.c') or in nested form """
    return any(((k == key or k.endswith('.' + key)) and not isinstance(v, dict))
               or (isinstance(v, dict) and _has_setting(v, key))
               for k, v in settings.items())


def _convert_index_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert dense_vector fields in index (or template) definitions and enable knn for the index if needed

    :param body: Index definition with optional `mappings` and `settings`
    :return: The converted index definition
    """
    body = copy.deepcopy(body)
    mappings = body.get('mappings')
    if isinstance(mappings, dict) and isinstance(mappings.get('properties'), dict):
        if _convert_properties(mappings['properties']):
            settings = body.get('settings') or {}
            if not _has_setting(settings, 'knn'):
                body['settings'] = {**settings, 'index.knn': True}
    return body


def _convert_index_kwargs(kwargs: Dict[str, Any]):
    """ Convert the body of index creation """
    if isinstance(kwargs.get('body'), dict):
        kwargs['body'] = _convert_index_body(kwargs['body'])


def _convert_template_kwargs(kwargs: Dict[str, Any]):
    """ Convert the body of index template creation """
    body = kwargs.get('body')
    if isinstance(body, dict) and isinstance(body.get('template'), dict):
        kwargs['body'] = {**body, 'template': _convert_index_body(body['template'])}


def _convert_mapping_kwargs(kwargs: Dict[str, Any]):
    """ Convert the body of mapping update """
    body = kwargs.get('body')
    if isinstance(body, dict) and isinstance(body.get('properties'), dict):
        body = kwargs['body'] = copy.deepcopy(body)
        _convert_properties(body['properties'])


def _compat(base: type, name: str, body_params: _BodyParams, convert=None):
    """
    Create a method which accepts the keyword arguments of the elasticsearch client

    :param base: The opensearch-py client class
    :param name: Name of the method
    :param body_params: The body parameters of the method
    :param convert: Optional function to convert the keyword arguments further
    :return: The new method
    """
    original = getattr(base, name)

    @wraps(original)
    async def method(self, *args, **kwargs):
        kwargs = _to_body(kwargs, body_params)
        if convert is not None:
            convert(kwargs)
        return await original(self, *args, **kwargs)

    return method


#
# Exception conversion
#

class _OpenSearchTransportMixin:
    """
    Transport mixin to raise the exceptions of the elasticsearch client instead of the opensearch-py ones
    """

    async def perform_request(self, *args, **kwargs):
        try:
            # noinspection PyUnresolvedReferences
            return await super().perform_request(*args, **kwargs)  # type: ignore
        except opensearchpy.ConnectionTimeout as e:
            raise elastic_transport.ConnectionTimeout(str(e)) from e
        except opensearchpy.SSLError as e:
            raise elastic_transport.TlsError(str(e)) from e
        except opensearchpy.ConnectionError as e:
            raise elastic_transport.ConnectionError(str(e)) from e
        except opensearchpy.TransportError as e:
            if not isinstance(e.status_code, int):
                raise
            raise self._api_error(e) from e

    def _api_error(self, e: 'opensearchpy.TransportError') -> elasticsearch.ApiError:
        """
        Create the elasticsearch exception from the opensearch-py one

        :param e: The opensearch-py exception
        :return: The elasticsearch exception
        """
        host: Dict[str, Any] = (getattr(self, 'hosts', None) or [{}])[0]
        node = NodeConfig(scheme=host.get('scheme', 'https' if host.get('use_ssl') else 'http'),
                          host=host.get('host', 'localhost'), port=int(host.get('port', 9200)))
        meta = ApiResponseMeta(status=e.status_code, http_version='1.1', headers=HttpHeaders(), duration=0.0,
                               node=node)
        error_class = elasticsearch.exceptions.HTTP_EXCEPTIONS.get(e.status_code, elasticsearch.ApiError)
        return error_class(message=str(e.error), meta=meta, body=e.info)


def _compat_transport_class(transport_class: Type[AsyncTransport]) -> Type[AsyncTransport]:
    """
    Create a transport class which raises the exceptions of the elasticsearch client

    :param transport_class: The original transport class
    :return: The compatible transport class
    """
    if issubclass(transport_class, _OpenSearchTransportMixin):
        return transport_class
    return cast(Type[AsyncTransport], type('Compat' + transport_class.__name__,
                                           (_OpenSearchTransportMixin, transport_class), {}))


#
# Client
#

class _CompatIndicesClient(IndicesClient):
    """
    Indices client which accepts the keyword arguments of the elasticsearch client
    """
    create = _compat(IndicesClient, 'create', (('aliases', 'mappings', 'settings'), {}), _convert_index_kwargs)
    put_mapping = _compat(IndicesClient, 'put_mapping', (
        ('date_detection', 'dynamic', 'dynamic_date_formats', 'dynamic_templates', 'field_names', 'meta',
         'numeric_detection', 'properties', 'routing', 'runtime', 'source'),
        {'field_names': '_field_names', 'meta': '_meta', 'routing': '_routing', 'source': '_source'}
    ), _convert_mapping_kwargs)
    put_settings = _compat(IndicesClient, 'put_settings', 'settings')
    put_index_template = _compat(IndicesClient, 'put_index_template', (
        ('composed_of', 'data_stream', 'index_patterns', 'meta', 'priority', 'template', 'version'),
        {'meta': '_meta'}
    ), _convert_template_kwargs)


class AsyncOpenSearchCompat(AsyncOpenSearch):
    """
    OpenSearch client which accepts the keyword arguments and raises the exceptions of the elasticsearch client
    """

    def __init__(self, hosts: Any = None, transport_class: Type[AsyncTransport] = AsyncTransport, **kwargs: Any):
        super().__init__(hosts, transport_class=_compat_transport_class(transport_class), **kwargs)
        self.indices = _CompatIndicesClient(self)

    index = _compat(AsyncOpenSearch, 'index', 'document')
    create = _compat(AsyncOpenSearch, 'create', 'document')
    update = _compat(AsyncOpenSearch, 'update', (
        ('detect_noop', 'doc', 'doc_as_upsert', 'script', 'scripted_upsert', 'source', 'upsert'),
        {'source': '_source'}
    ))
    bulk = _compat(AsyncOpenSearch, 'bulk', 'operations')
    mget = _compat(AsyncOpenSearch, 'mget', (('docs', 'ids'), {}))
    search = _compat(AsyncOpenSearch, 'search', _SEARCH_BODY, _convert_search_body)
    msearch = _compat(AsyncOpenSearch, 'msearch', 'searches')
    count = _compat(AsyncOpenSearch, 'count', (('query',), {}), _convert_search_body)
    delete_by_query = _compat(AsyncOpenSearch, 'delete_by_query', (('max_docs', 'query', 'slice'), {}),
                              _convert_search_body)
    update_by_query = _compat(AsyncOpenSearch, 'update_by_query',
                              (('conflicts', 'max_docs', 'query', 'script', 'slice'), {}), _convert_search_body)

    async def ping(self, *args, **kwargs) -> bool:
        """
        Returns True if the cluster is up, False otherwise
        """
        try:
            return await super().ping(*args, **kwargs)
        except (elasticsearch.ApiError, elastic_transport.TransportError):
            return False
