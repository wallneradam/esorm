"""
This module contains the ESModel classes and related functions
"""
from typing import (TypeVar, Any, Dict, Optional, Tuple, Type, Union, get_args, get_origin, List, Callable,
                    Awaitable, Literal)
from typing_extensions import TypedDict

import asyncio
import ast
import inspect
import textwrap
import traceback
from contextvars import ContextVar
from datetime import datetime, date, time
from functools import wraps

import elasticsearch
from elasticsearch import NotFoundError as ElasticNotFoundError, ConflictError as ElasticConflictError
from pydantic import main as pydantic_main
from pydantic import BaseModel, ConfigDict
from pydantic.fields import Field, PrivateAttr
# noinspection PyProtectedMember
from pydantic.fields import FieldInfo  # It is just not in __all__ of pydantic.fields, but we strongly need it

from .utils import snake_case, utcnow
from .aggs import ESAggs, ESAggsResponse

from .error import InvalidResponseError, NotFoundError
from .esorm import es
from .query import ESQuery
from .response import ESResponse

from .logger import logger

__all__ = [
    'TModel',
    'ESBaseModel',
    'ESModel',
    'ESModelTimestamp',
    'Pagination', 'Sort',
    'setup_mappings',
    'create_index_template',
    'set_default_index_prefix',
    'set_max_lazy_property_concurrency',
    'lazy_property',
    'retry_on_conflict'
]

#
# Global variables and types
#

# noinspection PyProtectedMember
_model_construction = getattr(pydantic_main, '_model_construction')
ModelMetaclass = _model_construction.ModelMetaclass

_default_index_prefix = 'esorm'

# Map python types to ES type
_pydantic_type_map = {
    str: 'keyword',  # Str is defaulted to keyword
    int: 'long',
    float: 'double',
    bool: 'boolean',
    datetime: 'date',
    date: 'date',
    time: 'date',
}

# TModel type variable
TModel = TypeVar('TModel', bound='ESModel')

# Context variable to store the current recursion depth of lazy properties
_lazy_property_recursion_depth: ContextVar[int] = ContextVar('_lazy_property_recursion_depth', default=0)
_lazy_semaphore_concurrency = 16
_lazy_property_semaphore: ContextVar[asyncio.Semaphore] = ContextVar('_lazy_property_semaphore',
                                                                     default=asyncio.Semaphore(
                                                                         _lazy_semaphore_concurrency))


#
# Monkey patching to support field descriptions from docstrings
#

def _description_from_docstring(model: Type[BaseModel]):
    """
    Set undefined field descriptions from variable docstrings

    :param model: The model to set the descriptions
    """
    try:
        source = textwrap.dedent(inspect.getsource(model))
        module = ast.parse(source)
        assert isinstance(module, ast.Module)
        class_def = module.body[0]
        assert isinstance(class_def, ast.ClassDef)
        if len(class_def.body) < 2:
            return
    except OSError:
        return

    for last, node in zip(class_def.body, class_def.body[1:]):
        try:
            if not (isinstance(last, ast.AnnAssign) and isinstance(last.target, ast.Name) and
                    isinstance(node, ast.Expr)):
                continue

            info = model.model_fields[last.target.id]
            if info.description is not None:
                continue

            doc_node = node.value
            if isinstance(doc_node, ast.Constant):  # 'regular' variable doc string
                docstring = doc_node.value.strip()
            else:
                raise NotImplementedError(doc_node)

            info.description = docstring

        except KeyError:
            pass


def _patch_set_model_fields():
    """
    Monkey patchon _model_construction.set_model_fields to set undefined field descriptions from docstrings
    """
    orig_set_model_fields = _model_construction.set_model_fields

    def set_model_fields(model: Type[BaseModel], bases: Tuple[Type[Any], ...], config_wrapper: Any,
                         types_namespace: Dict[str, Any]) -> None:
        orig_set_model_fields(model, bases, config_wrapper, types_namespace)
        _description_from_docstring(model)

    _model_construction.set_model_fields = set_model_fields


_patch_set_model_fields()


#
# ElasticSearch models
#

def set_default_index_prefix(default_index_prefix: str):
    """
    Set default index prefix we use for model and index creation

    :param default_index_prefix: The default index prefix
    """
    global _default_index_prefix
    _default_index_prefix = default_index_prefix


class _ESModelMeta(ModelMetaclass):
    """
    ESModel Metaclass
    """

    # All model classes collected
    __models__: Dict[str, Type['ESModel']] = {}

    # noinspection PyUnresolvedReferences
    def __new__(cls: Type[ModelMetaclass], name: str, bases: Tuple[type, ...],
                namespace: Dict[str, Any], **kwds: Any):
        model: Type[BaseModel] = super().__new__(cls, name, bases, namespace, **kwds)
        if name not in ("ESModel", "ESModelTimestamp", "ESBaseModel"):
            is_model = issubclass(model, ESModel)

            # ESConfig inheritance
            m_dict = {k: v for k, v in ESModel.ESConfig.__dict__.items() if k[0] != '_'}
            if bases and 'ESConfig' in bases[0].__dict__:
                m_dict.update({k: v for k, v in bases[0].ESConfig.__dict__.items() if k[0] != '_'})
            del m_dict['index']  # It is only allowed to be set on the actual model class
            if 'ESConfig' in model.__dict__:
                m_dict.update({k: v for k, v in model.ESConfig.__dict__.items() if k[0] != '_'})
            m_dict['_lazy_properties'] = {}

            # Create (new) ESConfig class inside the class
            model.ESConfig = type('ESConfig', (object,), dict(m_dict))

            # Set default index name if not already set
            if is_model and not getattr(model.ESConfig, 'index', None):
                # Default index is the name of the class in snake_case
                model.ESConfig.index = _default_index_prefix + '-' + snake_case(name)

            # If there is an 'id' field, set it as id_field
            if is_model and 'id' in model.model_fields.keys():
                model.ESConfig.id_field = 'id'

            # Add to models
            if is_model:
                cls.__models__[model.ESConfig.index] = model

            # Collect lazy properties
            for attr_name, attr in namespace.items():
                # Support computed fields
                if attr.__class__.__name__ == 'PydanticDescriptorProxy':
                    attr = getattr(attr, 'wrapped')
                # Is it a lazy property?
                if isinstance(attr, property) and hasattr(attr.fget, '__lazy_property__'):
                    # noinspection PyProtectedMember
                    model.ESConfig._lazy_properties[attr_name] = getattr(attr.fget, '__lazy_property__')

        return model


class ESBaseModel(BaseModel, metaclass=_ESModelMeta):
    """
    Base class for Elastic

    It is useful for nested models, if you don't need the model in ES mappings
    """

    class ESConfig:
        """
        ESBaseModel Config

        This is just for lazy properties, to make ESBasemodel compatible with them
        """

        lazy_property_max_recursion_depth: int = 1
        """ Maximum recursion depth of lazy properties """

        _lazy_properties: Dict[str, Callable[[], Awaitable[Any]]] = {}
        """ Lazy property async function definitions """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        populate_by_name=True,
        arbitrary_types_allowed=True,
        ser_json_bytes='base64',
        validate_assignment=True,
    )

    async def calc_lazy_properties(self):
        """
        (re)Calculate lazy properties
        """
        _lazy_semaphore = _lazy_property_semaphore.get()
        # noinspection PyProtectedMember
        for attr_name, attr in self.ESConfig._lazy_properties.items():
            async with _lazy_semaphore:
                # If we use create_task, this creates a new context, so changed contextvars live only upward
                res = await asyncio.create_task(attr(self))
                setattr(self, '_' + attr_name, res)

        # Calc lazy properties for nested models
        for k, v in self.__dict__.items():
            if isinstance(v, ESBaseModel):
                await v.calc_lazy_properties()


class ESModel(ESBaseModel):
    """
    ElasticSearch Base Model
    """

    _id: Optional[str] = PrivateAttr(None)
    """ The ES id of the document (it is always a string) """

    _routing: Optional[str] = PrivateAttr(None)
    """ The routing of the document """

    _version: Optional[int] = PrivateAttr(None)
    """ The version of the document """

    _primary_term: Optional[int] = PrivateAttr(None)
    """ The primary term of the document """

    _seq_no: Optional[int] = PrivateAttr(None)
    """ The sequence number of the document """

    class ESConfig:
        """ ESModel Config """
        index: Optional[str] = None
        """ The index name """

        id_field: Optional[str] = None
        """ The name of the 'id' field """

        default_sort: Optional[List[Dict[str, Dict[str, str]]]] = None
        """ Default sort """

        settings: Optional[Dict[str, Any]] = None
        """ Index settings """

        lazy_property_max_recursion_depth: int = 1
        """ Maximum recursion depth of lazy properties """

        _lazy_properties: Dict[str, Callable[[], Awaitable[Any]]] = {}
        """ Lazy property async function definitions """

    @property
    def __id__(self) -> str:
        """
        The id of the document

        This can be overridden to make computed ids

        :return: The id of the document
        """
        return getattr(self, self.ESConfig.id_field or '_id')

    @property
    def __routing__(self) -> Optional[str]:
        """
        Shard route name

        :return: Shard route name
        """
        return None

    @classmethod
    async def call(cls, method_name, *, wait_for=None, **kwargs) -> dict:
        """
        Call an elasticsearch method

        This is a low level ES method call, it is not recommended to use this directly.

        :param method_name: The name of the method to call
        :param wait_for: Waits for all shards to sync before returning response
        :param kwargs: The arguments to pass to the method
        :return: The result dictionary from ElasticSearch
        """
        kwargs = dict(kwargs)
        method = getattr(es, method_name)
        index = cls.ESConfig.index
        if wait_for is not None:
            kwargs['refresh'] = "wait_for"
        if 'request_timeout' not in kwargs:
            kwargs['request_timeout'] = 60

        return await method(index=index, **kwargs)

    def to_es(self, **kwargs) -> dict:
        """
        Generates a dictionary equivalent to what ElasticSearch returns in the '_source' property of a response.

        It automatically removes the id field from the document if it is set in ESConfig.id_field to prevent
        duplication of the id field.

        :param kwargs: Pydantic's model_dump parameters
        :return: The dictionary for ElasticSearch
        """
        kwargs = dict(kwargs)

        def recursive_exclude(m) -> Dict[str, Union[bool, dict]]:
            """ Recursively exclude computed fields """
            _exclude: Dict[str, Union[bool, dict]] = {k: True for k in m.model_computed_fields.keys()}
            for k, v in m:
                if k in _exclude:
                    continue
                if isinstance(v, BaseModel):
                    res = recursive_exclude(v)
                    if res:
                        _exclude[k] = res
            return _exclude

        # Update exclude field with computed fields
        exclude = kwargs.get('exclude', {})
        exclude.update(recursive_exclude(self))
        kwargs['exclude'] = exclude
        # Dump model to dict
        d = self.model_dump(**kwargs)

        def recursive_convert(_d: dict):
            """ Recursively modify data for Elasticsearch """
            for k, v in _d.items():
                # Encode datetime fields
                if isinstance(v, datetime):
                    # Update ESTimestamp fields
                    if k == 'modified_at' and d != _d:
                        v = utcnow()
                    elif k == 'created_at' and v is None and d != _d:
                        v = utcnow()
                    _d[k] = v.replace(tzinfo=None).isoformat() + 'Z'
                # Convert subclasses
                elif isinstance(v, dict):
                    recursive_convert(v)

        recursive_convert(d)
        return d

    def update_from_es(self, data: Dict[str, Any]):
        """
        Update the model from ElasticSearch data

        :param data: The ElasticSearch data
        :raises esorm.error.InvalidResponseError: Returned when _id or _source is missing from data
        """
        if not data:
            return None

        source: Optional[dict] = data.get("_source", None)
        # Get id field
        _id = data.get("_id", None)
        if not source or not _id:
            raise InvalidResponseError

        for k, v in source.items():
            if k in self.__fields_set__:
                print(k, '=', v)
                setattr(self, k, v)

        # Set routing field
        _routing = data.get("_routing", None)
        setattr(self, '_routing', _routing)

        # Set version field
        _version = data.get("_version", None)
        setattr(self, '_version', _version)
        # Set primary term field
        _primary_term = data.get("_primary_term", None)
        setattr(self, '_primary_term', _primary_term)
        # Set seq_no field
        _seq_no = data.get("_seq_no", None)
        setattr(self, '_seq_no', _seq_no)

    @classmethod
    def from_es(cls: Type[TModel], data: Dict[str, Any]) -> Optional[TModel]:
        """
        Returns an ESModel from an elasticsearch document that has _id, _source

        :param data: Elasticsearch document that has _id, _source
        :raises esorm.error.InvalidResponseError: Returned when _id or _source is missing from data
        :return: The ESModel instance
        """
        if not data:
            return None

        source: Optional[dict] = data.get("_source", None)
        # Get id field
        _id = data.get("_id", None)
        if not source or not _id:
            raise InvalidResponseError

        # Add id field to document
        if source is not None and cls.ESConfig.id_field:
            source[cls.ESConfig.id_field] = _id
        obj = cls(**source)

        # Set id field
        setattr(obj, '_id', _id)

        # Set routing field
        _routing = data.get("_routing", None)
        setattr(obj, '_routing', _routing)

        # Set version field
        _version = data.get("_version", None)
        setattr(obj, '_version', _version)
        # Set primary term field
        _primary_term = data.get("_primary_term", None)
        setattr(obj, '_primary_term', _primary_term)
        # Set seq_no field
        _seq_no = data.get("_seq_no", None)
        setattr(obj, '_seq_no', _seq_no)

        return obj

    async def save(self, *, wait_for=False, pipeline: Optional[str] = None, routing: Optional[str] = None) -> str:
        """
        Save document into elasticsearch.

        If document already exists, existing document will be updated as per native elasticsearch index operation.
        If model has id (Config.id_field or __id__), this will be used as the elasticsearch _id. The id field will be
        removed from the document before indexing.
        If no id is provided, then document will be indexed and elasticsearch will generate a suitable id that will be
        populated on the returned model.

        :param wait_for: Waits for all shards to sync before returning response - useful when writing
                         tests. Defaults to False.
        :param pipeline: Pipeline to use for indexing
        :param routing: Shard routing value
        :return: The new document's ID, it is always a string, even if the id field is an integer
        """
        kwargs = dict(
            document=self.to_es(),
            wait_for=wait_for,
        )

        kwargs['id'] = self.__id__
        if self.ESConfig.id_field:
            del kwargs['document'][self.ESConfig.id_field]

        if pipeline is not None:
            kwargs['pipeline'] = pipeline

        if routing is not None:
            kwargs['routing'] = routing
        else:
            kwargs['routing'] = self.__routing__

        if self._primary_term is not None:
            kwargs['if_primary_term'] = self._primary_term
        if self._seq_no is not None:
            kwargs['if_seq_no'] = self._seq_no

        es_res = await self.call('index', **kwargs)

        # Update private fields
        self._id = es_res.get('_id', None)
        self._version = es_res.get('_version', None)
        self._primary_term = es_res.get('_primary_term', None)
        self._seq_no = es_res.get('_seq_no', None)
        # Return the new document's ID
        return self._id

    # noinspection PyShadowingBuiltins
    @classmethod
    async def get(cls: Type[TModel], id: Union[str, int, float], *, routing: Optional[str] = None) -> TModel:
        """
        Fetches document and returns ESModel instance populated with properties.

        :param id: Document id
        :param routing: Shard routing value
        :raises esorm.error.NotFoundError: Returned if document not found
        :return: ESModel object
        """
        kwargs = {'id': id, 'routing': routing if routing else cls.__routing__}
        try:
            es_res = await cls.call('get', **kwargs)
            return await _lazy_process_results(cls.from_es(es_res))
        except ElasticNotFoundError:
            raise NotFoundError(f"Document with id {id} not found")

    async def delete(self, *, wait_for=False, routing: Optional[str] = None):
        """
        Deletes document from ElasticSearch.

        :param wait_for: Waits for all shards to sync before returning response - useful when writing
                         tests. Defaults to False.
        :param routing: Shard routing value
        :raises esorm.error.NotFoundError: Returned if document not found
        :raises ValueError: Returned when id attribute missing from instance
        """
        kwargs = dict(id=self.__id__)
        if self._primary_term is not None:
            kwargs['if_primary_term'] = self._primary_term
        if self._seq_no is not None:
            kwargs['if_seq_no'] = self._seq_no
        try:
            await self.call('delete', wait_for=wait_for,
                            routing=routing if routing is not None else self.__routing__,
                            **kwargs)
        except ElasticNotFoundError:
            raise NotFoundError(f"Document with id {self.__id__} not found!")

    async def reload(self, *, routing: Optional[str] = None) -> TModel:
        """
        Reloads the document from ElasticSearch

        :param routing: Shard routing value
        :raises esorm.error.NotFoundError: Returned if document not found
        """
        kwargs = dict(id=self.__id__, routing=routing if routing is not None else self.__routing__)
        try:
            es_res = await self.call('get', **kwargs)
            self.update_from_es(es_res)
            return await _lazy_process_results(self)
        except ElasticNotFoundError:
            raise NotFoundError(f"Document with id {id} not found")

    @classmethod
    async def _search(cls: Type[TModel],
                      query: Optional[ESQuery] = None,
                      *,
                      page_size: Optional[int] = None,
                      page: Optional[int] = None,
                      sort: Optional[Union[list, str]] = None,
                      routing: Optional[str] = None,
                      aggs: Optional[ESAggs] = None,
                      **kwargs) -> ESResponse:
        """
        Raw ES search method

        :param query: ElasticSearch query dict
        :param page_size: Pagination page size
        :param page: Pagination page num, 1st page is 1
        :param sort: Name of field to be sorted, or sort term list of dict, if not specified, model's default sort will
                     be used, or no sorting
        :param routing: Shard routing value
        :param aggs: Aggregations
        :param kwargs: Other search API params
        :return: Raw ES response.
        """
        if isinstance(sort, str):
            sort = [{sort: {'order': 'asc'}}]
        elif sort is None and cls.ESConfig.default_sort is not None:
            sort = cls.ESConfig.default_sort

        if page_size is not None and page is None:
            page = 1

        return await cls.call('search', query=query,
                              from_=((page - 1) * page_size) if page_size is not None else 0,
                              size=page_size, sort=sort, routing=routing,
                              aggs=aggs,
                              seq_no_primary_term=True, version=True,
                              **kwargs)

    @classmethod
    async def search(cls: Type[TModel], query: ESQuery, *,
                     page_size: Optional[int] = None,
                     page: Optional[int] = None,
                     sort: Optional[Union[list, str]] = None,
                     routing: Optional[str] = None,
                     res_dict: bool = False,
                     **kwargs) -> Union[List[TModel], Dict[str, TModel]]:
        """
        Search Model with query dict

        :param query: ElasticSearch query dict
        :param page_size: Pagination page size
        :param page: Pagination page num, 1st page is 1
        :param sort: Name of field to be sorted, or sort term list of dict, if not specified, model's default sort will
                     be used, or no sorting
        :param routing: Shard routing value
        :param res_dict: If the result should be a dict with id as key and model as value instead of a list of models
        :param kwargs: Other search API params
        :return: The result list
        """
        res = await cls._search(query, page_size=page_size, page=page, sort=sort, routing=routing, **kwargs)
        try:
            if res_dict:
                res = {hit['_id']: cls.from_es(hit) for hit in res['hits']['hits']}
            else:
                res = [cls.from_es(hit) for hit in res['hits']['hits']]
            return await _lazy_process_results(res)
        except KeyError:
            return []

    @classmethod
    async def search_one(cls: Type[TModel], query: ESQuery, *, routing: Optional[str] = None, **kwargs) \
            -> Optional[TModel]:
        """
        Search Model and return the first result

        :param query: ElasticSearch query dict
        :param routing: Shard routing value
        :param kwargs: Other search API params
        :return: The first result or None if no result
        """
        res = await cls.search(query, page_size=1, routing=routing, **kwargs)
        if len(res) > 0:
            return res[0]
        else:
            return None

    @staticmethod
    def create_query_from_dict(fields: Dict[str, Union[str, int, float]]) -> ESQuery:
        """
        Creates a query dict from a dictionary of fields and values

        :param fields: A dictionary of fields and values to search by
        :return: A query dict
        """
        return {
            'bool': {
                'must': [{
                    'match': {
                        k: {'query': v, 'operator': 'and'},
                    }
                } for k, v in fields.items()]
            }
        }

    @classmethod
    async def search_by_fields(cls: Type[TModel],
                               fields: Dict[str, Union[str, int, float]],
                               *,
                               page_size: Optional[int] = None,
                               page: Optional[int] = None,
                               sort: Optional[Union[list, str]] = None,
                               routing: Optional[str] = None,
                               aggs: Optional[ESAggs] = None,
                               res_dict: bool = False,
                               **kwargs) -> List[TModel]:
        """
        Search Model by fields as key-value pairs

        :param fields: A dictionary of fields and values to search by
        :param page_size: Pagination page size
        :param page: Pagination page num, 1st page is 1
        :param sort: Name of field to be sorted, or sort term list of dict, if not specified,
                     model's default sort will be used, or no sorting
        :param routing: Shard routing value
        :param aggs: Aggregations
        :param res_dict: If the result should be a dict with id as key and model as value instead of a list of models
        :param kwargs: Other search API params
        :return: The result list
        """
        query = cls.create_query_from_dict(fields)
        return await cls.search(query, page_size=page_size, page=page, sort=sort, routing=routing,
                                aggs=aggs, res_dict=res_dict, **kwargs)

    @classmethod
    async def search_one_by_fields(cls: Type[TModel],
                                   fields: Dict[str, Union[str, int, float]],
                                   *, routing: Optional[str] = None,
                                   aggs: Optional[ESAggs] = None,
                                   **kwargs) -> Optional[TModel]:
        """
        Search Model by fields as key-value pairs and return the first result

        :param fields: A dictionary of fields and values to search by
        :param routing: Shard routing value
        :param aggs: Aggregations
        :param kwargs: Other search API params
        :return: The first result or None if no result
        """
        query = cls.create_query_from_dict(fields)
        return await cls.search_one(query, routing=routing, aggs=aggs, **kwargs)

    @classmethod
    async def all(cls: Type[TModel], **kwargs) -> List[TModel]:
        """
        Get all documents

        :param kwargs: Other search API params
        :return: The result list
        """
        return await cls.search({'match_all': {}}, **kwargs)

    @classmethod
    async def aggregate(cls: Type[TModel],
                        aggs: ESAggs,
                        *,
                        query: Optional[ESQuery] = None,
                        routing: Optional[str] = None,
                        **kwargs) -> ESAggsResponse:
        """
        Aggregate Model with aggregation dict
        Before aggregation the model can be filtered by query dict.

        :param aggs: Aggregation dict
        :param query: ElasticSearch query dict
        :param routing: Shard routing value
        :param kwargs: Other search API params
        :return: The result list
        """
        try:
            res = await cls._search(query, aggs=aggs, routing=routing, page_size=0, **kwargs)
            return res['aggregations']
        except KeyError:
            return {}


class ESModelTimestamp(ESModel):
    """
    Model which stores `created_at` and `modified_at` fields automatcally.
    """
    created_at: Optional[datetime] = Field(None, description="Creation date and time")
    modified_at: Optional[datetime] = Field(default_factory=utcnow, description="Modification date and time")

    async def save(self, *, wait_for=False, force_new=False, pipeline: Optional[str] = None,
                   routing: Optional[str] = None) -> str:
        """
        Save document into elasticsearch.

        If document already exists, existing document will be updated as per native elasticsearch index operation.
        If model has id (Meta.id_field or __id__), this will be used as the elasticsearch _id. The id field will be
        removed from the document before indexing.
        If no id is provided, then document will be indexed and elasticsearch will generate a suitable id that will be
        populated on the returned model.

        :param wait_for: Waits for all shards to sync before returning response - useful when writing
            tests. Defaults to False.
        :param force_new: It is assumed to be a new document, so created_at will be set to current time
                          (it is no more necessary, because created_at is set to current time if it is None.
                          It is here for backward compatibility)
        :param pipeline: Pipeline to use for indexing
        :param routing: Shard routing value
        :return: The new document's ID
        """
        self.modified_at = utcnow()
        # Set created_at if not already set
        if force_new or not self.created_at:
            self.created_at = self.modified_at
        return await super().save(wait_for=wait_for, pipeline=pipeline, routing=routing)


#
# Lazy properties
#

def set_max_lazy_property_concurrency(concurrency: int):
    """
    Set the maximum concurrency of processing lazy properties

    If this is not set, the default is 16.

    :param concurrency: The maximum concurrency
    """
    global _lazy_semaphore_concurrency
    _lazy_semaphore_concurrency = concurrency
    _lazy_property_semaphore.set(asyncio.Semaphore(concurrency))


async def _lazy_process_results(res: Union[List[ESModel], ESModel, Dict[str, ESModel]]) \
        -> Union[List[ESModel], ESModel, Dict[str, ESModel]]:
    """
    Process the results of ES query to calculate lazy properties recursively

    :param res: The result of the endpoint
    :return: The result of the endpoint
    """
    tasks = []

    if isinstance(res, ESBaseModel):
        tasks.append(asyncio.create_task(res.calc_lazy_properties()))

    elif isinstance(res, list):
        for r in res:
            tasks.append(asyncio.create_task(r.calc_lazy_properties()))

    elif isinstance(res, dict):
        for r in res.values():
            tasks.append(asyncio.create_task(r.calc_lazy_properties()))

    else:
        raise TypeError(f"Invalid return type: {type(res)}")

    # Wait for tasks to be ready
    await asyncio.gather(*tasks)

    return res


def lazy_property(func: Callable[[], Awaitable[Any]]):
    """
    Decorator for lazy properties

    Lazy properties computed after search from ES

    :param func: The async function to decorate
    :return: The decorated function
    """
    assert inspect.iscoroutinefunction(func), \
        f"The function {func.__name__} must be a coroutine function"
    assert inspect.signature(func).return_annotation is not inspect.Signature.empty, \
        f"The function {func.__name__} must have a return annotation"

    @wraps(func)
    def wrapper(self):
        # Initialize call stack if not exists
        if not hasattr(self, '__lazy_call_stack__'):
            self.__lazy_call_stack__ = []

        # Return the property with underscore prefix
        try:
            return getattr(self, '_' + func.__name__)
        except AttributeError:
            return None

    @wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        # Check recursion depth
        depth = _lazy_property_recursion_depth.get()
        if depth >= self.ESConfig.lazy_property_max_recursion_depth:
            logger.warning(f"Recursion depth exceeded for {self.__class__.__name__}.{func.__name__}")
            return None
        # Increase recursion depth
        _lazy_property_recursion_depth.set(depth + 1)
        _lazy_property_semaphore.set(asyncio.Semaphore(_lazy_semaphore_concurrency))
        # Call the original function
        return await func(self, *args, **kwargs)

    # Create a property from it
    prop = property(wrapper)
    # Set the original function as __lazy_property_func__
    setattr(wrapper, '__lazy_property__', async_wrapper)

    return prop


#
# Optimistic concurrency control
#

def retry_on_conflict(max_retries=-1):
    """
    Decorator for optimistic concurrency control

    :param max_retries: The maximum number of retries, -1 for infinite
    :return: The decorated function
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except ElasticConflictError:
                    if max_retries == -1 or retries < max_retries:
                        retries += 1
                        logger.warning(f"Optimistic concurrency control conflict, retrying {retries}/{max_retries}")
                        continue
                    else:
                        raise

        return wrapper

    return decorator


#
# Pagination and sort
#

class Pagination(BaseModel):
    """
    Pagination parameters
    """
    page: int = 1
    """ The page number """
    page_size: int = 10
    """ The page size """
    callback: Optional[Callable[[int], Awaitable[None]]] = None
    """ Callback after the search is done with the total number of hits """

    def __call__(self, model_cls: Type[TModel]) -> Type[TModel]:
        """
        Decorate the model to apply pagination

        :param model_cls: The model to decorate
        :return: The decorated model
        """

        class Wrapped(model_cls):
            """
            Decorated model class with pagination
            """

            # noinspection PyProtectedMember
            @classmethod
            async def _search(cls: model_cls, query: ESQuery, *,
                              page_size: Optional[int] = None,
                              page: int = None,
                              sort: Optional[Union[list, str]] = None,
                              **kwargs) -> ESResponse:
                page_size = self.page_size if page_size is None else page_size
                page = self.page if page is None else page
                res = await model_cls._search(query, page_size=page_size, page=page, sort=sort, **kwargs)
                if self.callback is not None:
                    try:
                        await self.callback(res['hits']['total']['value'])
                    except KeyError:
                        pass
                return res

        # Copy magic attributes
        Wrapped.__name__ = model_cls.__name__ + '_pagination'
        Wrapped.__qualname__ = model_cls.__qualname__ + '_pagination'
        Wrapped.__module__ = model_cls.__module__
        Wrapped.__doc__ = model_cls.__doc__
        Wrapped.__annotations__ = model_cls.__annotations__
        # Copy ESConfig
        Wrapped.ESConfig = model_cls.ESConfig

        return Wrapped


class SortOrder(TypedDict):
    """
    Order definition
    """
    order: Literal['asc', 'desc']


class Sort(BaseModel):
    """
    Sort parameters
    """
    sort: Union[List[Dict[str, SortOrder]], str, None]

    def __call__(self, model_cls: Type[TModel]) -> Type[TModel]:
        """
        Decorate the model to apply sort

        :param model_cls: The model to decorate
        :return: The decorated model
        """

        class Wrapped(model_cls):
            """
            Decorated model class with sort
            """

            # noinspection PyProtectedMember
            @classmethod
            async def _search(cls: model_cls, query: ESQuery, *,
                              page_size: Optional[int] = None,
                              page: int = None,
                              sort: Optional[Union[list, str]] = None,
                              **kwargs) -> ESResponse:
                sort = self.sort if sort is None else sort
                return await model_cls._search(query, page_size=page_size, page=page, sort=sort, **kwargs)

        # Copy magic attributes
        Wrapped.__name__ = model_cls.__name__ + '_sort'
        Wrapped.__qualname__ = model_cls.__qualname__ + '_sort'
        Wrapped.__module__ = model_cls.__module__
        Wrapped.__doc__ = model_cls.__doc__
        Wrapped.__annotations__ = model_cls.__annotations__
        # Copy ESConfig
        Wrapped.ESConfig = model_cls.ESConfig

        return Wrapped


#
# Index templates and mappings
#

async def create_index_template(name: str,
                                *,
                                prefix_name: str,
                                shards=1, replicas=0,
                                **other_settings: Any) -> object:
    """
    Create index template

    :param name: The name of the template
    :param prefix_name: The prefix of index pattern
    :param shards: Number of shards
    :param replicas: Nuber of replicas
    :param other_settings: Other settings
    :return: The result object from ES
    """
    return await es.indices.put_template(
        name=name,
        index_patterns=[f'{prefix_name}-*'],
        settings=dict(
            number_of_shards=shards,
            number_of_replicas=replicas,
            **other_settings
        ),
        request_timeout=90
    )


async def setup_mappings(*_, debug=False):
    """
    Create mappings for indices or try to extend it if there are new fields
    """

    # noinspection PyShadowingNames
    def get_field_data(pydantic_type: type) -> dict:
        """ Get field data from pydantic type """
        origin = get_origin(pydantic_type)
        # Handle Union type, which must be a type definition from esorm.fields, because other unions not allowed
        if origin is Union:
            args = get_args(pydantic_type)

            # Optional may equal to Union[..., None], we don't use Optional in ES, but its child
            if type(None) in args:
                return get_field_data(args[0])

            for arg in args:
                if hasattr(arg, '__es_type__'):
                    return {'type': arg.__es_type__}

            raise ValueError('Union is not supported as ES field type!')

        # We don't use Optional in ES, but its child
        if origin is Optional:
            args = get_args(pydantic_type)
            return get_field_data(args[0])

        # List types
        if origin is list:
            properties = {}
            args = get_args(pydantic_type)
            create_mapping(args[0], properties)
            return {
                'type': 'nested',
                'properties': properties
            }

        # Not supported origin type
        if origin:
            raise ValueError(f'Unknown ES field type: {pydantic_type}')

        # Nested class
        if issubclass(pydantic_type, BaseModel):
            # If it is a model but has an es_type, use it (e.g. geo_point)
            if hasattr(pydantic_type, '__es_type__'):
                return {'type': pydantic_type.__es_type__}

            properties = {}
            create_mapping(pydantic_type, properties)
            return {'properties': properties}

        # Is it an ESORM type?
        if hasattr(pydantic_type, '__es_type__'):
            return {'type': pydantic_type.__es_type__}

        # Python type
        try:
            # noinspection PyTypeChecker
            return {'type': _pydantic_type_map[pydantic_type]}
        except KeyError:
            pass

        raise ValueError(f'Unknown ES field type: {pydantic_type}')

    # noinspection PyShadowingNames
    def create_mapping(model: Union[Type[BaseModel]], properties: dict):
        """ Creates mapping for the model """
        field_info: FieldInfo
        for name, field_info in model.model_fields.items():
            # Skip id field, because it won't be stored
            if hasattr(model, 'ESConfig') and model.ESConfig.id_field == name:
                continue
            extra = field_info.json_schema_extra or {}
            # Process field
            res = get_field_data(field_info.annotation)
            _type = res.get('type', None)
            if 'index' in extra and _type != 'binary':
                if 'properties' in res:
                    for v in res['properties'].values():
                        v['index'] = extra['index']
                else:
                    res['index'] = extra['index']
            properties[name] = res

    # Process all models and create mappings
    for index, model in _ESModelMeta.__models__.items():
        # Get mappings from ES if already exists
        index_exists = False
        try:
            properties = (await es.indices.get_mapping(index=index, request_timeout=90))[index]['mappings'][
                'properties']
            index_exists = True
        except (elasticsearch.NotFoundError, KeyError):
            properties = {}
        create_mapping(model, properties)

        if debug:
            from pprint import pformat
            logger.debug(
                f"`{index}` mappings:\n {pformat(properties, indent=2, width=100, compact=False, sort_dicts=False)}")

        try:
            if not index_exists:
                await es.indices.create(index=index,
                                        mappings={'properties': properties},
                                        settings=model.ESConfig.settings,
                                        request_timeout=90)
            else:
                await es.indices.put_mapping(index=index, properties=properties, request_timeout=90)
        except elasticsearch.BadRequestError:
            logger.warning(f"Index mappings error:\n{traceback.format_exc(limit=5)}")
