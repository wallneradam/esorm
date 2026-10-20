"""
Type hints of the ESModel methods which return models and get a query

For type checkers ESModel.search(), search_one(), scan() and scan_batches() are descriptors (the runtime methods are
normal classmethods), because PyCharm can't infer the type of a generic call (e.g. ``cls: Type[TModel]`` ->
``List[TModel]``), if an argument is a TypedDict literal which doesn't have all the keys (a PyCharm bug), so the
results of ``User.search({'term': {...}})`` would be untyped. The descriptors bind the model class on attribute access,
so the calls are not generic anymore.

PyCharm can't select an overload in this case either, it uses the first one. So the first overload of search() can't
be matched by other type checkers (it has a required keyword argument which can't be given), and its result type has
the methods of both possible results (a list, or a dict with res_dict=True).

Builder queries (esorm.builder.Query) have their own overloads after the query dict ones: PyCharm reports correct dict
literals if the parameter is a union of a TypedDict and something else, and it uses the first overload if the query is
a dict literal, so the query dict overloads must be first.
"""
from typing import (Any, AsyncGenerator, Coroutine, Dict, Generic, ItemsView, KeysView, List, Literal, NoReturn,
                    Optional, SupportsIndex, Type, TypeVar, Union, ValuesView, overload)

from .model import ESModel
from .query import ESQuery
from .builder import Query

_M = TypeVar('_M')
_TModel = TypeVar('_TModel', bound=ESModel)


class _Models(List[_M]):
    """
    The result of search() for PyCharm, if it can't infer the type: the models in a list, or in a dict by ids
    (``res_dict=True``), so it has the methods of both
    """

    @overload  # type: ignore[override]
    def __getitem__(self, key: SupportsIndex, /) -> _M: ...

    @overload
    def __getitem__(self, key: slice, /) -> List[_M]: ...

    @overload
    def __getitem__(self, key: str, /) -> _M: ...

    def items(self) -> ItemsView[str, _M]: ...

    def keys(self) -> KeysView[str]: ...

    def values(self) -> ValuesView[_M]: ...

    def get(self, key: str, default: Optional[_M] = None, /) -> Optional[_M]: ...


class _Search(Generic[_M]):
    """ ESModel.search() of a model class """

    @overload
    def __call__(self, query: ESQuery, *,
                 _pycharm_workaround: NoReturn,
                 page_size: Optional[int] = None,
                 page: Optional[int] = None,
                 sort: Optional[Union[List[Any], str]] = None,
                 routing: Optional[str] = None,
                 res_dict: bool = False,
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> Coroutine[Any, Any, _Models[_M]]:
        """ Search Model with query dict (PyCharm only, see the module docstring) """

    @overload
    def __call__(self, query: ESQuery, *,
                 page_size: Optional[int] = None,
                 page: Optional[int] = None,
                 sort: Optional[Union[List[Any], str]] = None,
                 routing: Optional[str] = None,
                 res_dict: Literal[False] = False,
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> Coroutine[Any, Any, List[_M]]:
        """
        Search Model with query dict

        :param query: ElasticSearch query dict or builder query
        :param page_size: Pagination page size
        :param page: Pagination page num, 1st page is 1
        :param sort: Name of field to be sorted, or sort term list of dict, if not specified, model's default sort will
                     be used, or no sorting
        :param routing: Shard routing value
        :param res_dict: If the result should be a dict with id as key and model as value instead of a list of models
        :param index: Index name, if not set, it will use the index from ESConfig
        :param _source: Fields to return, if not set, it will return all fields
        :param kwargs: Other search API params
        :return: The result list, or a dict of id - model pairs if `res_dict` is True
        """

    @overload
    def __call__(self, query: ESQuery, *,
                 page_size: Optional[int] = None,
                 page: Optional[int] = None,
                 sort: Optional[Union[List[Any], str]] = None,
                 routing: Optional[str] = None,
                 res_dict: Literal[True],
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> Coroutine[Any, Any, Dict[str, _M]]: ...

    @overload
    def __call__(self, query: ESQuery, *,
                 page_size: Optional[int] = None,
                 page: Optional[int] = None,
                 sort: Optional[Union[List[Any], str]] = None,
                 routing: Optional[str] = None,
                 res_dict: bool,
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> Coroutine[Any, Any, Union[List[_M], Dict[str, _M]]]: ...

    # Builder queries

    @overload
    def __call__(self, query: Query, *,
                 page_size: Optional[int] = None,
                 page: Optional[int] = None,
                 sort: Optional[Union[List[Any], str]] = None,
                 routing: Optional[str] = None,
                 res_dict: Literal[False] = False,
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> Coroutine[Any, Any, List[_M]]: ...

    @overload
    def __call__(self, query: Query, *,
                 page_size: Optional[int] = None,
                 page: Optional[int] = None,
                 sort: Optional[Union[List[Any], str]] = None,
                 routing: Optional[str] = None,
                 res_dict: Literal[True],
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> Coroutine[Any, Any, Dict[str, _M]]: ...

    @overload
    def __call__(self, query: Query, *,
                 page_size: Optional[int] = None,
                 page: Optional[int] = None,
                 sort: Optional[Union[List[Any], str]] = None,
                 routing: Optional[str] = None,
                 res_dict: bool,
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> Coroutine[Any, Any, Union[List[_M], Dict[str, _M]]]: ...


class SearchMethod:
    def __get__(self, obj: Any, owner: Type[_TModel]) -> _Search[_TModel]: ...


class _SearchOne(Generic[_M]):
    """ ESModel.search_one() of a model class """

    @overload
    def __call__(self, query: ESQuery, *, routing: Optional[str] = None,
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> Coroutine[Any, Any, Optional[_M]]:
        """
        Search Model and return the first result

        :param query: ElasticSearch query dict or builder query
        :param routing: Shard routing value
        :param index: Index name, if not set, it will use the index from ESConfig
        :param _source: Fields to return, if not set, it will return all fields
        :param kwargs: Other search API params
        :return: The first result or None if no result
        """

    @overload
    def __call__(self, query: Query, *, routing: Optional[str] = None,
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> Coroutine[Any, Any, Optional[_M]]: ...


class SearchOneMethod:
    def __get__(self, obj: Any, owner: Type[_TModel]) -> _SearchOne[_TModel]: ...


# The query has its own overload, because PyCharm checks a dict literal argument only if the parameter is a TypedDict
# (not Optional)

class _ScanBatches(Generic[_M]):
    """ ESModel.scan_batches() of a model class """

    @overload
    def __call__(self, query: ESQuery, *,
                 batch_size: int = 1000,
                 sort: Optional[Union[List[Any], str]] = None,
                 keep_alive: str = '1m',
                 routing: Optional[str] = None,
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> AsyncGenerator[List[_M], None]:
        """
        Iterate through all results of a query in batches, see :meth:`esorm.model.ESModel.scan_batches`

        :param query: ElasticSearch query dict or builder query, if not set, all documents are returned
        :param batch_size: The number of documents to fetch in one request
        :param sort: Name of field to be sorted, or sort term list of dict, if not specified, model's default sort
                     will be used, or the index order, which is the most efficient
        :param keep_alive: How long the point in time is kept alive between two requests
        :param routing: Shard routing value
        :param index: Index name, if not set, it will use the index from ESConfig
        :param _source: Fields to return, if not set, it will return all fields
        :param kwargs: Other search API params
        :return: Async iterator of result lists, each list has maximum `batch_size` models
        """

    @overload
    def __call__(self, query: Optional[ESQuery] = None, *,
                 batch_size: int = 1000,
                 sort: Optional[Union[List[Any], str]] = None,
                 keep_alive: str = '1m',
                 routing: Optional[str] = None,
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> AsyncGenerator[List[_M], None]: ...

    @overload
    def __call__(self, query: Query, *,
                 batch_size: int = 1000,
                 sort: Optional[Union[List[Any], str]] = None,
                 keep_alive: str = '1m',
                 routing: Optional[str] = None,
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> AsyncGenerator[List[_M], None]: ...


class ScanBatchesMethod:
    def __get__(self, obj: Any, owner: Type[_TModel]) -> _ScanBatches[_TModel]: ...


class _Scan(Generic[_M]):
    """ ESModel.scan() of a model class """

    @overload
    def __call__(self, query: ESQuery, *,
                 batch_size: int = 1000,
                 sort: Optional[Union[List[Any], str]] = None,
                 keep_alive: str = '1m',
                 routing: Optional[str] = None,
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> AsyncGenerator[_M, None]:
        """
        Iterate through all results of a query, see :meth:`esorm.model.ESModel.scan`

        :param query: ElasticSearch query dict or builder query, if not set, all documents are returned
        :param batch_size: The number of documents to fetch in one request
        :param sort: Name of field to be sorted, or sort term list of dict, if not specified, model's default sort
                     will be used, or the index order, which is the most efficient
        :param keep_alive: How long the point in time is kept alive between two requests
        :param routing: Shard routing value
        :param index: Index name, if not set, it will use the index from ESConfig
        :param _source: Fields to return, if not set, it will return all fields
        :param kwargs: Other search API params
        :return: Async iterator of results
        """

    @overload
    def __call__(self, query: Optional[ESQuery] = None, *,
                 batch_size: int = 1000,
                 sort: Optional[Union[List[Any], str]] = None,
                 keep_alive: str = '1m',
                 routing: Optional[str] = None,
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> AsyncGenerator[_M, None]: ...

    @overload
    def __call__(self, query: Query, *,
                 batch_size: int = 1000,
                 sort: Optional[Union[List[Any], str]] = None,
                 keep_alive: str = '1m',
                 routing: Optional[str] = None,
                 index: Optional[str] = None,
                 _source: Optional[Union[str, List[str]]] = None,
                 **kwargs: Any) -> AsyncGenerator[_M, None]: ...


class ScanMethod:
    def __get__(self, obj: Any, owner: Type[_TModel]) -> _Scan[_TModel]: ...
