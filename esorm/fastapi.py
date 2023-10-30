"""
FastaAPI utilities for ESORM
"""
from typing import Union, List, Dict, Callable, Awaitable, Type

import asyncio

import inspect
import io
import tokenize

from enum import Enum
from functools import wraps

from fastapi import Response
from fastapi.applications import AppType
from fastapi.routing import APIRoute

from .model import Pagination, Sort, ESModel
from .utils import camel_case

_max_page_size = 10000

__all__ = [
    'make_dep_pagination',
    'make_dep_sort',
    'set_max_page_size',
    'wait_lazy_results',
    'lazy_resuts_all_enpoints'
]


#
# Pagination and sort dependencies
#

def set_max_page_size(max_page_size: int):
    """
    Set the maximum page size for queries

    :param max_page_size: The maximum page size
    :return: None
    """
    global _max_page_size
    _max_page_size = max_page_size


def make_dep_pagination(default_page: int = 1, default_page_size: int = 10, set_headers: bool = True) -> callable:
    """
    Create a pagination dependency with default values

    :param default_page: Default page number, the first page is 1
    :param default_page_size: Default page size, the default is 10
    :param set_headers: Set X-Total-Hits header after search
    :return: Pagination dependency
    """

    async def _dep_pagination(response: Response,
                              _page: int = default_page, _page_size: int = default_page_size) -> Pagination:
        """
        Pagination dependency

        :param response: Response object
        :param _page: Page number, the first page is 1
        :param _page_size: Page size, the default is 10
        :return: Pagination object
        """

        async def _cb_search(total_hits: int):
            """
            Callback function after search with total hits

            :param total_hits: Total hits of the query
            :return:
            """
            # NOTE: it should be in the exposed_headers in CORS middleware!!
            response.headers['X-Total-Hits'] = str(total_hits)

        # Ensure page size is not too large
        _page_size = min(_page_size, max(_max_page_size, default_page_size))
        return Pagination(page=_page, page_size=_page_size, callback=_cb_search if set_headers else None)

    return _dep_pagination


def make_dep_sort(**kwargs: Union[List[Dict[str, dict]], Dict[str, any]]) -> callable:
    """
    Create a sort dependency with sort definitions

    :param kwargs: Sort definitions
    :return: Sort dependency
    """
    assert len(kwargs) > 0, "At least one sort field is required"

    # Get the name of the function where this function is called
    frame = inspect.currentframe().f_back
    module_name = inspect.getmodule(frame).__name__
    source_code = inspect.getsource(frame)
    tokens = list(tokenize.tokenize(io.BytesIO(source_code.encode('utf-8')).readline))
    function_name = ""
    for i, token in enumerate(tokens):
        if token.type == tokenize.NAME and token.string == "def" and token.start[0] <= frame.f_lineno:
            function_name = tokens[i + 1].string
        elif token.type == tokenize.NAME and token.string == "def" and token.start[0] > frame.f_lineno:
            break
    assert function_name, "Cannot find function name!"
    fqdn = f"{module_name}.{function_name}"
    # Create a new enum class for sort fields
    sort_enum = Enum('SortEnum' + camel_case(fqdn.replace('.', '_'), True),
                     {k: k for k in kwargs.keys()})

    async def _dep_sort(_sort: sort_enum = None) -> Sort:
        """
        Sort dependency

        :param _sort: Sort JSON string
        :return: Sort object
        """
        return Sort(sort=kwargs[_sort.name] if _sort else None)

    return _dep_sort


#
# Lazy properties
#

async def _lazy_process_endpoint_result(res: Union[List[ESModel], ESModel, Dict[str, ESModel]], concurrency: int = 5) \
        -> Union[List[ESModel], ESModel, Dict[str, ESModel]]:
    """
    Process the result of the endpoint

    :param res: The result of the endpoint
    :param concurrency: The concurrency of the tasks
    :return: The result of the endpoint
    """
    if isinstance(res, ESModel):
        await res.calc_lazy_properties()

    elif isinstance(res, list):
        tasks = []
        for r in res:
            tasks.append(r.calc_lazy_properties())
            if len(tasks) >= concurrency:
                await asyncio.gather(*tasks)
                tasks.clear()
        if tasks:
            await asyncio.gather(*tasks)

    elif isinstance(res, dict):
        tasks = []
        for r in res.values():
            tasks.append(r.calc_lazy_properties())
            if len(tasks) >= concurrency:
                await asyncio.gather(*tasks)
                tasks.clear()
        if tasks:
            await asyncio.gather(*tasks)
    else:
        raise TypeError(f"Invalid return type: {type(res)}")

    return res


def wait_lazy_results(
        func: Callable[..., Awaitable[Union[List[ESModel], ESModel, Dict[str, ESModel]]]] = None,
        *,
        concurrency: int = 5
) -> Callable[..., Awaitable[Union[List[ESModel], ESModel, Dict[str, ESModel]]]]:
    """
    Decorator to wait for lazy properties to be computed

    :param func: The function to decorate
    :param concurrency: The concurrency of the tasks
    :return: The decorated function
    :raises TypeError: If the return type is not ESModel or list/dict of ESModel
    """

    def decorator(_func) -> Callable[..., Awaitable[Union[List[ESModel], ESModel, Dict[str, ESModel]]]]:
        """ The actual decorator """

        @wraps(_func)
        async def wrapper(*args, **kwargs) -> Union[List[ESModel], ESModel, Dict[str, ESModel]]:
            res = await _func(*args, **kwargs)
            return await _lazy_process_endpoint_result(res, concurrency=concurrency)

        return wrapper

    if func is None:
        return decorator()
    return decorator(func)


def lazy_resuts_all_enpoints(app: AppType):
    """
    Wait for lazy properties to be computed for all routes that returns ESModel or list/dict of ESModel

    This will decorate all route endpoints of the app to make them wait for lazy properties to be computed.
    NOTE: The method must be annotated to return ESModel or list/dict of ESModel, without annotation it won't work.

    :param app: The FastAPI app
    """

    def is_esmodel_subclass(_response_model: Type) -> bool:
        """ Check if the response model is ESModel or list/dict of ESModel """
        # noinspection PyUnresolvedReferences
        if isinstance(_response_model, type) and issubclass(_response_model, ESModel):
            return True
        elif hasattr(_response_model, "__origin__") and _response_model.__origin__ in [list, dict]:
            return any(isinstance(arg, type) and issubclass(arg, ESModel) for arg in
                       getattr(_response_model, "__args__", ()))
        return False

    for route in app.router.routes:
        if isinstance(route, APIRoute):
            # Check if endpoints return type is ESModel or list/dict of ESModel
            response_model = route.response_model
            if response_model:
                try:
                    if is_esmodel_subclass(response_model):
                        # noinspection PyUnresolvedReferences
                        route.dependant.call = wait_lazy_results(route.dependant.call)
                except TypeError:
                    continue
