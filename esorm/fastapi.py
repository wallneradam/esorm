"""
FastaAPI utilities for ESORM
"""
from typing import Union, List, Dict

import inspect
import io
import tokenize

from enum import Enum

from fastapi import Response

from .model import Pagination, Sort
from .utils import camel_case

_max_page_size = 10000

__all__ = [
    'make_dep_pagination',
    'make_dep_sort',
    'set_max_page_size',
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
