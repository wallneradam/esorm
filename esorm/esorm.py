"""
ElasticSearch ORM main module
"""
from typing import Optional, cast, Union, List, Mapping, Type, Literal, Dict, Any

import asyncio
import importlib.util
import re

from elasticsearch import AsyncElasticsearch, ApiError, UnsupportedProductError
from elastic_transport import NodeConfig, AsyncTransport, HttpHeaders, TransportApiResponse, TransportError

from .logger import logger


class _ESProxy:
    """
    ElasticSearch client proxy
    """
    __client__: Optional[AsyncElasticsearch] = None

    def set_client(self, client):
        self.__client__ = client

    def __getattr__(self, name):
        if self.__client__ is None:
            if name == '__wrapped__':  # This is needed for pdoc3
                return object.__getattribute__(self, name)
            raise ValueError(f"ElasticSearch client has not been set yet, please call connect! ({name})")
        return getattr(self.__client__, name)


_COMPAT_RE = re.compile(r'compatible-with=(\d+)')


class _CompatTransportMixin:
    """
    Transport mixin to make the ElasticSearch client work with servers of an older major version

    The client sends `compatible-with=<client major version>` media types in the Accept and Content-Type headers,
    which are rejected by older servers (e.g. ES 8.x rejects `compatible-with=9`). So before the first request the
    major version of the server is detected, and the compatibility version is lowered to it if needed.
    """
    _server_major: Optional[int] = None
    _server_major_lock: Optional[asyncio.Lock] = None

    async def _detect_server_major(self, headers: HttpHeaders) -> Optional[int]:
        """
        Detect the major version of the server

        :param headers: Headers of the original request (e.g. for authentication)
        :return: Major version of the server or None if it could not be detected
        """
        if self._server_major_lock is None:
            self._server_major_lock = asyncio.Lock()
        async with self._server_major_lock:
            if self._server_major is None:
                detect_headers = HttpHeaders(headers)
                detect_headers['accept'] = 'application/json'
                detect_headers.pop('content-type', None)
                # noinspection PyUnresolvedReferences
                meta, body = await super().perform_request('GET', '/', headers=detect_headers)  # type: ignore
                try:
                    if 200 <= meta.status < 300:
                        self._server_major = int(body['version']['number'].split('.')[0])
                except (TypeError, KeyError, ValueError):
                    pass
            return self._server_major

    async def perform_request(self, method: str, target: str, **kwargs: Any) -> TransportApiResponse:
        headers = kwargs.get('headers')
        if isinstance(headers, Mapping) and any(_COMPAT_RE.search(headers.get(h) or '')
                                                for h in ('accept', 'content-type')):
            headers = kwargs['headers'] = HttpHeaders(headers)
            server_major = self._server_major or await self._detect_server_major(headers)
            if server_major is not None:
                for h in ('accept', 'content-type'):
                    if headers.get(h):
                        headers[h] = _COMPAT_RE.sub(lambda m: f'compatible-with={min(int(m[1]), server_major)}',
                                                    headers[h])
        # noinspection PyUnresolvedReferences
        return await super().perform_request(method, target, **kwargs)  # type: ignore


def _compat_transport_class(transport_class: Type[AsyncTransport]) -> Type[AsyncTransport]:
    """
    Create a transport class which is compatible with servers of an older major version

    :param transport_class: The original transport class
    :return: The compatible transport class
    """
    if issubclass(transport_class, _CompatTransportMixin):
        return transport_class
    return cast(Type[AsyncTransport], type('Compat' + transport_class.__name__,
                                           (_CompatTransportMixin, transport_class), {}))


# Global client proxy
es: AsyncElasticsearch = cast(AsyncElasticsearch, _ESProxy())

Backend = Literal['elasticsearch', 'opensearch']
""" Type of the server ESORM is connected to """

# The backend of the current connection
_backend: Backend = 'elasticsearch'

# ElasticSearch client keyword arguments which have a different name in the opensearch-py client
_OPENSEARCH_KWARGS = {
    'basic_auth': 'http_auth',
    'request_timeout': 'timeout',
}

__all__ = ['es', 'connect', 'get_es_version', 'is_opensearch', 'Backend']


async def _detect_backend(client: AsyncElasticsearch, wait: bool) -> Backend:
    """
    Detect if the server is ElasticSearch or OpenSearch

    :param client: The ElasticSearch client
    :param wait: Wait for the server to be ready
    :return: The detected backend, if it could not be detected, `elasticsearch`
    """
    while True:
        try:
            await client.perform_request('GET', '/', headers={'accept': 'application/json'})
            return 'elasticsearch'
        except UnsupportedProductError as e:
            # The client refuses every server other than ElasticSearch, but the response is in the exception
            if isinstance(e.body, Mapping) and e.body.get('version', {}).get('distribution') == 'opensearch':
                return 'opensearch'
            return 'elasticsearch'
        except (ApiError, TransportError):
            if not wait:
                logger.warning("Could not detect the server type, assuming ElasticSearch.")
                return 'elasticsearch'
            logger.info("Waiting for the server to be ready…")
            await asyncio.sleep(2.0)


def _opensearch_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert AsyncElasticsearch keyword arguments to AsyncOpenSearch ones

    :param kwargs: AsyncElasticsearch keyword arguments
    :return: AsyncOpenSearch keyword arguments
    """
    kwargs = {_OPENSEARCH_KWARGS.get(k, k): v for k, v in kwargs.items()}
    # It is an elastic_transport class, which is not compatible with opensearch-py
    kwargs.pop('transport_class', None)
    return kwargs


async def connect(hosts: Union[str, Union[List[Union[str, Mapping[str, Union[str, int]], NodeConfig]], None]],
                  *args: Any, wait: bool = False, backend: Optional[Backend] = None,
                  **kwargs: Any) -> Optional[AsyncElasticsearch]:
    """
    Connect to ElasticSearch or OpenSearch

    If the backend is not specified and the `opensearch-py` package is installed, the type of the server is
    detected automatically. In this case the keyword arguments of AsyncElasticsearch should be used, the most
    common ones (`basic_auth`, `request_timeout`) are converted for OpenSearch. If you need OpenSearch specific
    arguments (e.g. AWS authentication), specify `backend='opensearch'` and use the arguments of AsyncOpenSearch.

    :param hosts: ElasticSearch hosts to connect, either a list a mapping, or a single string
    :param args: Other AsyncElasticsearch (or AsyncOpenSearch) arguments
    :param wait: Wait for AsyncElasticsearch to be ready
    :param backend: The type of the server (`elasticsearch` or `opensearch`), if not set, it is detected
                    automatically when the `opensearch-py` package is installed, otherwise `elasticsearch`
    :param kwargs: Other AsyncElasticsearch (or AsyncOpenSearch) keyword arguments
    :return: AsyncElasticsearch client instance (or the compatible AsyncOpenSearch client instance)
    """
    global _backend

    try:
        client = None
        if backend != 'opensearch':
            es_kwargs = dict(kwargs)
            es_kwargs['transport_class'] = _compat_transport_class(kwargs.get('transport_class', AsyncTransport))
            client = AsyncElasticsearch(hosts=hosts, *args, **es_kwargs)
            if backend is None:
                backend = 'elasticsearch'
                if importlib.util.find_spec('opensearchpy') is not None:
                    logger.info(f"Detecting server type ({hosts})…")
                    backend = await _detect_backend(client, wait)
                    if backend == 'opensearch':
                        await client.close()
                        client = None
                        kwargs = _opensearch_kwargs(kwargs)

        if client is None:
            from .opensearch import AsyncOpenSearchCompat
            client = AsyncOpenSearchCompat(hosts, *args, **kwargs)

        _backend = backend
        cast(_ESProxy, es).set_client(client)

        logger.info(f"Connecting to {'OpenSearch' if backend == 'opensearch' else 'ElasticSearch'} ({hosts})…")
        if wait:
            # Wait for ES to start, this will block until ES is ready to prevent routers start early
            while True:
                status = await es.ping()
                if not status:
                    logger.info("Waiting for the server to be ready…")
                    await asyncio.sleep(2.0)
                else:
                    break

        return es

    except asyncio.CancelledError:
        return None


async def get_es_version() -> str:
    """
    Get ElasticSearch (or OpenSearch) version

    :return: ElasticSearch (or OpenSearch) version
    """
    info = await es.info()
    return info['version']['number']


def is_opensearch() -> bool:
    """
    Check if ESORM is connected to OpenSearch

    :return: True if the server is OpenSearch
    """
    return _backend == 'opensearch'
