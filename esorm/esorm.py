"""
ElasticSearch ORM main module
"""
from typing import Optional, cast, Union, List, Mapping, Type

import asyncio
import re

from elasticsearch import AsyncElasticsearch
from elastic_transport import NodeConfig, AsyncTransport, HttpHeaders, TransportApiResponse

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

    async def perform_request(self, method: str, target: str, **kwargs) -> TransportApiResponse:
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
es = cast(AsyncElasticsearch, _ESProxy())

__all__ = ['es', 'connect', 'get_es_version']


async def connect(hosts: Union[str, Union[List[Union[str, Mapping[str, Union[str, int]], NodeConfig]], None]],
                  *args, wait=False, **kwargs) -> Optional[AsyncElasticsearch]:
    """
    Connect to ElasticSearch

    :param hosts: ElasticSearch hosts to connect, either a list a mapping, or a single string
    :param args: Other AsyncElasticsearch arguments
    :param wait: Wait for AsyncElasticsearch to be ready
    :param kwargs: Other AsyncElasticsearch keyword arguments
    :return: AsyncElasticsearch client instance
    """
    kwargs['transport_class'] = _compat_transport_class(kwargs.get('transport_class', AsyncTransport))
    cast(_ESProxy, es).set_client(AsyncElasticsearch(hosts=hosts, *args, **kwargs))

    try:
        logger.info(f"Connecting to ElasticSearch ({hosts})…")
        if wait:
            # Wait for ES to start, this will block until ES is ready to prevent routers start early
            while True:
                status = await es.ping()
                if not status:
                    logger.info("Waiting for ElasticSearch to be ready…")
                    await asyncio.sleep(2.0)
                else:
                    break

        return es

    except asyncio.CancelledError:
        return None


async def get_es_version() -> str:
    """
    Get ElasticSearch version

    :return: ElasticSearch version
    """
    info = await es.info()
    return info['version']['number']
