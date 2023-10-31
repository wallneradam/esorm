"""
ElasticSearch ORM main module
"""
from typing import Optional, cast, Union, List, Mapping

import asyncio

from elasticsearch import AsyncElasticsearch
from elastic_transport import NodeConfig

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


# Global client proxy
es = cast(AsyncElasticsearch, _ESProxy())

__all__ = ['es', 'connect']


async def connect(hosts: Union[str, List[Union[str, Mapping[str, Union[str, int]], NodeConfig]]],
                  *args, wait=False, **kwargs) -> Optional[AsyncElasticsearch]:
    """
    Connect to ElasticSearch

    :param hosts: ElasticSearch hosts to connect, either a list a mapping, or a single string
    :param args: Other AsyncElasticsearch arguments
    :param wait: Wait for AsyncElasticsearch to be ready
    :param kwargs: Other AsyncElasticsearch keyword arguments
    :return: AsyncElasticsearch client instance
    """
    cast(_ESProxy, es).set_client(AsyncElasticsearch(hosts=hosts, *args, **kwargs))

    try:
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
