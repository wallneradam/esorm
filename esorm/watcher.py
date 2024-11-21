"""
ElasticSearch Watcher support for ESORM
"""
from typing import TypedDict, Tuple, Dict, List, Any, Union, Type, Optional, cast

import json

from .logger import logger
from .query import ESQuery
from .esorm import es


class WatcherMeta(type):
    """
    Watcher metaclass
    """
    __watchers__: List[Type['Watcher']] = []

    def __init__(cls: Type['Watcher'], name: str, bases: Tuple[type, ...], attrs: Dict[str, Any]) -> None:
        super().__init__(name, bases, attrs)

        WatcherMeta.__watchers__.append(cls)
        for base in bases:
            base = cast(Type['Watcher'], base)
            if base in WatcherMeta.__watchers__:
                WatcherMeta.__watchers__.remove(base)


class Schedule(TypedDict):
    """ Schedule definition """
    interval: str


class Trigger(TypedDict):
    """ Trigger definition """
    schedule: Schedule


class Order(TypedDict):
    """ Order definition """
    order: str


class Body(TypedDict, total=False):
    """ Body definition """
    size: int
    sort: Dict[str, Order]
    query: ESQuery


class Request(TypedDict, total=False):
    """ Request definition """
    indices: Union[List[str], str]
    body: ESQuery
    tepmlate: Dict[str, Any]


class Search(TypedDict, total=False):
    """ Search definition """
    request: Request
    extract: List[str]


class EmptyDict(TypedDict, total=False):
    """ Empty dict definition """


class Compare(TypedDict, total=False):
    """ Compare definition """
    eq: Any
    not_eq: Any
    gt: Union[int, float, str, Dict[str, Any]]
    gte: Union[int, float, str, Dict[str, Any]]
    lt: Union[int, float, str, Dict[str, Any]]
    lte: Union[int, float, str, Dict[str, Any]]


class ArrayCompare(TypedDict, total=False):
    """ Array compare definition """
    path: str
    eq: Any
    not_eq: Any
    gt: Union[int, float, str, Dict[str, Any]]
    gte: Union[int, float, str, Dict[str, Any]]
    lt: Union[int, float, str, Dict[str, Any]]
    lte: Union[int, float, str, Dict[str, Any]]


class Condition(TypedDict, total=False):
    """ Condition definition """
    compare: Dict[str, Compare]
    array_compare: Dict[str, Any]
    never: EmptyDict
    always: EmptyDict
    script: Dict[str, Any]


class Transform(TypedDict, total=False):
    """ Transform definition """
    search: Dict[str, Any]
    script: Dict[str, Any]
    chain: List[Dict[str, Any]]


class ActionWebhook(TypedDict, total=False):
    """ Action webhook definition """
    scheme: str
    host: str
    port: int
    method: str
    path: str
    params: Dict[str, Any]
    headers: Dict[str, Any]
    body: str
    auth: Dict[str, Any]
    proxy: Dict[str, Any]
    timeout: str
    connection_timeout: str
    read_timeout: str
    retries: int
    retry_on_status: List[int]
    ssl: Dict[str, Any]
    webhook: Dict[str, Any]


class Action(TypedDict, total=False):
    """ Action definition """
    transform: Dict[str, Transform]
    throttle_period: str

    email: Dict[str, Any]
    index: Dict[str, Any]
    logging: Dict[str, Any]
    pagerduty: Dict[str, Any]
    slack: Dict[str, Any]
    webhook: ActionWebhook


class Watcher(metaclass=WatcherMeta):
    """
    Watcher definition
    """
    metadata: Optional[Dict[str, Any]] = None
    trigger: Optional[Trigger] = None
    input: Optional[Search] = None
    condition: Optional[Condition] = None
    actions: Optional[Dict[str, Action]] = None

    def __init__(self):
        assert self.trigger is not None, "Trigger is not defined"

    def to_es(self):
        res = {}
        if self.metadata is not None:
            res['metadata'] = self.metadata
        if self.trigger is not None:
            res['trigger'] = self.trigger
        if self.input is not None:
            res['input'] = self.input
        if self.condition is not None:
            res['condition'] = self.condition
        if self.actions is not None:
            res['actions'] = self.actions
        return res


class DeleteWatcher(Watcher):
    """
    Watcher for deleting documents matching a query
    """
    _scheme: str = "http"
    """ ES scheme """
    _host: str = "127.0.0.1"
    """ ES host """
    _port: int = 9200
    """ ES port """

    _index: Optional[str] = None
    """ Target index """
    _query: Optional[ESQuery] = None
    """ Query to match documents to delete """

    def __init__(self):
        super().__init__()

        assert self._index is not None, "Index is not defined"
        assert self._query is not None, "Query is not defined"

        self.actions = {
            "delete_doc": {
                "webhook": {
                    "scheme": self._scheme,
                    "method": "POST",
                    "host": self._host,
                    "port": self._port,
                    "path": f"/{self._index}/_delete_by_query",
                    "body": f'{{"query": {json.dumps(self._query)}}}',
                }
            }
        }


async def setup_watchers(*_, debug=False):
    """
    Setup watchers
    :param _: Unused
    :param debug: Whether to print the watcher definition
    """
    for WatcherClass in WatcherMeta.__watchers__:
        watcher = WatcherClass()
        if debug:
            from pprint import pformat
            logger.debug(
                f"`{WatcherClass.__name__}` watcher:\n {pformat(watcher.to_es(), indent=2, width=100,
                                                                compact=False, sort_dicts=False)}")
        await es.watcher.put_watch(id=WatcherClass.__name__, **watcher.to_es())
        logger.info(f"Watcher {WatcherClass.__name__} created.")
