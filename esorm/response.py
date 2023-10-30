"""
This module contains type definitions for the response from Elasticsearch.
"""
from typing import TypedDict, List, Dict, Union, Any


class Hit(TypedDict):
    """Represents a single hit (result) from Elasticsearch."""
    _index: str
    _type: str
    _id: str
    _score: Union[float, None]
    _source: Dict[str, Any]


class Hits(TypedDict):
    """Represents the hits section of the Elasticsearch response."""
    total: Dict[str, int]
    max_score: Union[float, None]
    hits: List[Hit]


class ESResponse(TypedDict):
    """Represents the overall structure of an Elasticsearch response."""
    took: int
    timed_out: bool
    _shards: Dict[str, int]
    hits: Hits
