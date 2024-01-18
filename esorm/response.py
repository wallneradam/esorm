"""
This module contains type definitions for the response from Elasticsearch.
"""
from typing import TypedDict, List, Dict, Union, Any
from . import aggs


class Hit(TypedDict):
    """
    Represents a single hit (result) from Elasticsearch.
    """
    _index: str
    """ The index the hit is from. """
    _type: str
    """ The type of the hit. """
    _id: str
    """ The id of the hit. """
    _score: Union[float, None]
    """ The score of the hit. """
    _source: Dict[str, Any]
    """ The source of the hit. """


class Hits(TypedDict):
    """
    Represents the hits section of the Elasticsearch response.
    """
    total: Dict[str, int]
    """ The total number of hits. """
    max_score: Union[float, None]
    """ The maximum score of the hits. """
    hits: List[Hit]
    """ List of hits. """


class ESResponse(TypedDict):
    """
    Represents the overall structure of an Elasticsearch response.
    """
    took: int
    """ The time in milliseconds it took to execute the query. """
    timed_out: bool
    """ Whether the query timed out. """
    _shards: Dict[str, int]
    """ The number of shards the query was executed on. """
    hits: Hits
    """ The hits section of the response. """
    aggregations: aggs.ESAggsResponse
    """ The aggregations section of the response. """
