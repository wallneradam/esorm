"""
ElasticSearch aggregation type definitions for ESORM
"""
from typing import TypedDict, Dict
from typing_extensions import TypeAlias


class ESTermsAgg(TypedDict, total=False):
    """ Represents the parameters for a terms aggregation in Elasticsearch. """
    field: str
    size: int
    order: Dict[str, str]


class ESAvgAgg(TypedDict):
    """Represents the parameters for an average aggregation in Elasticsearch. """
    field: str


class ESSumAgg(TypedDict):
    """ Represents the parameters for a sum aggregation in Elasticsearch. """
    field: str


class ESMinAgg(TypedDict):
    """ Represents the parameters for a minimum aggregation in Elasticsearch. """
    field: str


class ESMaxAgg(TypedDict):
    """ Represents the parameters for a maximum aggregation in Elasticsearch. """
    field: str


class ESAggregation(TypedDict, total=False):
    """ Holds all types of aggregations supported """
    terms: ESTermsAgg
    avg: ESAvgAgg
    sum: ESSumAgg
    min: ESMinAgg
    max: ESMaxAgg


ESAggregations: TypeAlias = Dict[str, ESAggregation]
""" ElasticSearch aggregations type definition """
