"""
ElasticSearch aggregation type definitions for ESORM
"""
from typing import TypedDict, Dict, List, Union
from typing_extensions import TypeAlias


#
# Queries
#

class ESTermsAgg(TypedDict, total=False):
    """
    Represents the parameters for a terms aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """
    size: int
    """ The number of buckets to return. """
    order: Dict[str, str]
    """ The order of the buckets. """


class ESAvgAgg(TypedDict):
    """
    Represents the parameters for an average aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """


class ESSumAgg(TypedDict):
    """
    Represents the parameters for a sum aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """


class ESMinAgg(TypedDict):
    """
    Represents the parameters for a minimum aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """


class ESMaxAgg(TypedDict):
    """
    Represents the parameters for a maximum aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """


class ESExtendedBounds(TypedDict):
    """
    Represents the parameters for extended bounds in Elasticsearch.
    """
    min: int
    """ The minimum value. """
    max: int
    """ The maximum value. """


class ESHistogramAgg(TypedDict, total=False):
    """
    Represents the parameters for a histogram aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """
    interval: int
    """ The interval of the histogram. """
    min_doc_count: int
    """ The minimum number of documents in a bucket. """
    extended_bounds: ESExtendedBounds
    """ The extended bounds of the histogram. """


class ESAgg(TypedDict, total=False):
    """
    Holds all types of aggregations supported
    """
    terms: ESTermsAgg
    """ Terms aggregation """
    avg: ESAvgAgg
    """ Average aggregation """
    sum: ESSumAgg
    """ Sum aggregation """
    min: ESMinAgg
    """ Minimum aggregation """
    max: ESMaxAgg
    """ Maximum aggregation """
    histogram: ESHistogramAgg
    """ Histogram aggregation """


ESAggs: TypeAlias = Dict[str, ESAgg]
""" ElasticSearch aggregations type definition """


#
# Responses
#

class ESBucket(TypedDict):
    """
    Represents a single bucket in a bucket aggregation.
    """
    key: str
    """ The key of the bucket. """
    doc_count: int
    """ The number of documents in this bucket. """


class ESTermsAggResponse(TypedDict):
    """
    Represents the response for a terms aggregation.
    """
    buckets: List[ESBucket]
    """ A list of buckets in the terms aggregation. """


class ESAvgMinMaxAggResponse(TypedDict):
    """
    Represents the response for an average, min, or max aggregation.
    """
    value: float
    """ The average, min, or max value. """


class ESHistogramBucket(TypedDict):
    """
    Represents a bucket in a histogram aggregation.
    """
    key: float
    """ Numeric key corresponding to the bucket's range. """
    doc_count: int
    """ The number of documents in this bucket."""


class ESHistogramAggResponse(TypedDict):
    """
    Represents the response for a histogram aggregation.
    """
    buckets: List[ESHistogramBucket]
    """ A list of buckets in the histogram aggregation. """


ESAggsResponse: TypeAlias = Dict[str, Union[
    ESTermsAggResponse,
    ESAvgMinMaxAggResponse,
    ESHistogramAggResponse
]]
""" ElasticSearch aggregations response type definition """
