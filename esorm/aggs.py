"""
ElasticSearch aggregation type definitions for ESORM
"""
from typing import TypedDict, Dict, List, Union
from typing_extensions import TypeAlias


#
# Queries
#

class ESAggTermParams(TypedDict, total=False):
    """
    Represents the parameters for a terms aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """
    size: int
    """ The number of buckets to return. """
    order: Dict[str, str]
    """ The order of the buckets. """


class ESAggFieldParams(TypedDict):
    """
    Represents field parameter in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """


class ESAggExtendedBounds(TypedDict):
    """
    Represents the parameters for extended bounds in Elasticsearch.
    """
    min: int
    """ The minimum value. """
    max: int
    """ The maximum value. """


class ESAggHistogramParams(TypedDict, total=False):
    """
    Represents the parameters for a histogram aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """
    interval: int
    """ The interval of the histogram. """
    min_doc_count: int
    """ The minimum number of documents in a bucket. """
    extended_bounds: ESAggExtendedBounds
    """ The extended bounds of the histogram. """


class ESAgg(TypedDict, total=False):
    """
    Holds all types of aggregations supported
    """
    terms: ESAggTermParams
    """ Terms aggregation """
    avg: ESAggFieldParams
    """ Average aggregation """
    sum: ESAggFieldParams
    """ Sum aggregation """
    min: ESAggFieldParams
    """ Minimum aggregation """
    max: ESAggFieldParams
    """ Maximum aggregation """
    histogram: ESAggHistogramParams
    """ Histogram aggregation """


ESAggs: TypeAlias = Dict[str, ESAgg]
""" ElasticSearch aggregations type definition """


#
# Responses
#

class ESAggBucketResponse(TypedDict):
    """
    Represents a single bucket in a bucket aggregation.
    """
    key: str
    """ The key of the bucket. """
    doc_count: int
    """ The number of documents in this bucket. """


class ESAggTermsResponse(TypedDict):
    """
    Represents the response for a terms aggregation.
    """
    buckets: List[ESAggBucketResponse]
    """ A list of buckets in the terms aggregation. """


class ESAggValueResponse(TypedDict):
    """
    Represents the response for an average, min, or max aggregation.
    """
    value: float
    """ The average, min, or max value. """


class ESAggHistogramBucketresponse(TypedDict):
    """
    Represents a bucket in a histogram aggregation.
    """
    key: float
    """ Numeric key corresponding to the bucket's range. """
    doc_count: int
    """ The number of documents in this bucket."""


class ESAggHistogramResponse(TypedDict):
    """
    Represents the response for a histogram aggregation.
    """
    buckets: List[ESAggHistogramBucketresponse]
    """ A list of buckets in the histogram aggregation. """


ESAggsResponse: TypeAlias = Dict[str, Union[
    ESAggValueResponse,
    ESAggTermsResponse,
    ESAggHistogramResponse
]]
""" ElasticSearch aggregations response type definition """
