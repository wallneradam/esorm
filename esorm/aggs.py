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


class ESDateHistogramParams(TypedDict, total=False):
    """
    Represents the parameters for a date histogram aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """
    calendar_interval: str
    """ Calendar-aware interval (e.g., "1M", "1d", "month", "day"). """
    fixed_interval: str
    """ Fixed interval in time units (e.g., "30s", "1h", "3h"). """
    format: str
    """ Date format for bucket keys (e.g., "yyyy-MM-dd"). """
    time_zone: str
    """ Time zone for bucketing (e.g., "+01:00", "America/Los_Angeles"). """
    min_doc_count: int
    """ The minimum number of documents in a bucket. """
    extended_bounds: ESAggExtendedBounds
    """ The extended bounds of the histogram. """
    offset: str
    """ Offset to shift bucket boundaries (e.g., "+6h", "-1d"). """
    keyed: bool
    """ Whether to return buckets as a hash instead of an array. """
    missing: str
    """ Value to use for documents missing the aggregation field. """
    order: Dict[str, str]
    """ Order of the buckets (e.g., {"_key": "asc"}). """


class ESAggPercentilesParams(TypedDict, total=False):
    """
    Represents the parameters for a percentiles aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """
    percents: List[float]
    """ Array of percentiles to compute (e.g., [25, 50, 75, 95, 99]). """
    keyed: bool
    """ Whether to return percentiles as a hash instead of an array. """
    tdigest: Dict[str, Union[int, float]]
    """ T-Digest algorithm configuration. """
    hdr: Dict[str, int]
    """ HDR histogram configuration. """
    missing: float
    """ Value to use for documents missing the aggregation field. """


class ESAggCardinalityParams(TypedDict, total=False):
    """
    Represents the parameters for a cardinality aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """
    precision_threshold: int
    """ Precision threshold for the cardinality calculation (default: 3000). """
    missing: Union[str, int, float]
    """ Value to use for documents missing the aggregation field. """


class ESAggStatsParams(TypedDict, total=False):
    """
    Represents the parameters for a stats aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """
    missing: float
    """ Value to use for documents missing the aggregation field. """


class ESAggExtendedStatsParams(TypedDict, total=False):
    """
    Represents the parameters for an extended stats aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """
    sigma: float
    """ Number of standard deviations for bounds (default: 2.0). """
    missing: float
    """ Value to use for documents missing the aggregation field. """


class ESAggValueCountParams(TypedDict, total=False):
    """
    Represents the parameters for a value count aggregation in Elasticsearch.
    """
    field: str
    """ The field to count values for. """


class ESAggRangeParams(TypedDict, total=False):
    """
    Represents the parameters for a range aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """
    ranges: List[Dict[str, float]]
    """ Array of range buckets (e.g., [{"to": 50}, {"from": 50, "to": 100}, {"from": 100}]). """
    keyed: bool
    """ Whether to return buckets as a hash instead of an array. """


class ESAggDateRangeParams(TypedDict, total=False):
    """
    Represents the parameters for a date range aggregation in Elasticsearch.
    """
    field: str
    """ The field to aggregate on. """
    ranges: List[Dict[str, str]]
    """ Array of date range buckets (e.g., [{"to": "2020-01-01"}, {"from": "2020-01-01"}]). """
    format: str
    """ Date format for parsing range values. """
    time_zone: str
    """ Time zone for date calculations. """
    keyed: bool
    """ Whether to return buckets as a hash instead of an array. """


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
    date_histogram: ESDateHistogramParams
    """ Date histogram aggregation """
    cardinality: ESAggCardinalityParams
    """ Cardinality aggregation (count distinct values) """
    percentiles: ESAggPercentilesParams
    """ Percentiles aggregation """
    stats: ESAggStatsParams
    """ Stats aggregation (count, min, max, avg, sum) """
    extended_stats: ESAggExtendedStatsParams
    """ Extended stats aggregation (includes variance, std deviation) """
    value_count: ESAggValueCountParams
    """ Value count aggregation """
    range: ESAggRangeParams
    """ Range aggregation """
    date_range: ESAggDateRangeParams
    """ Date range aggregation """


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


class ESAggCardinalityResponse(TypedDict):
    """
    Represents the response for a cardinality aggregation.
    """
    value: int
    """ The approximate count of distinct values. """


class ESAggPercentilesResponse(TypedDict):
    """
    Represents the response for a percentiles aggregation.
    """
    values: Dict[str, float]
    """ Dictionary of percentile values (e.g., {"25.0": 10.5, "50.0": 25.0}). """


class ESAggStatsResponse(TypedDict):
    """
    Represents the response for a stats aggregation.
    """
    count: int
    """ Number of values. """
    min: float
    """ Minimum value. """
    max: float
    """ Maximum value. """
    avg: float
    """ Average value. """
    sum: float
    """ Sum of all values. """


class ESAggExtendedStatsResponse(TypedDict):
    """
    Represents the response for an extended stats aggregation.
    """
    count: int
    """ Number of values. """
    min: float
    """ Minimum value. """
    max: float
    """ Maximum value. """
    avg: float
    """ Average value. """
    sum: float
    """ Sum of all values. """
    sum_of_squares: float
    """ Sum of squares. """
    variance: float
    """ Variance of the values. """
    variance_population: float
    """ Population variance. """
    variance_sampling: float
    """ Sampling variance. """
    std_deviation: float
    """ Standard deviation. """
    std_deviation_population: float
    """ Population standard deviation. """
    std_deviation_sampling: float
    """ Sampling standard deviation. """
    std_deviation_bounds: Dict[str, float]
    """ Upper and lower bounds based on std deviation. """


class ESAggRangeBucketResponse(TypedDict):
    """
    Represents a single bucket in a range aggregation.
    """
    key: str
    """ The key of the bucket. """
    from_: float
    """ The start of the range (inclusive). """
    to: float
    """ The end of the range (exclusive). """
    doc_count: int
    """ The number of documents in this bucket. """


class ESAggRangeResponse(TypedDict):
    """
    Represents the response for a range aggregation.
    """
    buckets: List[ESAggRangeBucketResponse]
    """ A list of buckets in the range aggregation. """


class ESAggDateRangeBucketResponse(TypedDict):
    """
    Represents a single bucket in a date range aggregation.
    """
    key: str
    """ The key of the bucket. """
    from_: str
    """ The start date of the range (inclusive). """
    from_as_string: str
    """ The start date as formatted string. """
    to: str
    """ The end date of the range (exclusive). """
    to_as_string: str
    """ The end date as formatted string. """
    doc_count: int
    """ The number of documents in this bucket. """


class ESAggDateRangeResponse(TypedDict):
    """
    Represents the response for a date range aggregation.
    """
    buckets: List[ESAggDateRangeBucketResponse]
    """ A list of buckets in the date range aggregation. """


ESAggsResponse: TypeAlias = Dict[str, Union[
    ESAggValueResponse,
    ESAggTermsResponse,
    ESAggHistogramResponse,
    ESAggCardinalityResponse,
    ESAggPercentilesResponse,
    ESAggStatsResponse,
    ESAggExtendedStatsResponse,
    ESAggRangeResponse,
    ESAggDateRangeResponse
]]
""" ElasticSearch aggregations response type definition """
