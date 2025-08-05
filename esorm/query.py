"""
Elasticsearch query type definitions for ESORM
"""
from typing import Literal, TypedDict, List, Dict, Union, Optional, Any
from typing_extensions import TypeAlias
from . import aggs


class ESRange(TypedDict, total=False):
    """
    Range query structure
    """
    gt: Union[int, float, str]
    """ Greater than """
    gte: Union[int, float, str]
    """ Greater than or equal """
    lt: Union[int, float, str]
    """ Less than """
    lte: Union[int, float, str]
    """ Less than or equal """


class FieldRange(TypedDict):
    """
    Range qyery field
    """
    range: Dict[str, ESRange]
    """ Range query structure """


class ESTerm(TypedDict, total=False):
    """
    Represents the parameters for a term query in Elasticsearch.
    """
    value: Union[str, int, float]
    """ The value to search for. """
    boost: Union[int, float]
    """ Optional boosting value for the query. """


class FieldTerm(TypedDict):
    """
    Represents a term query for exact value matching in Elasticsearch.
    """
    term: Dict[str, ESTerm]
    """ Term query structure """


class FieldTerms(TypedDict):
    """
    Represents a terms query for exact value matching in Elasticsearch.
    """
    terms: Dict[str, List[Union[str, int, float]]]
    """ Terms query structure """


class ESMatch(TypedDict, total=False):
    """
    Represents the parameters for a match query in Elasticsearch.
    """
    query: Union[str, int, float]
    """ The value to search for. """
    operator: str
    """ The operator to use for the query. """
    boost: Union[int, float]
    """ Optional boosting value for the query. """
    analyzer: str
    """ Optional analyzer to use for the query. """
    fuzziness: Union[int, str]
    """ Optional fuzziness value for the query. """
    prefix_length: int
    """ Optional prefix length for the query. """
    max_expansions: int
    """ Optional maximum number of expansions for the query. """
    zero_terms_query: str
    """ Optional zero terms query for the query. """


class FieldMatch(TypedDict):
    """
    Represents a match query for matching based on the provided text in Elasticsearch.
    """
    match: Dict[str, ESMatch]
    """ Match query structure """


class ESMatchPhrase(TypedDict, total=False):
    """
    Represents the parameters for a match_phrase query in Elasticsearch.
    """
    query: str
    """ The value to search for. """
    analyzer: str
    """ Optional analyzer to use for the query. """
    boost: Union[int, float]
    """ Optional boosting value for the query. """
    slop: int
    """ Optional slop value for the query. """


class FieldMatchPhrase(TypedDict):
    """
    Represents a match_phrase query for exact phrase matching in Elasticsearch.
    """
    match_phrase: Dict[str, ESMatchPhrase]
    """ Match phrase query structure """


class ESExists(TypedDict):
    """
    Represents an exists query to check if a field exists.
    """
    field: str
    """ The field to check. """


class FieldExists(TypedDict):
    """
    Represents an exists query to check if a field exists in Elasticsearch.
    """
    exists: ESExists
    """ Exists query structure """


class ESWildcard(TypedDict, total=False):
    """
    Represents a wildcard query for pattern matching in Elasticsearch.
    """
    value: str
    """ The pattern to search for. e.g., "te?t" or "test*" """
    boost: float
    """ Optional boosting value for the query """
    rewrite: str
    """ Optional, method used to rewrite the query (e.g., "constant_score", "scoring_boolean") """
    case_insensitive: bool
    """ Optional, whether the query is case insensitive. """


class FieldWildcard(TypedDict):
    """
    Represents a wildcard query for pattern matching in Elasticsearch.
    """
    wildcard: Dict[str, ESWildcard]
    """ Wildcard query structure """


class ESPrefix(TypedDict, total=False):
    """
    Represents a prefix query for prefix matching in Elasticsearch.
    """
    value: str
    """ The prefix to search for. """
    boost: float
    """ Optional boosting value for the query """
    rewrite: str
    """ Optional, method used to rewrite the query (e.g., "constant_score", "scoring_boolean") """


class FieldPrefix(TypedDict):
    """
    Represents a prefix query for prefix matching in Elasticsearch.
    """
    prefix: Dict[str, ESPrefix]
    """ Prefix query structure """


class ESFuzzy(TypedDict, total=False):
    """
    Represents a fuzzy query for approximate matching in Elasticsearch.
    """
    value: str
    """ The value to search for. """
    fuzziness: Union[int, str]
    """ Fuzziness value for the query """
    prefix_length: int
    """ Prefix length for the query """
    max_expansions: int
    """ Maximum number of expansions for the query """
    transpositions: bool
    """ Whether to allow transpositions for the query """
    boost: float
    """ Optional boosting value for the query """


class FieldFuzzy(TypedDict):
    """
    Represents a fuzzy query for approximate matching in Elasticsearch.
    """
    fuzzy: Dict[str, ESFuzzy]
    """ Fuzzy query structure """


class ESGeoDistance(TypedDict, total=False):
    """
    Represents a geo_distance query for distance-based geospatial queries in Elasticsearch.
    """
    distance: Union[str, float]
    """ The distance to search for. """
    distance_type: str
    """ The distance type to use for the query. """
    validation_method: str
    """ The validation method to use for the query. """
    location_field: str
    """ The field containing the location to search from. """
    location: Union[
        Dict[str, float],
        str
    ]
    """ The location to search from. """


class FieldGeoDistance(TypedDict):
    """
    Represents a geo_distance query for distance-based geospatial queries in Elasticsearch.
    """
    geo_distance: Dict[str, ESGeoDistance]
    """ Geo distance query structure """


class ESMatchAll(TypedDict, total=False):
    """
    Represents a match_all query for matching all documents in Elasticsearch.
    """
    boost: float
    """ Optional boosting value for the query """


class FieldMatchAll(TypedDict):
    """
    Represents a match_all query for matching all documents in Elasticsearch.
    """
    match_all: ESMatchAll
    """ Match all query structure """


class ESMatchNone(TypedDict):
    """
    Represents a match_none query for matching no documents in Elasticsearch.
    """
    pass


class ESMultiMatch(TypedDict, total=False):
    """
    Represents a multi_match query in Elasticsearch.
    """
    query: str
    """ Query string """
    
    type: Optional[Literal[
        "best_fields", 
        "most_fields",
        "cross_fields", 
        "phrase", 
        "phrase_prefix", 
        "bool_prefix",
    ]]
    """ Type of multi_match query """
    
    fields: Optional[List[str]]
    """ Optional fields list to match on """
    


class FieldESMatchNone(TypedDict):
    """
    Represents a match_none query for matching no documents in Elasticsearch.
    """
    match_none: ESMatchNone
    """ Match none query structure """


class ESKnnQuery(TypedDict, total=False):
    """
    Represents the parameters for a knn query in Elasticsearch.
    """
    field: str
    """The field to search on."""
    query_vector: List[float]
    """The query vector."""
    k: int
    """The number of neighbors to return."""
    num_candidates: Optional[int]
    """The number of candidates to consider."""
    filter: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
    """Optional filter to apply to the search."""


class FieldKnn(TypedDict):
    """
    Represents a knn query for vector similarity search in Elasticsearch.
    """
    knn: ESKnnQuery
    """KNN query structure"""


#: Represents must queries in Elasticsearch
ESMust: TypeAlias = List[
    Union[
        FieldRange, FieldTerm, FieldTerms, FieldMatch, FieldMatchPhrase, FieldExists, FieldWildcard, FieldPrefix,
        FieldFuzzy, FieldGeoDistance,
        FieldMatchAll, FieldESMatchNone,
        'FieldBool', FieldKnn
    ]
]
""" Represents must queries in Elasticsearch """

#: Represents filter queries in Elasticsearch
ESFilter: TypeAlias = List[
    Union[
        FieldRange, FieldTerm, FieldTerms, FieldMatch, FieldMatchPhrase, FieldExists, FieldWildcard, FieldPrefix,
        FieldFuzzy, FieldGeoDistance,
        FieldMatchAll, FieldESMatchNone,
        'FieldBool'
    ]
]
""" Represents filter queries in Elasticsearch """

#: Represents should queries in Elasticsearch
ESShould: TypeAlias = List[
    Union[
        FieldRange, FieldTerm, FieldTerms, FieldMatch, FieldMatchPhrase, FieldExists, FieldWildcard, FieldPrefix,
        FieldFuzzy, FieldGeoDistance,
        FieldMatchAll, FieldESMatchNone,
        'FieldBool', FieldKnn
    ]
]
""" Represents should queries in Elasticsearch """

#: Represents must_not queries in Elasticsearch
ESMustNot: TypeAlias = List[
    Union[
        FieldRange, FieldTerm, FieldTerms, FieldMatch, FieldMatchPhrase, FieldExists, FieldWildcard, FieldPrefix,
        FieldFuzzy, FieldGeoDistance,
        FieldMatchAll, FieldESMatchNone,
        'FieldBool'
    ]
]
""" Represents must_not queries in Elasticsearch """


class ESBool(TypedDict, total=False):  # We use total=False, because not every key is required
    """
    Bool query structure
    """
    must: ESMust
    """ Must queries """
    filter: ESFilter
    """ Filter queries """
    should: ESShould
    """ Should queries """
    must_not: ESMustNot
    """ Must not queries """
    minimum_should_match: Union[int, str]
    """ Minimum number of should queries to match """
    boost: float
    """ Boosting value for the query """


class FieldBool(TypedDict):
    """
    Represents a bool query for combining other queries in Elasticsearch.
    """
    bool: ESBool
    """ Bool query structure """


class ESQuery(TypedDict, total=False):
    """
    Elasticsearch query structure
    """
    bool: ESBool
    """ Bool query structure """
    match: Dict[str, ESMatch]
    """ Match query structure """
    match_phrase: Dict[str, ESMatchPhrase]
    """ Match phrase query structure """
    term: Dict[str, ESTerm]
    """ Term query structure """
    prefix: Dict[str, ESPrefix]
    """ Prefix query structure """
    fuzzy: Dict[str, ESFuzzy]
    """ Fuzzy query structure """
    wildcard: Dict[str, ESWildcard]
    """ Wildcard query structure """
    geo_distance: Dict[str, ESGeoDistance]
    """ Geo distance query structure """
    exists: ESExists
    """ Exists query structure """
    match_all: ESMatchAll
    """ Match all query structure """
    match_none: ESMatchNone
    """ Match none query structure """
    multi_match: ESMultiMatch
    """ MultiMatch query structure """
    knn: ESKnnQuery
    """ KNN query structure """
    aggs: aggs.ESAggs
    """ Aggregations query structure """
