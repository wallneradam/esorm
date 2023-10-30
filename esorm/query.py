"""
Elasticsearch query type definitions for ESORM
"""
from typing import TypedDict, List, Dict, Union
from typing_extensions import TypeAlias


class ESRange(TypedDict, total=False):
    """
    Range query structure
    """
    gt: Union[int, float, str]
    gte: Union[int, float, str]
    lt: Union[int, float, str]
    lte: Union[int, float, str]


class FieldRange(TypedDict):
    """
    Range qyery field
    """
    range: Dict[str, ESRange]


class ESTerm(TypedDict, total=False):
    """Represents the parameters for a term query in Elasticsearch."""
    value: Union[str, int, float]
    boost: Union[int, float]


class FieldTerm(TypedDict):
    """Represents a term query for exact value matching in Elasticsearch."""
    term: Dict[str, ESTerm]


class FieldTerms(TypedDict):
    """Represents a terms query for exact value matching in Elasticsearch."""
    terms: Dict[str, List[Union[str, int, float]]]


class ESMatch(TypedDict, total=False):
    """Represents the parameters for a match query in Elasticsearch."""
    query: Union[str, int, float]
    operator: str
    boost: Union[int, float]
    analyzer: str
    fuzziness: Union[int, str]
    prefix_length: int
    max_expansions: int
    zero_terms_query: str


class FieldMatch(TypedDict):
    """Represents a match query for matching based on the provided text in Elasticsearch."""
    match: Dict[str, ESMatch]


class ESMatchPhrase(TypedDict, total=False):
    """Represents the parameters for a match_phrase query in Elasticsearch."""
    query: str
    analyzer: str
    boost: Union[int, float]
    slop: int


class FieldMatchPhrase(TypedDict):
    """Represents a match_phrase query for exact phrase matching in Elasticsearch."""
    match_phrase: Dict[str, ESMatchPhrase]


class ESExists(TypedDict):
    """Represents an exists query to check if a field exists."""
    field: str


class FieldExists(TypedDict):
    """Represents an exists query to check if a field exists in Elasticsearch."""
    exists: ESExists


class ESWildcard(TypedDict, total=False):
    """Represents a wildcard query for pattern matching in Elasticsearch."""
    value: str  # The pattern to search for, e.g., "te?t" or "test*"
    boost: float  # Optional boosting value for the query
    rewrite: str  # Optional, method used to rewrite the query (e.g., "constant_score", "scoring_boolean")
    case_insensitive: bool  # Optional, whether the query should be case insensitive


class FieldWildcard(TypedDict):
    """Represents a wildcard query for pattern matching in Elasticsearch."""
    wildcard: Dict[str, ESWildcard]


class ESPrefix(TypedDict, total=False):
    """Represents a prefix query for prefix matching in Elasticsearch."""
    value: str  # The prefix value to search for
    boost: float  # Optional boosting value for the query
    rewrite: str  # Optional, method used to rewrite the query (e.g., "constant_score", "scoring_boolean")


class FieldPrefix(TypedDict):
    """Represents a prefix query for prefix matching in Elasticsearch."""
    prefix: Dict[str, ESPrefix]


class ESFuzzy(TypedDict, total=False):
    """Represents a fuzzy query for approximate matching in Elasticsearch."""
    value: str  # The value to search for
    fuzziness: Union[int, str]  # Allowed amount of fuzziness or "AUTO"
    prefix_length: int  # Length of the prefix that remains unchanged
    max_expansions: int  # Maximum number of expansions
    transpositions: bool  # Allow or disallow character transpositions (e.g., "ab" -> "ba")
    boost: float  # Boosting value for the query


class FieldFuzzy(TypedDict):
    """Represents a fuzzy query for approximate matching in Elasticsearch."""
    fuzzy: Dict[str, ESFuzzy]


class ESGeoDistance(TypedDict, total=False):
    """Represents a geo_distance query for distance-based geospatial queries in Elasticsearch."""
    distance: Union[str, float]  # e.g., "12km" or 12.0
    distance_type: str  # Optional, e.g., "arc" or "plane"
    validation_method: str  # Optional, e.g., "STRICT" or "IGNORE_MALFORMED"
    location_field: str  # The field name containing the geospatial data
    location: Union[
        Dict[str, float],  # e.g., {"lat": 52.3760, "lon": 4.894}
        str  # e.g., "52.3760, 4.894"
    ]


class FieldGeoDistance(TypedDict):
    """Represents a geo_distance query for distance-based geospatial queries in Elasticsearch."""
    geo_distance: Dict[str, ESGeoDistance]


class ESMatchAll(TypedDict, total=False):
    """Represents a match_all query for matching all documents in Elasticsearch."""
    boost: float


class FieldMatchAll(TypedDict):
    """Represents a match_all query for matching all documents in Elasticsearch."""
    match_all: ESMatchAll


class ESMatchNone(TypedDict):
    """Represents a match_none query for matching no documents in Elasticsearch."""
    pass


class FieldESMatchNone(TypedDict):
    """Represents a match_none query for matching no documents in Elasticsearch."""
    match_none: ESMatchNone

#: Represents must queries in Elasticsearch
ESMust: TypeAlias = List[
    Union[
        FieldRange, FieldTerm, FieldTerms, FieldMatch, FieldMatchPhrase, FieldExists, FieldWildcard, FieldPrefix,
        FieldFuzzy, FieldGeoDistance,
        FieldMatchAll, FieldESMatchNone,
        'FieldBool'
    ]
]

#: Represents filter queries in Elasticsearch
ESFilter: TypeAlias = List[
    Union[
        FieldRange, FieldTerm, FieldTerms, FieldMatch, FieldMatchPhrase, FieldExists, FieldWildcard, FieldPrefix,
        FieldFuzzy, FieldGeoDistance,
        FieldMatchAll, FieldESMatchNone,
        'FieldBool'
    ]
]

#: Represents should queries in Elasticsearch
ESShould: TypeAlias = List[
    Union[
        FieldRange, FieldTerm, FieldTerms, FieldMatch, FieldMatchPhrase, FieldExists, FieldWildcard, FieldPrefix,
        FieldFuzzy, FieldGeoDistance,
        FieldMatchAll, FieldESMatchNone,
        'FieldBool'
    ]
]

#: Represents must_not queries in Elasticsearch
ESMustNot: TypeAlias = List[
    Union[
        FieldRange, FieldTerm, FieldTerms, FieldMatch, FieldMatchPhrase, FieldExists, FieldWildcard, FieldPrefix,
        FieldFuzzy, FieldGeoDistance,
        FieldMatchAll, FieldESMatchNone,
        'FieldBool'
    ]
]


class ESBool(TypedDict, total=False):  # We use total=False, because not every key is required
    """
    Bool query structure
    """
    must: ESMust
    filter: ESFilter
    should: ESShould
    must_not: ESMustNot
    minimum_should_match: Union[int, str]
    boost: float  # Boosting value for the query


class FieldBool(TypedDict):
    """Represents a bool query for combining other queries in Elasticsearch."""
    bool: ESBool


class ESQuery(TypedDict, total=False):
    """
    Elasticsearch query structure
    """
    bool: ESBool
    match: Dict[str, ESMatch]
    match_phrase: Dict[str, ESMatchPhrase]
    term: Dict[str, ESTerm]
    prefix: Dict[str, ESPrefix]
    fuzzy: Dict[str, ESFuzzy]
    wildcard: Dict[str, ESWildcard]
    geo_distance: Dict[str, ESGeoDistance]
    exists: ESExists
    match_all: ESMatchAll
    match_none: ESMatchNone
