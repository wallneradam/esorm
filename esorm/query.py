"""
Elasticsearch (and OpenSearch) query type definitions for ESORM

Every query clause has the same type: :class:`ESQuery`. It is used for the whole query, and also for the clauses
of compound queries (e.g. ``bool.must``, ``nested.query``), so this is the only type you need to annotate your
query variables::

    from esorm.query import ESQuery

    must: list[ESQuery] = []
    if name:
        must.append({'match': {'name': name}})
    query: ESQuery = {'bool': {'must': must}}
"""
from typing import Literal, Union, Any, List, Mapping, Sequence, Iterator, Protocol, TypeVar, SupportsIndex, overload, \
    get_args
from typing_extensions import TypeAlias, TypedDict, Required

from datetime import date
from decimal import Decimal
from uuid import UUID

__all__ = [
    'ESQuery',
    'ESValue', 'ESRangeValue', 'ESScript', 'ESGeoPoint', 'SequenceNotStr',
    # Leaf query parameters
    'ESTerm', 'ESTermsLookup', 'ESTermsSet', 'ESRange', 'ESExists', 'ESIds', 'ESPrefix', 'ESWildcard', 'ESRegexp',
    'ESFuzzy',
    'ESMatch', 'ESMatchPhrase', 'ESMatchPhrasePrefix', 'ESMatchBoolPrefix', 'ESMultiMatch', 'ESCombinedFields',
    'ESQueryString', 'ESSimpleQueryString',
    'ESMatchAll', 'ESMatchNone',
    'ESGeoBoundingBox',
    'ESKnnQuery', 'ESSemantic', 'ESSparseVector',
    # Compound and joining query parameters
    'ESBool', 'ESBoosting', 'ESConstantScore', 'ESDisMax', 'ESFunctionScore', 'ESScoreFunction',
    'ESFieldValueFactor', 'ESRandomScore', 'ESScriptScoreFunction',
    'ESNested', 'ESHasChild', 'ESHasParent', 'ESParentId',
    # Specialized query parameters
    'ESScriptDef', 'ESScriptQuery', 'ESScriptScore', 'ESDistanceFeature', 'ESMoreLikeThis', 'ESRankFeature',
    'ESPercolate', 'ESWrapper', 'ESPinned', 'ESHybrid',
    # Backward compatible names
    'ESMust', 'ESFilter', 'ESShould', 'ESMustNot', 'ESGeoDistance',
    'FieldRange', 'FieldTerm', 'FieldTerms', 'FieldMatch', 'FieldMatchPhrase', 'FieldExists', 'FieldWildcard',
    'FieldPrefix', 'FieldFuzzy', 'FieldGeoDistance', 'FieldMatchAll', 'FieldESMatchNone', 'FieldKnn', 'FieldBool',
]

_T_co = TypeVar('_T_co', covariant=True)


class SequenceNotStr(Protocol[_T_co]):
    """
    A sequence (list, tuple, ...) which is not a string

    A string is a sequence of strings in Python, so ``Sequence[str]`` accepts ``'abc'`` too. This type doesn't, so
    ``{'terms': {'tag': 'abc'}}`` is reported, while ``{'terms': {'tag': ['abc']}}`` is accepted.
    """

    @overload
    def __getitem__(self, index: SupportsIndex, /) -> _T_co: ...

    @overload
    def __getitem__(self, index: slice, /) -> Sequence[_T_co]: ...

    def __contains__(self, value: object, /) -> bool: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[_T_co]: ...

    def index(self, value: Any, start: int = 0, stop: int = ..., /) -> int: ...

    def count(self, value: Any, /) -> int: ...

    def __reversed__(self) -> Iterator[_T_co]: ...


def _sequence_not_str_schema(_cls: Any, source: Any, handler: Any) -> Any:
    """
    Pydantic validates it as a list, so the query types can be used in pydantic models (e.g. FastAPI request bodies)
    """
    args = get_args(source)
    return handler(List[args[0]] if args else List[Any])


# It is set outside the class body, because everything in a protocol class body would be a protocol member
setattr(SequenceNotStr, '__get_pydantic_core_schema__', classmethod(_sequence_not_str_schema))


#
# Value types
#
# NOTE: builtin names which are also query keys (e.g. bool, range, filter) must not be used in the TypedDict
# classes, because inside the class body they mean the key, not the builtin. So these aliases are used instead.
#

ESValue: TypeAlias = Union[str, int, float, bool, date, UUID, Decimal]
""" Field value, date, datetime, UUID and Decimal are serialized by the ES client """

ESRangeValue: TypeAlias = Union[str, int, float, date, Decimal]
""" Range boundary: number, date, datetime or string (e.g. date math like "now-1d/d") """

_Text: TypeAlias = Union[str, int, float, bool, date, UUID, Decimal]
""" Full text query value """

_Boolean: TypeAlias = bool
_Float: TypeAlias = float
_Number: TypeAlias = Union[int, float]
_Fuzziness: TypeAlias = Union[int, str]
_MinimumShouldMatch: TypeAlias = Union[int, str]
_Operator: TypeAlias = Literal['and', 'or', 'AND', 'OR']
_ZeroTermsQuery: TypeAlias = Literal['none', 'all']
_Fields: TypeAlias = Sequence[str]
_Any: TypeAlias = Mapping[str, Any]


class ESScriptDef(TypedDict, total=False):
    """
    Script definition
    """
    source: str
    """ The script source """
    id: str
    """ The id of a stored script """
    lang: str
    """ The script language, default is painless """
    params: Mapping[str, Any]
    """ Parameters of the script """


ESScript: TypeAlias = Union[str, ESScriptDef]
""" Script: source string or script definition """


class _ESLatLon(TypedDict):
    """
    Geo point as an object
    """
    lat: float
    """ Latitude """
    lon: float
    """ Longitude """


ESGeoPoint: TypeAlias = Union[_ESLatLon, Sequence[float], str]
""" Geo point: ``{'lat': .., 'lon': ..}``, ``[lon, lat]``, ``"lat,lon"``, geohash or WKT string """


#
# Term level queries
#

class ESTerm(TypedDict, total=False):
    """
    Represents the parameters for a term query in Elasticsearch.
    """
    value: Required[ESValue]
    """ The value to search for. """
    boost: _Float
    """ Optional boosting value for the query. """
    case_insensitive: _Boolean
    """ Allows ASCII case insensitive matching """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESTermsLookup(TypedDict, total=False):
    """
    Terms lookup: fetch the terms from a field of an existing document
    """
    index: Required[str]
    """ The index to fetch the terms from """
    id: Required[str]
    """ The id of the document """
    path: Required[str]
    """ The field to fetch the terms from """
    routing: str
    """ The routing of the document """


class ESTermsSet(TypedDict, total=False):
    """
    Represents the parameters for a terms_set query in Elasticsearch.
    """
    terms: Required[SequenceNotStr[ESValue]]
    """ The terms to search for """
    minimum_should_match_field: str
    """ Numeric field containing the number of matching terms required """
    minimum_should_match_script: ESScript
    """ Script containing the number of matching terms required """
    minimum_should_match: _MinimumShouldMatch
    """ The number of matching terms required """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESRange(TypedDict, total=False):
    """
    Range query structure
    """
    gt: ESRangeValue
    """ Greater than """
    gte: ESRangeValue
    """ Greater than or equal """
    lt: ESRangeValue
    """ Less than """
    lte: ESRangeValue
    """ Less than or equal """
    format: str
    """ Date format used to convert date values in the query """
    time_zone: str
    """ Time zone used to convert date values in the query (e.g. "+01:00", "Europe/Budapest") """
    relation: Literal['INTERSECTS', 'CONTAINS', 'WITHIN']
    """ How the range query matches values for range fields """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESExists(TypedDict, total=False):
    """
    Represents an exists query to check if a field exists.
    """
    field: Required[str]
    """ The field to check. """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESIds(TypedDict, total=False):
    """
    Represents an ids query: returns documents based on their ids
    """
    values: Required[SequenceNotStr[str]]
    """ The document ids """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESPrefix(TypedDict, total=False):
    """
    Represents a prefix query for prefix matching in Elasticsearch.
    """
    value: Required[str]
    """ The prefix to search for. """
    rewrite: str
    """ Optional, method used to rewrite the query (e.g., "constant_score", "scoring_boolean") """
    case_insensitive: _Boolean
    """ Allows ASCII case insensitive matching """
    boost: _Float
    """ Optional boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESWildcard(TypedDict, total=False):
    """
    Represents a wildcard query for pattern matching in Elasticsearch.
    """
    value: str
    """ The pattern to search for. e.g., "te?t" or "test*" """
    wildcard: str
    """ An alias for value """
    rewrite: str
    """ Optional, method used to rewrite the query (e.g., "constant_score", "scoring_boolean") """
    case_insensitive: _Boolean
    """ Optional, whether the query is case insensitive. """
    boost: _Float
    """ Optional boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESRegexp(TypedDict, total=False):
    """
    Represents a regexp query in Elasticsearch.
    """
    value: Required[str]
    """ The regular expression """
    flags: str
    """ Enables optional operators, e.g. "ALL", "COMPLEMENT|INTERVAL" """
    case_insensitive: _Boolean
    """ Allows ASCII case insensitive matching """
    max_determinized_states: int
    """ Maximum number of automaton states required for the query """
    rewrite: str
    """ Method used to rewrite the query """
    boost: _Float
    """ Optional boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESFuzzy(TypedDict, total=False):
    """
    Represents a fuzzy query for approximate matching in Elasticsearch.
    """
    value: Required[str]
    """ The value to search for. """
    fuzziness: _Fuzziness
    """ Fuzziness value for the query """
    prefix_length: int
    """ Prefix length for the query """
    max_expansions: int
    """ Maximum number of expansions for the query """
    transpositions: _Boolean
    """ Whether to allow transpositions for the query """
    rewrite: str
    """ Method used to rewrite the query """
    boost: _Float
    """ Optional boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


#
# Full text queries
#

class ESMatch(TypedDict, total=False):
    """
    Represents the parameters for a match query in Elasticsearch.
    """
    query: Required[_Text]
    """ The value to search for. """
    operator: _Operator
    """ The operator to use for the query. """
    minimum_should_match: _MinimumShouldMatch
    """ Minimum number of clauses that must match """
    analyzer: str
    """ Optional analyzer to use for the query. """
    fuzziness: _Fuzziness
    """ Optional fuzziness value for the query. """
    prefix_length: int
    """ Optional prefix length for the query. """
    max_expansions: int
    """ Optional maximum number of expansions for the query. """
    fuzzy_transpositions: _Boolean
    """ Whether fuzzy matching allows transpositions """
    fuzzy_rewrite: str
    """ Method used to rewrite the fuzzy query """
    lenient: _Boolean
    """ Ignore format based errors """
    zero_terms_query: _ZeroTermsQuery
    """ Optional zero terms query for the query. """
    auto_generate_synonyms_phrase_query: _Boolean
    """ Create match phrase queries for multi-term synonyms """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESMatchPhrase(TypedDict, total=False):
    """
    Represents the parameters for a match_phrase query in Elasticsearch.
    """
    query: Required[str]
    """ The value to search for. """
    analyzer: str
    """ Optional analyzer to use for the query. """
    slop: int
    """ Optional slop value for the query. """
    zero_terms_query: _ZeroTermsQuery
    """ Optional zero terms query for the query. """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESMatchPhrasePrefix(TypedDict, total=False):
    """
    Represents the parameters for a match_phrase_prefix query in Elasticsearch.
    """
    query: Required[str]
    """ The value to search for. """
    analyzer: str
    """ Optional analyzer to use for the query. """
    max_expansions: int
    """ Maximum number of terms the last term is expanded to """
    slop: int
    """ Optional slop value for the query. """
    zero_terms_query: _ZeroTermsQuery
    """ Optional zero terms query for the query. """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESMatchBoolPrefix(TypedDict, total=False):
    """
    Represents the parameters for a match_bool_prefix query in Elasticsearch.
    """
    query: Required[str]
    """ The value to search for. """
    analyzer: str
    """ Optional analyzer to use for the query. """
    operator: _Operator
    """ The operator to use for the query. """
    minimum_should_match: _MinimumShouldMatch
    """ Minimum number of clauses that must match """
    fuzziness: _Fuzziness
    """ Optional fuzziness value for the query. """
    prefix_length: int
    """ Optional prefix length for the query. """
    max_expansions: int
    """ Optional maximum number of expansions for the query. """
    fuzzy_transpositions: _Boolean
    """ Whether fuzzy matching allows transpositions """
    fuzzy_rewrite: str
    """ Method used to rewrite the fuzzy query """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESMultiMatch(TypedDict, total=False):
    """
    Represents a multi_match query in Elasticsearch.
    """
    query: Required[str]
    """ Query string """
    fields: _Fields
    """ Optional fields list to match on, field names can have boosts (e.g. "title^3") """
    type: Literal[
        "best_fields",
        "most_fields",
        "cross_fields",
        "phrase",
        "phrase_prefix",
        "bool_prefix",
    ]
    """ Type of multi_match query """
    operator: _Operator
    """ The operator to use for the query. """
    minimum_should_match: _MinimumShouldMatch
    """ Minimum number of clauses that must match """
    analyzer: str
    """ Optional analyzer to use for the query. """
    fuzziness: _Fuzziness
    """ Optional fuzziness value for the query. """
    prefix_length: int
    """ Optional prefix length for the query. """
    max_expansions: int
    """ Optional maximum number of expansions for the query. """
    fuzzy_transpositions: _Boolean
    """ Whether fuzzy matching allows transpositions """
    fuzzy_rewrite: str
    """ Method used to rewrite the fuzzy query """
    lenient: _Boolean
    """ Ignore format based errors """
    slop: int
    """ Slop for phrase types """
    tie_breaker: _Float
    """ Tie breaker for best_fields and most_fields types """
    zero_terms_query: _ZeroTermsQuery
    """ Optional zero terms query for the query. """
    auto_generate_synonyms_phrase_query: _Boolean
    """ Create match phrase queries for multi-term synonyms """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESCombinedFields(TypedDict, total=False):
    """
    Represents a combined_fields query in Elasticsearch.
    """
    query: Required[str]
    """ Query string """
    fields: Required[_Fields]
    """ Fields to search in """
    operator: _Operator
    """ The operator to use for the query. """
    minimum_should_match: _MinimumShouldMatch
    """ Minimum number of clauses that must match """
    zero_terms_query: _ZeroTermsQuery
    """ Optional zero terms query for the query. """
    auto_generate_synonyms_phrase_query: _Boolean
    """ Create match phrase queries for multi-term synonyms """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESQueryString(TypedDict, total=False):
    """
    Represents a query_string query in Elasticsearch.
    """
    query: Required[str]
    """ Query string in Lucene query syntax """
    default_field: str
    """ Default field to search in """
    fields: _Fields
    """ Fields to search in """
    default_operator: _Operator
    """ Default boolean operator """
    analyzer: str
    """ Analyzer to use for the query """
    quote_analyzer: str
    """ Analyzer to use for quoted text """
    quote_field_suffix: str
    """ Suffix appended to quoted text """
    analyze_wildcard: _Boolean
    """ Analyze wildcard terms """
    allow_leading_wildcard: _Boolean
    """ Allow wildcard as the first character """
    enable_position_increments: _Boolean
    """ Enable position increments """
    auto_generate_synonyms_phrase_query: _Boolean
    """ Create match phrase queries for multi-term synonyms """
    fuzziness: _Fuzziness
    """ Fuzziness for fuzzy queries """
    fuzzy_max_expansions: int
    """ Maximum number of terms for fuzzy queries """
    fuzzy_prefix_length: int
    """ Prefix length for fuzzy queries """
    fuzzy_transpositions: _Boolean
    """ Whether fuzzy matching allows transpositions """
    lenient: _Boolean
    """ Ignore format based errors """
    max_determinized_states: int
    """ Maximum number of automaton states """
    minimum_should_match: _MinimumShouldMatch
    """ Minimum number of clauses that must match """
    phrase_slop: int
    """ Slop for phrases """
    rewrite: str
    """ Method used to rewrite the query """
    time_zone: str
    """ Time zone for date values """
    type: Literal["best_fields", "most_fields", "cross_fields", "phrase", "phrase_prefix", "bool_prefix"]
    """ How to search multiple fields """
    tie_breaker: _Float
    """ Tie breaker for multiple fields """
    escape: _Boolean
    """ Escape reserved characters """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESSimpleQueryString(TypedDict, total=False):
    """
    Represents a simple_query_string query in Elasticsearch.
    """
    query: Required[str]
    """ Query string in simple query string syntax """
    fields: _Fields
    """ Fields to search in """
    default_operator: _Operator
    """ Default boolean operator """
    analyzer: str
    """ Analyzer to use for the query """
    analyze_wildcard: _Boolean
    """ Analyze wildcard terms """
    auto_generate_synonyms_phrase_query: _Boolean
    """ Create match phrase queries for multi-term synonyms """
    flags: str
    """ Enabled operators, e.g. "OR|AND|PREFIX" """
    fuzzy_max_expansions: int
    """ Maximum number of terms for fuzzy queries """
    fuzzy_prefix_length: int
    """ Prefix length for fuzzy queries """
    fuzzy_transpositions: _Boolean
    """ Whether fuzzy matching allows transpositions """
    lenient: _Boolean
    """ Ignore format based errors """
    minimum_should_match: _MinimumShouldMatch
    """ Minimum number of clauses that must match """
    quote_field_suffix: str
    """ Suffix appended to quoted text """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESMatchAll(TypedDict, total=False):
    """
    Represents a match_all query for matching all documents in Elasticsearch.
    """
    boost: _Float
    """ Optional boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESMatchNone(TypedDict, total=False):
    """
    Represents a match_none query for matching no documents in Elasticsearch.
    """
    _name: str
    """ Name of the query, it is returned in matched_queries """


#
# Geo queries
#

class ESGeoBoundingBox(TypedDict, total=False):
    """
    Bounding box of a geo_bounding_box query
    """
    top_left: ESGeoPoint
    """ Top left corner """
    bottom_right: ESGeoPoint
    """ Bottom right corner """
    top_right: ESGeoPoint
    """ Top right corner """
    bottom_left: ESGeoPoint
    """ Bottom left corner """
    top: _Float
    """ Top latitude """
    left: _Float
    """ Left longitude """
    bottom: _Float
    """ Bottom latitude """
    right: _Float
    """ Right longitude """
    wkt: str
    """ Bounding box as WKT BBOX """


#
# Vector and semantic queries
#

class ESKnnQuery(TypedDict, total=False):
    """
    Represents the parameters for a knn query in Elasticsearch.
    """
    field: Required[str]
    """The field to search on."""
    query_vector: Sequence[float]
    """The query vector."""
    query_vector_builder: Mapping[str, Any]
    """ Configuration to build the query vector from a model """
    k: int
    """The number of neighbors to return."""
    num_candidates: int
    """The number of candidates to consider."""
    filter: Union['ESQuery', Sequence['ESQuery']]
    """Optional filter to apply to the search."""
    similarity: _Float
    """ Minimum similarity of the results """
    rescore_vector: Mapping[str, Any]
    """ Rescoring options of quantized vectors (e.g. ``{'oversample': 2.0}``) """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESSemantic(TypedDict, total=False):
    """
    Represents a semantic query in Elasticsearch.
    """
    field: Required[str]
    """ The semantic_text field to search on """
    query: Required[str]
    """ The query text """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESSparseVector(TypedDict, total=False):
    """
    Represents a sparse_vector query in Elasticsearch.
    """
    field: Required[str]
    """ The sparse_vector field to search on """
    query_vector: Mapping[str, float]
    """ Token - weight pairs """
    inference_id: str
    """ The inference endpoint to convert the query text into tokens """
    query: str
    """ The query text """
    prune: _Boolean
    """ Prune the query """
    pruning_config: Mapping[str, Any]
    """ Pruning configuration """
    boost: _Float
    """ Optional boosting value for the query. """
    _name: str
    """ Name of the query, it is returned in matched_queries """


#
# Compound queries
#

ESClauses: TypeAlias = Union['ESQuery', Sequence['ESQuery']]
""" One query clause or a list of query clauses """


class ESBool(TypedDict, total=False):  # We use total=False, because not every key is required
    """
    Bool query structure
    """
    must: ESClauses
    """ Must queries """
    filter: ESClauses
    """ Filter queries """
    should: ESClauses
    """ Should queries """
    must_not: ESClauses
    """ Must not queries """
    minimum_should_match: _MinimumShouldMatch
    """ Minimum number of should queries to match """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESBoosting(TypedDict, total=False):
    """
    Represents a boosting query in Elasticsearch.
    """
    positive: Required['ESQuery']
    """ Documents must match this query """
    negative: Required['ESQuery']
    """ The score of matching documents is decreased """
    negative_boost: Required[_Float]
    """ Multiplier for the score of the negative matches (between 0 and 1) """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESConstantScore(TypedDict, total=False):
    """
    Represents a constant_score query in Elasticsearch.
    """
    filter: Required['ESQuery']
    """ Filter query """
    boost: _Float
    """ The constant score of the matching documents """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESDisMax(TypedDict, total=False):
    """
    Represents a dis_max query in Elasticsearch.
    """
    queries: Required[Sequence['ESQuery']]
    """ Query clauses """
    tie_breaker: _Float
    """ Increases the score of documents matching multiple clauses """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESFieldValueFactor(TypedDict, total=False):
    """
    Score function based on a field value
    """
    field: Required[str]
    """ The field to use """
    factor: _Float
    """ Multiplier of the field value """
    modifier: Literal['none', 'log', 'log1p', 'log2p', 'ln', 'ln1p', 'ln2p', 'square', 'sqrt', 'reciprocal']
    """ Modifier to apply to the field value """
    missing: _Float
    """ Value to use if the field is missing """


class ESRandomScore(TypedDict, total=False):
    """
    Random score function
    """
    seed: Union[int, str]
    """ Seed of the random numbers """
    field: str
    """ Field to compute the random score from """


class ESScriptScoreFunction(TypedDict):
    """
    Script score function
    """
    script: ESScript
    """ The script computing the score """


class _ESScoreFunctions(TypedDict, total=False):
    """
    Score functions of a function_score query
    """
    weight: _Float
    """ Weight of the function """
    field_value_factor: ESFieldValueFactor
    """ Score based on a field value """
    random_score: ESRandomScore
    """ Random score """
    script_score: ESScriptScoreFunction
    """ Score computed by a script """
    gauss: Mapping[str, Any]
    """ Gauss decay function """
    linear: Mapping[str, Any]
    """ Linear decay function """
    exp: Mapping[str, Any]
    """ Exponential decay function """


class ESScoreFunction(_ESScoreFunctions, total=False):
    """
    A score function in the functions list of a function_score query
    """
    filter: 'ESQuery'
    """ Apply the function only to documents matching this query """


class ESFunctionScore(_ESScoreFunctions, total=False):
    """
    Represents a function_score query in Elasticsearch.

    A single function can be specified directly, more functions in the ``functions`` list.
    """
    query: 'ESQuery'
    """ The query, default is match_all """
    functions: Sequence[ESScoreFunction]
    """ Score functions """
    boost_mode: Literal['multiply', 'replace', 'sum', 'avg', 'max', 'min']
    """ How the computed score is combined with the query score """
    score_mode: Literal['multiply', 'sum', 'avg', 'first', 'max', 'min']
    """ How the scores of the functions are combined """
    max_boost: _Float
    """ Maximum of the computed score """
    min_score: _Float
    """ Documents with lower scores are excluded """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


#
# Joining queries
#

_InnerHits: TypeAlias = Mapping[str, Any]
_ScoreMode: TypeAlias = Literal['avg', 'max', 'min', 'none', 'sum']


class ESNested(TypedDict, total=False):
    """
    Represents a nested query in Elasticsearch.
    """
    path: Required[str]
    """ Path of the nested field """
    query: Required['ESQuery']
    """ Query on the nested documents """
    score_mode: _ScoreMode
    """ How the scores of the matching nested documents are combined """
    ignore_unmapped: _Boolean
    """ Ignore unmapped path instead of returning an error """
    inner_hits: _InnerHits
    """ Return the matching nested documents """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESHasChild(TypedDict, total=False):
    """
    Represents a has_child query in Elasticsearch.
    """
    type: Required[str]
    """ Name of the child relationship """
    query: Required['ESQuery']
    """ Query on the child documents """
    score_mode: _ScoreMode
    """ How the scores of the matching child documents are combined """
    min_children: int
    """ Minimum number of matching children """
    max_children: int
    """ Maximum number of matching children """
    ignore_unmapped: _Boolean
    """ Ignore unmapped type instead of returning an error """
    inner_hits: _InnerHits
    """ Return the matching child documents """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESHasParent(TypedDict, total=False):
    """
    Represents a has_parent query in Elasticsearch.
    """
    parent_type: Required[str]
    """ Name of the parent relationship """
    query: Required['ESQuery']
    """ Query on the parent documents """
    score: _Boolean
    """ Use the score of the parent document """
    ignore_unmapped: _Boolean
    """ Ignore unmapped parent_type instead of returning an error """
    inner_hits: _InnerHits
    """ Return the matching parent documents """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESParentId(TypedDict, total=False):
    """
    Represents a parent_id query in Elasticsearch.
    """
    type: Required[str]
    """ Name of the child relationship """
    id: Required[str]
    """ The id of the parent document """
    ignore_unmapped: _Boolean
    """ Ignore unmapped type instead of returning an error """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


#
# Specialized queries
#

class ESScriptQuery(TypedDict, total=False):
    """
    Represents a script query in Elasticsearch.
    """
    script: Required[ESScript]
    """ The script, it must return a boolean """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESScriptScore(TypedDict, total=False):
    """
    Represents a script_score query in Elasticsearch.
    """
    query: Required['ESQuery']
    """ The query """
    script: Required[ESScript]
    """ The script computing the score """
    min_score: _Float
    """ Documents with lower scores are excluded """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESDistanceFeature(TypedDict, total=False):
    """
    Represents a distance_feature query in Elasticsearch.
    """
    field: Required[str]
    """ Date or geo_point field """
    origin: Required[Union[str, date, ESGeoPoint]]
    """ Origin date or point """
    pivot: Required[str]
    """ Distance from the origin where the score is the half """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESMoreLikeThis(TypedDict, total=False):
    """
    Represents a more_like_this query in Elasticsearch.
    """
    like: Required[Union[str, Mapping[str, Any], Sequence[Union[str, Mapping[str, Any]]]]]
    """ Text or documents to find similar documents to """
    unlike: Union[str, Mapping[str, Any], Sequence[Union[str, Mapping[str, Any]]]]
    """ Text or documents to find dissimilar documents to """
    fields: _Fields
    """ Fields to use """
    min_term_freq: int
    """ Minimum term frequency """
    max_query_terms: int
    """ Maximum number of query terms """
    min_doc_freq: int
    """ Minimum document frequency """
    max_doc_freq: int
    """ Maximum document frequency """
    min_word_length: int
    """ Minimum word length """
    max_word_length: int
    """ Maximum word length """
    stop_words: Sequence[str]
    """ Stop words """
    analyzer: str
    """ Analyzer of the like text """
    minimum_should_match: _MinimumShouldMatch
    """ Minimum number of terms that must match """
    fail_on_unsupported_field: _Boolean
    """ Fail on unsupported field types """
    boost_terms: _Float
    """ Boost terms by their tf-idf score """
    include: _Boolean
    """ Include the input documents in the results """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESRankFeature(TypedDict, total=False):
    """
    Represents a rank_feature query in Elasticsearch.
    """
    field: Required[str]
    """ rank_feature or rank_features field """
    saturation: Mapping[str, float]
    """ Saturation function """
    log: Mapping[str, float]
    """ Logarithmic function """
    sigmoid: Mapping[str, float]
    """ Sigmoid function """
    linear: Mapping[str, float]
    """ Linear function """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESPercolate(TypedDict, total=False):
    """
    Represents a percolate query in Elasticsearch.
    """
    field: Required[str]
    """ The percolator field """
    document: Mapping[str, Any]
    """ The document to match """
    documents: Sequence[Mapping[str, Any]]
    """ The documents to match """
    index: str
    """ Index of a stored document to match """
    id: str
    """ Id of a stored document to match """
    routing: str
    """ Routing of the stored document """
    preference: str
    """ Preference of the stored document """
    version: int
    """ Version of the stored document """
    name: str
    """ Name of the query, used in the response with more percolate queries """
    boost: _Float
    """ Boosting value for the query """
    _name: str
    """ Name of the query, it is returned in matched_queries """


class ESWrapper(TypedDict):
    """
    Represents a wrapper query in Elasticsearch.
    """
    query: str
    """ Base64 encoded query """


class ESPinned(TypedDict, total=False):
    """
    Represents a pinned query in Elasticsearch.
    """
    organic: Required['ESQuery']
    """ The query of the not pinned documents """
    ids: Sequence[str]
    """ Ids of the pinned documents """
    docs: Sequence[Mapping[str, str]]
    """ The pinned documents (``_id`` and ``_index``) """


class ESHybrid(TypedDict, total=False):
    """
    Represents a hybrid query in OpenSearch.
    """
    queries: Required[Sequence['ESQuery']]
    """ Query clauses, their scores are normalized by a search pipeline """
    filter: 'ESQuery'
    """ Filter applied to all subqueries """


#
# The query
#

_GeoParam: TypeAlias = Union[ESGeoPoint, ESGeoBoundingBox, str, float, bool]


class ESQuery(TypedDict, total=False):
    """
    Elasticsearch query structure

    It is used for the whole query and also for the clauses of compound queries. Only one key should be used in one
    query clause, combine queries with ``bool``.
    """
    # Compound queries
    bool: ESBool
    """ Bool query structure """
    boosting: ESBoosting
    """ Boosting query structure """
    constant_score: ESConstantScore
    """ Constant score query structure """
    dis_max: ESDisMax
    """ Disjunction max query structure """
    function_score: ESFunctionScore
    """ Function score query structure """

    # Full text queries
    match: Mapping[str, Union[ESMatch, _Text]]
    """ Match query structure: ``{field: {'query': value, ...}}`` or ``{field: value}`` """
    match_phrase: Mapping[str, Union[ESMatchPhrase, str]]
    """ Match phrase query structure: ``{field: {'query': value, ...}}`` or ``{field: value}`` """
    match_phrase_prefix: Mapping[str, Union[ESMatchPhrasePrefix, str]]
    """ Match phrase prefix query structure: ``{field: {'query': value, ...}}`` or ``{field: value}`` """
    match_bool_prefix: Mapping[str, Union[ESMatchBoolPrefix, str]]
    """ Match bool prefix query structure: ``{field: {'query': value, ...}}`` or ``{field: value}`` """
    multi_match: ESMultiMatch
    """ MultiMatch query structure """
    combined_fields: ESCombinedFields
    """ Combined fields query structure """
    query_string: ESQueryString
    """ Query string query structure """
    simple_query_string: ESSimpleQueryString
    """ Simple query string query structure """
    intervals: Mapping[str, Mapping[str, Any]]
    """ Intervals query structure """

    # Term level queries
    term: Mapping[str, Union[ESTerm, ESValue]]
    """ Term query structure: ``{field: {'value': value, ...}}`` or ``{field: value}`` """
    terms: Mapping[str, Union[SequenceNotStr[ESValue], ESTermsLookup]]
    """ Terms query structure: ``{field: [value, ...]}`` or ``{field: <terms lookup>}`` """
    terms_set: Mapping[str, ESTermsSet]
    """ Terms set query structure """
    range: Mapping[str, ESRange]
    """ Range query structure: ``{field: {'gte': value, ...}}`` """
    exists: ESExists
    """ Exists query structure """
    ids: ESIds
    """ Ids query structure """
    prefix: Mapping[str, Union[ESPrefix, str]]
    """ Prefix query structure: ``{field: {'value': value, ...}}`` or ``{field: value}`` """
    wildcard: Mapping[str, Union[ESWildcard, str]]
    """ Wildcard query structure: ``{field: {'value': value, ...}}`` or ``{field: value}`` """
    regexp: Mapping[str, Union[ESRegexp, str]]
    """ Regexp query structure: ``{field: {'value': value, ...}}`` or ``{field: value}`` """
    fuzzy: Mapping[str, Union[ESFuzzy, str]]
    """ Fuzzy query structure: ``{field: {'value': value, ...}}`` or ``{field: value}`` """

    # Match all / none
    match_all: ESMatchAll
    """ Match all query structure """
    match_none: ESMatchNone
    """ Match none query structure """

    # Geo queries, field name and parameters are in the same dict
    geo_distance: Mapping[str, _GeoParam]
    """ Geo distance query structure: ``{'distance': '10km', field: <geo point>}`` """
    geo_bounding_box: Mapping[str, _GeoParam]
    """ Geo bounding box query structure: ``{field: {'top_left': .., 'bottom_right': ..}}`` """
    geo_shape: Mapping[str, Any]
    """ Geo shape query structure """
    geo_polygon: Mapping[str, Any]
    """ Geo polygon query structure (deprecated) """
    geo_grid: Mapping[str, Any]
    """ Geo grid query structure """
    shape: Mapping[str, Any]
    """ Shape query structure """

    # Joining queries
    nested: ESNested
    """ Nested query structure """
    has_child: ESHasChild
    """ Has child query structure """
    has_parent: ESHasParent
    """ Has parent query structure """
    parent_id: ESParentId
    """ Parent id query structure """

    # Vector and semantic queries
    knn: ESKnnQuery
    """ KNN query structure """
    semantic: ESSemantic
    """ Semantic query structure """
    sparse_vector: ESSparseVector
    """ Sparse vector query structure """
    text_expansion: Mapping[str, Any]
    """ Text expansion query structure (deprecated, use sparse_vector) """
    weighted_tokens: Mapping[str, Any]
    """ Weighted tokens query structure """

    # Specialized queries
    script: ESScriptQuery
    """ Script query structure """
    script_score: ESScriptScore
    """ Script score query structure """
    distance_feature: ESDistanceFeature
    """ Distance feature query structure """
    more_like_this: ESMoreLikeThis
    """ More like this query structure """
    rank_feature: ESRankFeature
    """ Rank feature query structure """
    percolate: ESPercolate
    """ Percolate query structure """
    wrapper: ESWrapper
    """ Wrapper query structure """
    pinned: ESPinned
    """ Pinned query structure """
    rule: Mapping[str, Any]
    """ Rule query structure """

    # Span queries
    span_containing: Mapping[str, Any]
    """ Span containing query structure """
    span_field_masking: Mapping[str, Any]
    """ Span field masking query structure """
    field_masking_span: Mapping[str, Any]
    """ Span field masking query structure (old name) """
    span_first: Mapping[str, Any]
    """ Span first query structure """
    span_multi: Mapping[str, Any]
    """ Span multi-term query structure """
    span_near: Mapping[str, Any]
    """ Span near query structure """
    span_not: Mapping[str, Any]
    """ Span not query structure """
    span_or: Mapping[str, Any]
    """ Span or query structure """
    span_term: Mapping[str, Any]
    """ Span term query structure """
    span_within: Mapping[str, Any]
    """ Span within query structure """

    # OpenSearch specific queries
    hybrid: ESHybrid
    """ Hybrid query structure (OpenSearch) """
    neural: Mapping[str, Any]
    """ Neural query structure (OpenSearch) """
    neural_sparse: Mapping[str, Any]
    """ Neural sparse query structure (OpenSearch) """


#
# Backward compatible names
#
# Earlier versions had a separate type for every query clause in bool queries. Every clause is an ESQuery now.
#

ESMust: TypeAlias = ESClauses
""" Represents must queries in Elasticsearch """
ESFilter: TypeAlias = ESClauses
""" Represents filter queries in Elasticsearch """
ESShould: TypeAlias = ESClauses
""" Represents should queries in Elasticsearch """
ESMustNot: TypeAlias = ESClauses
""" Represents must_not queries in Elasticsearch """

ESGeoDistance: TypeAlias = Mapping[str, _GeoParam]
""" Represents the parameters of a geo_distance query """

FieldRange: TypeAlias = ESQuery
""" A range query clause (use ESQuery) """
FieldTerm: TypeAlias = ESQuery
""" A term query clause (use ESQuery) """
FieldTerms: TypeAlias = ESQuery
""" A terms query clause (use ESQuery) """
FieldMatch: TypeAlias = ESQuery
""" A match query clause (use ESQuery) """
FieldMatchPhrase: TypeAlias = ESQuery
""" A match_phrase query clause (use ESQuery) """
FieldExists: TypeAlias = ESQuery
""" An exists query clause (use ESQuery) """
FieldWildcard: TypeAlias = ESQuery
""" A wildcard query clause (use ESQuery) """
FieldPrefix: TypeAlias = ESQuery
""" A prefix query clause (use ESQuery) """
FieldFuzzy: TypeAlias = ESQuery
""" A fuzzy query clause (use ESQuery) """
FieldGeoDistance: TypeAlias = ESQuery
""" A geo_distance query clause (use ESQuery) """
FieldMatchAll: TypeAlias = ESQuery
""" A match_all query clause (use ESQuery) """
FieldESMatchNone: TypeAlias = ESQuery
""" A match_none query clause (use ESQuery) """
FieldKnn: TypeAlias = ESQuery
""" A knn query clause (use ESQuery) """
FieldBool: TypeAlias = ESQuery
""" A bool query clause (use ESQuery) """
