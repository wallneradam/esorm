"""
Query builder: build ElasticSearch queries from classes instead of nested dicts

The builder produces exactly the same query dicts (:class:`esorm.query.ESQuery`) as you would write by hand, it
doesn't run anything. Compound queries can be built in two ways, which can be mixed:

- with parameters::

    q = Bool(must=[Match('name', 'John'), Range('age', gte=18)], filter=[Term('active', True)])

- with ``with`` blocks, every query clause created inside the block is added to the block::

    with Bool() as q:
        with q.must:
            Match('name', 'John')
            Range('age', gte=18)
        with q.filter:
            Term('active', True)

The search methods of the models accept builder queries, and :meth:`Query.to_query` returns the query dict::

    users = await User.search(q)
    query = q.to_query()

See the Query Builder chapter of the documentation for details.
"""
from typing import (Any, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple, Type, Union, cast,
                    overload)
from typing_extensions import Self, TypeAlias

import asyncio
import threading
from contextvars import ContextVar, Token
from types import TracebackType

from .query import (ESQuery, ESValue, ESRangeValue, ESGeoPoint, ESScript, ESTermsLookup, ESScoreFunction,
                    ESFieldValueFactor, ESRandomScore, ESScriptScoreFunction, SequenceNotStr,
                    _Text, _Fuzziness, _MinimumShouldMatch, _Operator, _ZeroTermsQuery, _ScoreMode, _InnerHits)

__all__ = [
    'Query', 'QueryLike', 'Clauses', 'Clause',
    # Compound queries
    'Bool', 'Boosting', 'ConstantScore', 'DisMax', 'FunctionScore',
    # Term level queries
    'Term', 'Terms', 'TermsSet', 'Range', 'Exists', 'Ids', 'Prefix', 'Wildcard', 'Regexp', 'Fuzzy',
    # Full text queries
    'Match', 'MatchPhrase', 'MatchPhrasePrefix', 'MatchBoolPrefix', 'MultiMatch', 'QueryString',
    'SimpleQueryString',
    'MatchAll', 'MatchNone',
    # Joining queries
    'Nested', 'HasChild', 'HasParent', 'ParentId',
    # Geo queries
    'GeoDistance', 'GeoBoundingBox',
    # Vector queries
    'Knn',
    # Any other query
    'Raw',
]

QueryLike: TypeAlias = Union['Query', ESQuery]
""" A query clause: a builder query or a query dict """

# PyCharm doesn't check a dict literal against a union of a TypedDict and something else (e.g. Optional), it reports
# even the correct dicts. So the methods which often get dict literals have separate overloads for the TypedDicts, and
# the optional TypedDict parameters have a TypedDict type with this default.
_UNSET: Any = None

_MultiMatchType: TypeAlias = Literal['best_fields', 'most_fields', 'cross_fields', 'phrase', 'phrase_prefix',
                                     'bool_prefix']
_ValidationMethod: TypeAlias = Literal['STRICT', 'IGNORE_MALFORMED', 'COERCE']


#
# Collecting the queries created in with blocks
#

class _Collector:
    """
    Base class of the objects which collect the query clauses created inside their ``with`` block
    """

    def __init__(self) -> None:
        self._blocks: List[Tuple['_Block', Token[Optional['_Block']]]] = []

    def _collect(self, query: 'Query') -> None:
        """ Add a query clause which was created inside the with block """
        raise NotImplementedError

    def _discard(self, query: 'Query') -> None:
        """ Remove a collected query clause, because it was given to another query explicitly """
        raise NotImplementedError

    def _collected(self) -> List['Query']:
        """ The collected query clauses """
        raise NotImplementedError

    def _push(self) -> None:
        """ Start collecting """
        block = _Block(self)
        self._blocks.append((block, _current.set(block)))

    def _pop(self) -> None:
        """ Stop collecting, the collected clauses can't be moved anymore """
        block, token = self._blocks.pop()
        block.active = False
        _current.reset(token)
        for query in self._collected():
            if query._collected_by is self:
                query._collected_by = None


def _owner() -> Tuple[threading.Thread, Optional['asyncio.Task[Any]']]:
    """ The thread and the async task the code runs in """
    try:
        task = asyncio.current_task()
    except RuntimeError:  # No running event loop
        task = None
    return threading.current_thread(), task


class _Block:
    """
    An entered with block of a collector

    A new async task (or a thread started with a copied context) inherits the context variables, so it would see the
    block of its parent. So the block remembers where it was entered, and collects only the queries created there,
    and only while it is active.
    """

    def __init__(self, collector: _Collector) -> None:
        self.collector = collector
        self.owner = _owner()
        self.active = True

    def collector_for_owner(self) -> Optional[_Collector]:
        """ The collector, if the block is active and the code runs where it was entered """
        if not self.active:
            return None
        thread, task = _owner()
        if thread is not self.owner[0] or task is not self.owner[1]:
            return None
        return self.collector


# The innermost with block. It is a context variable, so threads and async tasks have their own.
_current: ContextVar[Optional[_Block]] = ContextVar('esorm_builder_block', default=None)


def _adopt(item: QueryLike) -> QueryLike:
    """
    A query clause is given to a query explicitly, so it is removed from the with block it was collected by
    """
    if isinstance(item, Query):
        collector = item._collected_by
        if collector is not None:
            item._collected_by = None
            collector._discard(item)
    elif not isinstance(item, dict):
        raise TypeError(f"A query or a query dict is expected, got {type(item).__name__}")
    return item


def _remove_last(items: List['QueryLike'], query: 'Query') -> None:
    """ Remove the last occurrence of the query (by identity) """
    for i in range(len(items) - 1, -1, -1):
        if items[i] is query:
            del items[i]
            return


def _render(item: QueryLike) -> ESQuery:
    """ Query dict of a query clause """
    return item.to_query() if isinstance(item, Query) else item


def _params(**params: Any) -> Dict[str, Any]:
    """ The parameters which are set (not None), ``name`` is ``_name`` in ES """
    return {('_name' if k == 'name' else k): v for k, v in params.items() if v is not None}


def _list(value: Any) -> Any:
    """ Sequences (e.g. tuples) are converted to lists, so the query dict is the same as a JSON """
    return list(cast(Iterable[Any], value)) if isinstance(value, (list, tuple)) else value


#
# Base classes
#

class Query:
    """
    Base class of the query clauses

    If a query clause is created inside a ``with`` block (e.g. ``with q.must:``), it is added to the block. If it is
    given to another query explicitly (as a parameter, by an operator or by ``append()``) inside the block, it is
    moved there.

    Query clauses can be combined with operators:

    - ``a & b``: both must match (``bool.must``)
    - ``a | b``: at least one of them must match (``bool.should``)
    - ``~a``: must not match (``bool.must_not``)

    Subclasses must call ``super().__init__()`` at the end of their ``__init__()``, when the query is complete,
    and implement :meth:`to_query`.
    """

    def __init__(self) -> None:
        self._collected_by: Optional[_Collector] = None
        block = _current.get()
        collector = block.collector_for_owner() if block is not None else None
        if collector is not None:
            collector._collect(self)
            self._collected_by = collector

    def to_query(self) -> ESQuery:
        """
        Create the query dict

        :return: The query dict, which can be given to the search methods of the models
        """
        raise NotImplementedError

    def __and__(self, other: QueryLike) -> 'Bool':
        return Bool(must=[*_must_parts(self), *_must_parts(other)])

    def __rand__(self, other: QueryLike) -> 'Bool':
        return Bool(must=[*_must_parts(other), *_must_parts(self)])

    def __or__(self, other: QueryLike) -> 'Bool':
        return Bool(should=[*_should_parts(self), *_should_parts(other)], minimum_should_match=1)

    def __ror__(self, other: QueryLike) -> 'Bool':
        return Bool(should=[*_should_parts(other), *_should_parts(self)], minimum_should_match=1)

    def __invert__(self) -> 'Bool':
        return Bool(must_not=[self])

    def __repr__(self) -> str:
        try:
            return f"<{type(self).__name__} {self.to_query()!r}>"
        except ValueError as e:  # The query is not complete yet
            return f"<{type(self).__name__} ({e})>"


class _Compound(Query):
    """
    Base class of the compound queries, they can be used in a ``with`` block

    The query clauses created directly in the block are added to the default place of the query (e.g. ``must`` of a
    bool query).
    """

    def __init__(self) -> None:
        self._entered: List[_Collector] = []
        super().__init__()

    def _default(self) -> _Collector:
        """ The default place of the query clauses created in the with block """
        raise NotImplementedError

    def __enter__(self) -> Self:
        collector = self._default()
        collector._push()
        self._entered.append(collector)
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException],
                 tb: Optional[TracebackType]) -> None:
        self._entered.pop()._pop()


class Clauses(_Collector):
    """
    A list of query clauses, e.g. the ``must`` clauses of a bool query

    It can be used as a list (``append()``, ``extend()``, ``+=``, ``len()``, iteration, indexing), and as a ``with``
    block: the query clauses created inside the block are added to it::

        with q.must:
            Term('active', True)
            Range('age', gte=18)

    :param clauses: The initial query clauses
    """

    def __init__(self, clauses: Optional[Iterable[QueryLike]] = None) -> None:
        super().__init__()
        self._items: List[QueryLike] = []
        if clauses is not None:
            self.extend(clauses)

    @overload
    def append(self, clause: ESQuery) -> None: ...

    @overload
    def append(self, clause: 'Query') -> None: ...

    def append(self, clause: QueryLike) -> None:
        """
        Add a query clause

        :param clause: A query or a query dict
        """
        self._items.append(_adopt(clause))

    def extend(self, clauses: Iterable[QueryLike]) -> None:
        """
        Add more query clauses

        :param clauses: Queries or query dicts
        """
        # A copy, because moving a clause changes the clauses it is moved from (it can be this object or the clauses
        # of an active with block)
        for clause in list(clauses):
            self.append(clause)

    def clear(self) -> None:
        """ Remove all query clauses """
        self._items.clear()

    def __iadd__(self, other: Union[QueryLike, Iterable[QueryLike]]) -> Self:
        if isinstance(other, (Query, dict)):
            self.append(cast(QueryLike, other))
        else:
            self.extend(cast(Iterable[QueryLike], other))
        return self

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __iter__(self) -> Iterator[QueryLike]:
        return iter(self._items)

    @overload
    def __getitem__(self, index: int) -> QueryLike: ...

    @overload
    def __getitem__(self, index: slice) -> List[QueryLike]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[QueryLike, List[QueryLike]]:
        return self._items[index]

    def __enter__(self) -> Self:
        self._push()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException],
                 tb: Optional[TracebackType]) -> None:
        self._pop()

    def to_list(self) -> List[ESQuery]:
        """
        Create the list of query dicts

        :return: The query dicts of the clauses
        """
        return [_render(item) for item in self._items]

    def _collect(self, query: Query) -> None:
        self._items.append(query)

    def _discard(self, query: Query) -> None:
        _remove_last(self._items, query)

    def _collected(self) -> List[Query]:
        return [item for item in self._items if isinstance(item, Query)]

    def __repr__(self) -> str:
        return f"<Clauses {self._items!r}>"


class Clause(_Collector):
    """
    The place of a single query clause, e.g. the query of a nested query

    The query can be set by :meth:`set`, or in a ``with`` block. If more query clauses are created in the block, all
    of them must match (they are combined into ``bool.must``)::

        with q.positive:
            Match('text', 'apple')

    :param query: The initial query clause
    """

    def __init__(self, query: Optional[QueryLike] = None) -> None:
        super().__init__()
        self._items: List[QueryLike] = []
        if query is not None:
            self.set(query)

    @overload
    def set(self, query: ESQuery) -> None: ...

    @overload
    def set(self, query: 'Query') -> None: ...

    def set(self, query: QueryLike) -> None:
        """
        Set (replace) the query clause

        :param query: A query or a query dict
        """
        self._items = [_adopt(query)]

    def __bool__(self) -> bool:
        return bool(self._items)

    def __enter__(self) -> Self:
        self._push()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException],
                 tb: Optional[TracebackType]) -> None:
        self._pop()

    def render(self, owner: str) -> ESQuery:
        """
        Create the query dict

        :param owner: The name of the query which has this clause, for the error message
        :return: The query dict
        :raises ValueError: If the query clause is not set
        """
        if not self._items:
            raise ValueError(f"the query clause of {owner} is not set")
        if len(self._items) == 1:
            return _render(self._items[0])
        return cast(ESQuery, {'bool': {'must': [_render(item) for item in self._items]}})

    def _collect(self, query: Query) -> None:
        self._items.append(query)

    def _discard(self, query: Query) -> None:
        _remove_last(self._items, query)

    def _collected(self) -> List[Query]:
        return [item for item in self._items if isinstance(item, Query)]

    def __repr__(self) -> str:
        return f"<Clause {self._items!r}>"


ClausesInput: TypeAlias = Union[Query, Iterable[QueryLike]]
""" Query clauses as a parameter: a list of queries or query dicts, or a single builder query """


def _clauses(value: Optional[ClausesInput]) -> Clauses:
    """ Clauses from a parameter (a new list, a Clauses object is copied) """
    if value is None:
        return Clauses()
    if isinstance(value, (Query, dict)):
        return Clauses([cast(QueryLike, value)])
    return Clauses(value)


#
# Compound queries
#

class Bool(_Compound):
    """
    Bool query: combine queries, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-bool-query>`__

    The clauses can be given as parameters, or added in ``with`` blocks. The query clauses created directly in the
    ``with Bool()`` block are added to ``must``::

        with Bool() as q:
            Match('name', 'John')           # must
            with q.filter:
                Term('active', True)
            with q.must_not:
                Exists('deleted_at')

    Result: ``{'bool': {'must': [{'match': {'name': 'John'}}], 'filter': [{'term': {'active': True}}],
    'must_not': [{'exists': {'field': 'deleted_at'}}]}}``. The empty clause lists are left out.

    :param must: The clauses which must match, they contribute to the score
    :param filter: The clauses which must match, in filter context (no scoring, cached)
    :param should: The clauses which should match
    :param must_not: The clauses which must not match
    :param minimum_should_match: The number or percentage of the should clauses which must match
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self,
                 must: Optional[ClausesInput] = None,
                 filter: Optional[ClausesInput] = None,
                 should: Optional[ClausesInput] = None,
                 must_not: Optional[ClausesInput] = None,
                 *,
                 minimum_should_match: Optional[_MinimumShouldMatch] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self._must = _clauses(must)
        self._filter = _clauses(filter)
        self._should = _clauses(should)
        self._must_not = _clauses(must_not)
        self.minimum_should_match = minimum_should_match
        """ The number or percentage of the should clauses which must match """
        self.boost = boost
        """ Boosting value of the query """
        self.name = name
        """ Name of the query, it is returned in ``matched_queries`` """
        super().__init__()

    @property
    def must(self) -> Clauses:
        """ The clauses which must match, they contribute to the score """
        return self._must

    @must.setter
    def must(self, value: Clauses) -> None:
        if value is not self._must:  # e.g. q.must += ...
            self._must = _clauses(value)

    @property
    def filter(self) -> Clauses:
        """ The clauses which must match, in filter context (no scoring, cached) """
        return self._filter

    @filter.setter
    def filter(self, value: Clauses) -> None:
        if value is not self._filter:  # e.g. q.filter += ...
            self._filter = _clauses(value)

    @property
    def should(self) -> Clauses:
        """ The clauses which should match """
        return self._should

    @should.setter
    def should(self, value: Clauses) -> None:
        if value is not self._should:  # e.g. q.should += ...
            self._should = _clauses(value)

    @property
    def must_not(self) -> Clauses:
        """ The clauses which must not match """
        return self._must_not

    @must_not.setter
    def must_not(self, value: Clauses) -> None:
        if value is not self._must_not:  # e.g. q.must_not += ...
            self._must_not = _clauses(value)

    def _default(self) -> _Collector:
        return self._must

    def _only(self, key: str) -> bool:
        """ It has only the given clauses, and no parameters (except minimum_should_match 1 of should) """
        if self.boost is not None or self.name is not None:
            return False
        if self.minimum_should_match not in ((None, 1) if key == 'should' else (None,)):
            return False
        return all(not clauses for k, clauses in (('must', self._must), ('filter', self._filter),
                                                  ('should', self._should), ('must_not', self._must_not))
                   if k != key)

    def to_query(self) -> ESQuery:
        body: Dict[str, Any] = {}
        for key, clauses in (('must', self._must), ('filter', self._filter), ('should', self._should),
                             ('must_not', self._must_not)):
            if clauses:
                body[key] = clauses.to_list()
        body.update(_params(minimum_should_match=self.minimum_should_match, boost=self.boost, name=self.name))
        return cast(ESQuery, {'bool': body})


def _must_parts(item: QueryLike) -> List[QueryLike]:
    """ The parts of a query for &: the clauses of a bool query which has only must clauses """
    _adopt(item)
    if isinstance(item, Bool) and item._only('must'):
        return list(item.must)
    return [item]


def _should_parts(item: QueryLike) -> List[QueryLike]:
    """ The parts of a query for |: the clauses of a bool query which has only should clauses """
    _adopt(item)
    # An empty bool query matches every document, so it can't be left out
    if isinstance(item, Bool) and item.should and item._only('should'):
        return list(item.should)
    return [item]


class Boosting(_Compound):
    """
    Boosting query: decrease the score of the documents matching the negative query, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-boosting-query>`__

    ::

        with Boosting(negative_boost=0.5) as q:
            with q.positive:
                Match('text', 'apple')
            with q.negative:
                Match('text', 'pie')

    The query clauses created directly in the ``with Boosting()`` block are added to ``positive``.

    :param positive: Documents must match this query
    :param negative: The score of the matching documents is decreased
    :param negative_boost: Multiplier of the score of the negative matches (between 0 and 1)
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self,
                 positive: Optional[QueryLike] = None,
                 negative: Optional[QueryLike] = None,
                 *,
                 negative_boost: float,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.positive = Clause(positive)
        """ Documents must match this query """
        self.negative = Clause(negative)
        """ The score of the matching documents is decreased """
        self.params = _params(negative_boost=negative_boost, boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def _default(self) -> _Collector:
        return self.positive

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'boosting': {'positive': self.positive.render('Boosting.positive'),
                                           'negative': self.negative.render('Boosting.negative'),
                                           **self.params}})


class ConstantScore(_Compound):
    """
    Constant score query: the matching documents have the same score, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-constant-score-query>`__

    ::

        q = ConstantScore(Term('active', True), boost=1.2)

    Result: ``{'constant_score': {'filter': {'term': {'active': True}}, 'boost': 1.2}}``

    The query clauses created in the ``with ConstantScore()`` block are the filter.

    :param filter: The filter query
    :param boost: The score of the matching documents
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, filter: Optional[QueryLike] = None, *,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.filter = Clause(filter)
        """ The filter query """
        self.params = _params(boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def _default(self) -> _Collector:
        return self.filter

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'constant_score': {'filter': self.filter.render('ConstantScore'), **self.params}})


class DisMax(_Compound):
    """
    Disjunction max query: the score is the score of the best matching clause, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-dis-max-query>`__

    ::

        with DisMax(tie_breaker=0.7) as q:
            Match('title', 'quick fox')
            Match('body', 'quick fox')

    The query clauses created in the ``with DisMax()`` block are added to ``queries``.

    :param queries: The query clauses
    :param tie_breaker: Increases the score of documents matching more clauses
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, queries: Optional[ClausesInput] = None, *,
                 tie_breaker: Optional[float] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.queries = _clauses(queries)
        """ The query clauses """
        self.params = _params(tie_breaker=tie_breaker, boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def _default(self) -> _Collector:
        return self.queries

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'dis_max': {'queries': self.queries.to_list(), **self.params}})


class FunctionScore(_Compound):
    """
    Function score query: modify the score of the documents, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-function-score-query>`__

    ::

        q = FunctionScore(Match('title', 'elasticsearch'),
                          field_value_factor={'field': 'likes', 'modifier': 'log1p'},
                          boost_mode='sum')

    The query clauses created in the ``with FunctionScore()`` block are the query. If the query is not set, it is
    left out (ES uses ``match_all``).

    :param query: The query
    :param functions: Score functions, if there are more
    :param weight: Weight function
    :param field_value_factor: Score based on a field value
    :param random_score: Random score
    :param script_score: Score computed by a script
    :param boost_mode: How the computed score is combined with the query score
    :param score_mode: How the scores of the functions are combined
    :param max_boost: Maximum of the computed score
    :param min_score: Documents with lower scores are excluded
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, query: Optional[QueryLike] = None, *,
                 functions: List[ESScoreFunction] = _UNSET,
                 weight: Optional[float] = None,
                 field_value_factor: ESFieldValueFactor = _UNSET,
                 random_score: ESRandomScore = _UNSET,
                 script_score: ESScriptScoreFunction = _UNSET,
                 boost_mode: Optional[Literal['multiply', 'replace', 'sum', 'avg', 'max', 'min']] = None,
                 score_mode: Optional[Literal['multiply', 'sum', 'avg', 'first', 'max', 'min']] = None,
                 max_boost: Optional[float] = None,
                 min_score: Optional[float] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.query = Clause(query)
        """ The query """
        self.params = _params(functions=functions, weight=weight, field_value_factor=field_value_factor,
                              random_score=random_score, script_score=script_score, boost_mode=boost_mode,
                              score_mode=score_mode, max_boost=max_boost, min_score=min_score, boost=boost,
                              name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def _default(self) -> _Collector:
        return self.query

    def to_query(self) -> ESQuery:
        body: Dict[str, Any] = {}
        if self.query:
            body['query'] = self.query.render('FunctionScore')
        body.update(self.params)
        return cast(ESQuery, {'function_score': body})


#
# Queries of a field
#

class _FieldQuery(Query):
    """
    A query of a field with a value: ``{type: {field: value}}``, or ``{type: {field: {value_key: value, ...}}}`` if
    it has parameters
    """
    _type = ''
    _value_key = 'value'

    def __init__(self, field: str, value: Any, params: Dict[str, Any]) -> None:
        self.field = field
        """ The field name """
        self.value = value
        """ The value """
        self.params = params
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        if not self.params:
            return cast(ESQuery, {self._type: {self.field: self.value}})
        return cast(ESQuery, {self._type: {self.field: {self._value_key: self.value, **self.params}}})


class Term(_FieldQuery):
    """
    Term query: the field contains the exact value, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-term-query>`__

    ::

        Term('active', True)              # {'term': {'active': True}}
        Term('tag', 'python', boost=2.0)  # {'term': {'tag': {'value': 'python', 'boost': 2.0}}}

    :param field: The field name
    :param value: The exact value (use keyword fields for strings)
    :param case_insensitive: Allows ASCII case insensitive matching
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """
    _type = 'term'

    def __init__(self, field: str, value: ESValue, *,
                 case_insensitive: Optional[bool] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(field, value, _params(case_insensitive=case_insensitive, boost=boost, name=name))


class Terms(Query):
    """
    Terms query: the field contains any of the exact values, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-terms-query>`__

    ::

        Terms('tag', ['python', 'es'])  # {'terms': {'tag': ['python', 'es']}}

    The values can be a terms lookup too: ``{'index': ..., 'id': ..., 'path': ...}``.

    :param field: The field name
    :param values: The exact values, or a terms lookup
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    @overload
    def __init__(self, field: str, values: SequenceNotStr[ESValue], *,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None: ...

    @overload
    def __init__(self, field: str, values: ESTermsLookup, *,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None: ...

    def __init__(self, field: str, values: Union[SequenceNotStr[ESValue], ESTermsLookup], *,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.field = field
        """ The field name """
        self.values = values
        """ The values, or a terms lookup """
        self.params = _params(boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'terms': {self.field: _list(self.values), **self.params}})


class TermsSet(Query):
    """
    Terms set query: the field contains a minimum number of the exact values, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-terms-set-query>`__

    ::

        TermsSet('skills', ['python', 'es', 'docker'], minimum_should_match_field='required_skills')

    :param field: The field name
    :param terms: The exact values
    :param minimum_should_match_field: Numeric field containing the number of matching terms required
    :param minimum_should_match_script: Script returning the number of matching terms required
    :param minimum_should_match: The number of matching terms required
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, field: str, terms: SequenceNotStr[ESValue], *,
                 minimum_should_match_field: Optional[str] = None,
                 minimum_should_match_script: Optional[ESScript] = None,
                 minimum_should_match: Optional[_MinimumShouldMatch] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.field = field
        """ The field name """
        self.terms = terms
        """ The exact values """
        self.params = _params(minimum_should_match_field=minimum_should_match_field,
                              minimum_should_match_script=minimum_should_match_script,
                              minimum_should_match=minimum_should_match, boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'terms_set': {self.field: {'terms': _list(self.terms), **self.params}}})


class Range(Query):
    """
    Range query: the field value is in the range, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-range-query>`__

    ::

        Range('age', gte=18, lt=65)          # {'range': {'age': {'gte': 18, 'lt': 65}}}
        Range('created_at', gte='now-1d/d')  # date math

    :param field: The field name
    :param gt: Greater than
    :param gte: Greater than or equal
    :param lt: Less than
    :param lte: Less than or equal
    :param format: Date format used to convert the date values in the query
    :param time_zone: Time zone used to convert the date values in the query (e.g. ``"+01:00"``)
    :param relation: How the query matches the values of range fields
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, field: str, *,
                 gt: Optional[ESRangeValue] = None,
                 gte: Optional[ESRangeValue] = None,
                 lt: Optional[ESRangeValue] = None,
                 lte: Optional[ESRangeValue] = None,
                 format: Optional[str] = None,
                 time_zone: Optional[str] = None,
                 relation: Optional[Literal['INTERSECTS', 'CONTAINS', 'WITHIN']] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.field = field
        """ The field name """
        self.params = _params(gt=gt, gte=gte, lt=lt, lte=lte, format=format, time_zone=time_zone,
                              relation=relation, boost=boost, name=name)
        """ The boundaries and the other parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'range': {self.field: dict(self.params)}})


class Exists(Query):
    """
    Exists query: the field has a value, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-exists-query>`__

    ::

        Exists('email')   # {'exists': {'field': 'email'}}
        ~Exists('email')  # the field has no value

    :param field: The field name
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, field: str, *,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.field = field
        """ The field name """
        self.params = _params(boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'exists': {'field': self.field, **self.params}})


class Ids(Query):
    """
    Ids query: documents by ids, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-ids-query>`__

    ::

        Ids(['1', '4', '100'])  # {'ids': {'values': ['1', '4', '100']}}

    :param values: The document ids
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, values: SequenceNotStr[str], *,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.values = values
        """ The document ids """
        self.params = _params(boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'ids': {'values': _list(self.values), **self.params}})


class Prefix(_FieldQuery):
    """
    Prefix query: the field contains a term with the prefix, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-prefix-query>`__

    ::

        Prefix('name', 'jo')  # {'prefix': {'name': 'jo'}}

    :param field: The field name
    :param value: The prefix
    :param rewrite: Method used to rewrite the query
    :param case_insensitive: Allows ASCII case insensitive matching
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """
    _type = 'prefix'

    def __init__(self, field: str, value: str, *,
                 rewrite: Optional[str] = None,
                 case_insensitive: Optional[bool] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(field, value, _params(rewrite=rewrite, case_insensitive=case_insensitive, boost=boost,
                                               name=name))


class Wildcard(_FieldQuery):
    """
    Wildcard query: the field contains a term matching the pattern, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-wildcard-query>`__

    ::

        Wildcard('name', 'jo*n')  # {'wildcard': {'name': 'jo*n'}}

    :param field: The field name
    :param value: The pattern, ``?`` matches one character, ``*`` matches zero or more characters
    :param rewrite: Method used to rewrite the query
    :param case_insensitive: Allows ASCII case insensitive matching
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """
    _type = 'wildcard'

    def __init__(self, field: str, value: str, *,
                 rewrite: Optional[str] = None,
                 case_insensitive: Optional[bool] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(field, value, _params(rewrite=rewrite, case_insensitive=case_insensitive, boost=boost,
                                               name=name))


class Regexp(_FieldQuery):
    """
    Regexp query: the field contains a term matching the regular expression, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-regexp-query>`__

    ::

        Regexp('code', 'ab[0-9]+')  # {'regexp': {'code': 'ab[0-9]+'}}

    :param field: The field name
    :param value: The regular expression
    :param flags: Enables optional operators, e.g. ``"ALL"``, ``"COMPLEMENT|INTERVAL"``
    :param case_insensitive: Allows ASCII case insensitive matching
    :param max_determinized_states: Maximum number of automaton states required for the query
    :param rewrite: Method used to rewrite the query
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """
    _type = 'regexp'

    def __init__(self, field: str, value: str, *,
                 flags: Optional[str] = None,
                 case_insensitive: Optional[bool] = None,
                 max_determinized_states: Optional[int] = None,
                 rewrite: Optional[str] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(field, value, _params(flags=flags, case_insensitive=case_insensitive,
                                               max_determinized_states=max_determinized_states, rewrite=rewrite,
                                               boost=boost, name=name))


class Fuzzy(_FieldQuery):
    """
    Fuzzy query: the field contains a term similar to the value, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-fuzzy-query>`__

    ::

        Fuzzy('name', 'jonh', fuzziness='AUTO')  # {'fuzzy': {'name': {'value': 'jonh', 'fuzziness': 'AUTO'}}}

    :param field: The field name
    :param value: The value
    :param fuzziness: Maximum edit distance, e.g. ``'AUTO'``, ``1``, ``2``
    :param prefix_length: Number of beginning characters left unchanged
    :param max_expansions: Maximum number of variations created
    :param transpositions: Whether the edits include transpositions of two adjacent characters
    :param rewrite: Method used to rewrite the query
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """
    _type = 'fuzzy'

    def __init__(self, field: str, value: str, *,
                 fuzziness: Optional[_Fuzziness] = None,
                 prefix_length: Optional[int] = None,
                 max_expansions: Optional[int] = None,
                 transpositions: Optional[bool] = None,
                 rewrite: Optional[str] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(field, value, _params(fuzziness=fuzziness, prefix_length=prefix_length,
                                               max_expansions=max_expansions, transpositions=transpositions,
                                               rewrite=rewrite, boost=boost, name=name))


#
# Full text queries
#

class Match(_FieldQuery):
    """
    Match query: full text search in a field, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-query>`__

    ::

        Match('name', 'John Doe')                  # {'match': {'name': 'John Doe'}}
        Match('name', 'John Doe', operator='and')  # {'match': {'name': {'query': 'John Doe', 'operator': 'and'}}}

    :param field: The field name
    :param query: The text to search for
    :param operator: ``'or'`` (default): any of the terms, ``'and'``: all of the terms must match
    :param minimum_should_match: Minimum number or percentage of the terms which must match
    :param analyzer: Analyzer of the query text
    :param fuzziness: Maximum edit distance for fuzzy matching, e.g. ``'AUTO'``
    :param prefix_length: Number of beginning characters left unchanged for fuzzy matching
    :param max_expansions: Maximum number of terms the query expands to
    :param fuzzy_transpositions: Whether fuzzy matching allows transpositions
    :param fuzzy_rewrite: Method used to rewrite the fuzzy query
    :param lenient: Ignore format based errors
    :param zero_terms_query: What to return if the analyzer removes all terms
    :param auto_generate_synonyms_phrase_query: Create match phrase queries for multi-term synonyms
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """
    _type = 'match'
    _value_key = 'query'

    def __init__(self, field: str, query: _Text, *,
                 operator: Optional[_Operator] = None,
                 minimum_should_match: Optional[_MinimumShouldMatch] = None,
                 analyzer: Optional[str] = None,
                 fuzziness: Optional[_Fuzziness] = None,
                 prefix_length: Optional[int] = None,
                 max_expansions: Optional[int] = None,
                 fuzzy_transpositions: Optional[bool] = None,
                 fuzzy_rewrite: Optional[str] = None,
                 lenient: Optional[bool] = None,
                 zero_terms_query: Optional[_ZeroTermsQuery] = None,
                 auto_generate_synonyms_phrase_query: Optional[bool] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(field, query, _params(
            operator=operator, minimum_should_match=minimum_should_match, analyzer=analyzer, fuzziness=fuzziness,
            prefix_length=prefix_length, max_expansions=max_expansions, fuzzy_transpositions=fuzzy_transpositions,
            fuzzy_rewrite=fuzzy_rewrite, lenient=lenient, zero_terms_query=zero_terms_query,
            auto_generate_synonyms_phrase_query=auto_generate_synonyms_phrase_query, boost=boost, name=name))


class MatchPhrase(_FieldQuery):
    """
    Match phrase query: the field contains the phrase, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-query-phrase>`__

    ::

        MatchPhrase('text', 'quick brown fox', slop=1)

    :param field: The field name
    :param query: The phrase
    :param analyzer: Analyzer of the query text
    :param slop: Maximum number of positions allowed between the terms
    :param zero_terms_query: What to return if the analyzer removes all terms
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """
    _type = 'match_phrase'
    _value_key = 'query'

    def __init__(self, field: str, query: str, *,
                 analyzer: Optional[str] = None,
                 slop: Optional[int] = None,
                 zero_terms_query: Optional[_ZeroTermsQuery] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(field, query, _params(analyzer=analyzer, slop=slop, zero_terms_query=zero_terms_query,
                                               boost=boost, name=name))


class MatchPhrasePrefix(_FieldQuery):
    """
    Match phrase prefix query: the field contains the phrase, the last term is a prefix, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-query-phrase-prefix>`__

    ::

        MatchPhrasePrefix('title', 'quick brown f')  # search as you type

    :param field: The field name
    :param query: The phrase, its last term is a prefix
    :param analyzer: Analyzer of the query text
    :param max_expansions: Maximum number of terms the last term is expanded to
    :param slop: Maximum number of positions allowed between the terms
    :param zero_terms_query: What to return if the analyzer removes all terms
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """
    _type = 'match_phrase_prefix'
    _value_key = 'query'

    def __init__(self, field: str, query: str, *,
                 analyzer: Optional[str] = None,
                 max_expansions: Optional[int] = None,
                 slop: Optional[int] = None,
                 zero_terms_query: Optional[_ZeroTermsQuery] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(field, query, _params(analyzer=analyzer, max_expansions=max_expansions, slop=slop,
                                               zero_terms_query=zero_terms_query, boost=boost, name=name))


class MatchBoolPrefix(_FieldQuery):
    """
    Match bool prefix query: the terms are term queries, the last one is a prefix query, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-bool-prefix-query>`__

    ::

        MatchBoolPrefix('title', 'quick brown f')

    :param field: The field name
    :param query: The text, its last term is a prefix
    :param analyzer: Analyzer of the query text
    :param operator: ``'or'`` (default): any of the terms, ``'and'``: all of the terms must match
    :param minimum_should_match: Minimum number or percentage of the terms which must match
    :param fuzziness: Maximum edit distance for fuzzy matching, e.g. ``'AUTO'``
    :param prefix_length: Number of beginning characters left unchanged for fuzzy matching
    :param max_expansions: Maximum number of terms the query expands to
    :param fuzzy_transpositions: Whether fuzzy matching allows transpositions
    :param fuzzy_rewrite: Method used to rewrite the fuzzy query
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """
    _type = 'match_bool_prefix'
    _value_key = 'query'

    def __init__(self, field: str, query: str, *,
                 analyzer: Optional[str] = None,
                 operator: Optional[_Operator] = None,
                 minimum_should_match: Optional[_MinimumShouldMatch] = None,
                 fuzziness: Optional[_Fuzziness] = None,
                 prefix_length: Optional[int] = None,
                 max_expansions: Optional[int] = None,
                 fuzzy_transpositions: Optional[bool] = None,
                 fuzzy_rewrite: Optional[str] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(field, query, _params(
            analyzer=analyzer, operator=operator, minimum_should_match=minimum_should_match, fuzziness=fuzziness,
            prefix_length=prefix_length, max_expansions=max_expansions, fuzzy_transpositions=fuzzy_transpositions,
            fuzzy_rewrite=fuzzy_rewrite, boost=boost, name=name))


class MultiMatch(Query):
    """
    Multi match query: full text search in more fields, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-multi-match-query>`__

    ::

        MultiMatch('quick fox', ['title^3', 'body'], type='best_fields')

    :param query: The text to search for
    :param fields: The fields to search in, they can have boosts (e.g. ``'title^3'``), the default is the
                   ``index.query.default_field`` setting
    :param type: How the fields are searched and scored
    :param operator: ``'or'`` (default): any of the terms, ``'and'``: all of the terms must match
    :param minimum_should_match: Minimum number or percentage of the terms which must match
    :param analyzer: Analyzer of the query text
    :param fuzziness: Maximum edit distance for fuzzy matching, e.g. ``'AUTO'``
    :param prefix_length: Number of beginning characters left unchanged for fuzzy matching
    :param max_expansions: Maximum number of terms the query expands to
    :param fuzzy_transpositions: Whether fuzzy matching allows transpositions
    :param fuzzy_rewrite: Method used to rewrite the fuzzy query
    :param lenient: Ignore format based errors
    :param slop: Slop of the phrase types
    :param tie_breaker: Tie breaker of the best_fields and most_fields types
    :param zero_terms_query: What to return if the analyzer removes all terms
    :param auto_generate_synonyms_phrase_query: Create match phrase queries for multi-term synonyms
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, query: str, fields: Optional[SequenceNotStr[str]] = None, *,
                 type: Optional[_MultiMatchType] = None,
                 operator: Optional[_Operator] = None,
                 minimum_should_match: Optional[_MinimumShouldMatch] = None,
                 analyzer: Optional[str] = None,
                 fuzziness: Optional[_Fuzziness] = None,
                 prefix_length: Optional[int] = None,
                 max_expansions: Optional[int] = None,
                 fuzzy_transpositions: Optional[bool] = None,
                 fuzzy_rewrite: Optional[str] = None,
                 lenient: Optional[bool] = None,
                 slop: Optional[int] = None,
                 tie_breaker: Optional[float] = None,
                 zero_terms_query: Optional[_ZeroTermsQuery] = None,
                 auto_generate_synonyms_phrase_query: Optional[bool] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.query = query
        """ The text to search for """
        self.params = _params(
            fields=_list(fields), type=type, operator=operator, minimum_should_match=minimum_should_match,
            analyzer=analyzer, fuzziness=fuzziness, prefix_length=prefix_length, max_expansions=max_expansions,
            fuzzy_transpositions=fuzzy_transpositions, fuzzy_rewrite=fuzzy_rewrite, lenient=lenient, slop=slop,
            tie_breaker=tie_breaker, zero_terms_query=zero_terms_query,
            auto_generate_synonyms_phrase_query=auto_generate_synonyms_phrase_query, boost=boost, name=name)
        """ The fields and the other parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'multi_match': {'query': self.query, **self.params}})


class QueryString(Query):
    """
    Query string query: search with the Lucene query syntax, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-query-string-query>`__

    ::

        QueryString('(new york city) OR (big apple)', default_field='content')

    It returns an error for an invalid syntax, use :class:`SimpleQueryString` for user input.

    :param query: The query string
    :param default_field: The default field to search in
    :param fields: The fields to search in
    :param default_operator: The default boolean operator
    :param analyzer: Analyzer of the query string
    :param quote_analyzer: Analyzer of the quoted text
    :param quote_field_suffix: Suffix appended to the quoted text
    :param analyze_wildcard: Analyze the wildcard terms
    :param allow_leading_wildcard: Allow wildcard as the first character
    :param enable_position_increments: Enable position increments
    :param auto_generate_synonyms_phrase_query: Create match phrase queries for multi-term synonyms
    :param fuzziness: Maximum edit distance of fuzzy queries
    :param fuzzy_max_expansions: Maximum number of terms of fuzzy queries
    :param fuzzy_prefix_length: Prefix length of fuzzy queries
    :param fuzzy_transpositions: Whether fuzzy matching allows transpositions
    :param lenient: Ignore format based errors
    :param max_determinized_states: Maximum number of automaton states
    :param minimum_should_match: Minimum number or percentage of the clauses which must match
    :param phrase_slop: Slop of the phrases
    :param rewrite: Method used to rewrite the query
    :param time_zone: Time zone of the date values
    :param type: How more fields are searched
    :param tie_breaker: Tie breaker of more fields
    :param escape: Escape the reserved characters
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, query: str, *,
                 default_field: Optional[str] = None,
                 fields: Optional[SequenceNotStr[str]] = None,
                 default_operator: Optional[_Operator] = None,
                 analyzer: Optional[str] = None,
                 quote_analyzer: Optional[str] = None,
                 quote_field_suffix: Optional[str] = None,
                 analyze_wildcard: Optional[bool] = None,
                 allow_leading_wildcard: Optional[bool] = None,
                 enable_position_increments: Optional[bool] = None,
                 auto_generate_synonyms_phrase_query: Optional[bool] = None,
                 fuzziness: Optional[_Fuzziness] = None,
                 fuzzy_max_expansions: Optional[int] = None,
                 fuzzy_prefix_length: Optional[int] = None,
                 fuzzy_transpositions: Optional[bool] = None,
                 lenient: Optional[bool] = None,
                 max_determinized_states: Optional[int] = None,
                 minimum_should_match: Optional[_MinimumShouldMatch] = None,
                 phrase_slop: Optional[int] = None,
                 rewrite: Optional[str] = None,
                 time_zone: Optional[str] = None,
                 type: Optional[_MultiMatchType] = None,
                 tie_breaker: Optional[float] = None,
                 escape: Optional[bool] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.query = query
        """ The query string """
        self.params = _params(
            default_field=default_field, fields=_list(fields), default_operator=default_operator, analyzer=analyzer,
            quote_analyzer=quote_analyzer, quote_field_suffix=quote_field_suffix, analyze_wildcard=analyze_wildcard,
            allow_leading_wildcard=allow_leading_wildcard, enable_position_increments=enable_position_increments,
            auto_generate_synonyms_phrase_query=auto_generate_synonyms_phrase_query, fuzziness=fuzziness,
            fuzzy_max_expansions=fuzzy_max_expansions, fuzzy_prefix_length=fuzzy_prefix_length,
            fuzzy_transpositions=fuzzy_transpositions, lenient=lenient,
            max_determinized_states=max_determinized_states, minimum_should_match=minimum_should_match,
            phrase_slop=phrase_slop, rewrite=rewrite, time_zone=time_zone, type=type, tie_breaker=tie_breaker,
            escape=escape, boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'query_string': {'query': self.query, **self.params}})


class SimpleQueryString(Query):
    """
    Simple query string query: search with a simple syntax, it never returns an error, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-simple-query-string-query>`__

    ::

        SimpleQueryString('"fried eggs" +(eggplant | potato) -frittata', fields=['title^5', 'body'])

    It is safe for user input.

    :param query: The query string
    :param fields: The fields to search in
    :param default_operator: The default boolean operator
    :param analyzer: Analyzer of the query string
    :param analyze_wildcard: Analyze the wildcard terms
    :param auto_generate_synonyms_phrase_query: Create match phrase queries for multi-term synonyms
    :param flags: The enabled operators, e.g. ``"OR|AND|PREFIX"``
    :param fuzzy_max_expansions: Maximum number of terms of fuzzy queries
    :param fuzzy_prefix_length: Prefix length of fuzzy queries
    :param fuzzy_transpositions: Whether fuzzy matching allows transpositions
    :param lenient: Ignore format based errors
    :param minimum_should_match: Minimum number or percentage of the clauses which must match
    :param quote_field_suffix: Suffix appended to the quoted text
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, query: str, *,
                 fields: Optional[SequenceNotStr[str]] = None,
                 default_operator: Optional[_Operator] = None,
                 analyzer: Optional[str] = None,
                 analyze_wildcard: Optional[bool] = None,
                 auto_generate_synonyms_phrase_query: Optional[bool] = None,
                 flags: Optional[str] = None,
                 fuzzy_max_expansions: Optional[int] = None,
                 fuzzy_prefix_length: Optional[int] = None,
                 fuzzy_transpositions: Optional[bool] = None,
                 lenient: Optional[bool] = None,
                 minimum_should_match: Optional[_MinimumShouldMatch] = None,
                 quote_field_suffix: Optional[str] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.query = query
        """ The query string """
        self.params = _params(
            fields=_list(fields), default_operator=default_operator, analyzer=analyzer,
            analyze_wildcard=analyze_wildcard,
            auto_generate_synonyms_phrase_query=auto_generate_synonyms_phrase_query, flags=flags,
            fuzzy_max_expansions=fuzzy_max_expansions, fuzzy_prefix_length=fuzzy_prefix_length,
            fuzzy_transpositions=fuzzy_transpositions, lenient=lenient, minimum_should_match=minimum_should_match,
            quote_field_suffix=quote_field_suffix, boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'simple_query_string': {'query': self.query, **self.params}})


class MatchAll(Query):
    """
    Match all query: all documents, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-all-query>`__

    ::

        MatchAll()  # {'match_all': {}}

    :param boost: The score of the documents
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, *, boost: Optional[float] = None, name: Optional[str] = None) -> None:
        self.params = _params(boost=boost, name=name)
        """ The parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'match_all': dict(self.params)})


class MatchNone(Query):
    """
    Match none query: no documents, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-all-query#query-dsl-match-none-query>`__

    ::

        MatchNone()  # {'match_none': {}}

    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, *, name: Optional[str] = None) -> None:
        self.params = _params(name=name)
        """ The parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'match_none': dict(self.params)})


#
# Joining queries
#

class Nested(_Compound):
    """
    Nested query: query the nested documents (``List[SubModel]`` fields), `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-nested-query>`__

    The field names in the query are full paths (e.g. ``comments.author``). A nested document must match all
    clauses in the query, that's the difference from querying an object field::

        with Nested('comments'):
            Term('comments.author', 'john')
            Range('comments.stars', gte=4)

    Result: ``{'nested': {'path': 'comments', 'query': {'bool': {'must': [...]}}}}``

    :param path: The path of the nested field
    :param query: The query of the nested documents
    :param score_mode: How the scores of the matching nested documents are combined
    :param ignore_unmapped: Ignore an unmapped path instead of returning an error
    :param inner_hits: Return the matching nested documents, ``{}`` for the defaults
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, path: str, query: Optional[QueryLike] = None, *,
                 score_mode: Optional[_ScoreMode] = None,
                 ignore_unmapped: Optional[bool] = None,
                 inner_hits: Optional[_InnerHits] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.path = path
        """ The path of the nested field """
        self.query = Clause(query)
        """ The query of the nested documents """
        self.params = _params(score_mode=score_mode, ignore_unmapped=ignore_unmapped, inner_hits=inner_hits,
                              boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def _default(self) -> _Collector:
        return self.query

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'nested': {'path': self.path, 'query': self.query.render('Nested'), **self.params}})


class HasChild(_Compound):
    """
    Has child query: the parent documents of the matching child documents (join field), `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-has-child-query>`__

    ::

        with HasChild('answer', min_children=2):
            Match('body', 'elasticsearch')

    :param type: Name of the child relationship
    :param query: The query of the child documents
    :param score_mode: How the scores of the matching child documents are combined
    :param min_children: Minimum number of matching children
    :param max_children: Maximum number of matching children
    :param ignore_unmapped: Ignore an unmapped type instead of returning an error
    :param inner_hits: Return the matching child documents, ``{}`` for the defaults
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, type: str, query: Optional[QueryLike] = None, *,
                 score_mode: Optional[_ScoreMode] = None,
                 min_children: Optional[int] = None,
                 max_children: Optional[int] = None,
                 ignore_unmapped: Optional[bool] = None,
                 inner_hits: Optional[_InnerHits] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.type = type
        """ Name of the child relationship """
        self.query = Clause(query)
        """ The query of the child documents """
        self.params = _params(score_mode=score_mode, min_children=min_children, max_children=max_children,
                              ignore_unmapped=ignore_unmapped, inner_hits=inner_hits, boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def _default(self) -> _Collector:
        return self.query

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'has_child': {'type': self.type, 'query': self.query.render('HasChild'),
                                            **self.params}})


class HasParent(_Compound):
    """
    Has parent query: the child documents of the matching parent documents (join field), `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-has-parent-query>`__

    ::

        with HasParent('question'):
            Term('tags', 'python')

    :param parent_type: Name of the parent relationship
    :param query: The query of the parent documents
    :param score: Use the score of the parent document
    :param ignore_unmapped: Ignore an unmapped parent type instead of returning an error
    :param inner_hits: Return the matching parent documents, ``{}`` for the defaults
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, parent_type: str, query: Optional[QueryLike] = None, *,
                 score: Optional[bool] = None,
                 ignore_unmapped: Optional[bool] = None,
                 inner_hits: Optional[_InnerHits] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.parent_type = parent_type
        """ Name of the parent relationship """
        self.query = Clause(query)
        """ The query of the parent documents """
        self.params = _params(score=score, ignore_unmapped=ignore_unmapped, inner_hits=inner_hits, boost=boost,
                              name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def _default(self) -> _Collector:
        return self.query

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'has_parent': {'parent_type': self.parent_type,
                                             'query': self.query.render('HasParent'), **self.params}})


class ParentId(Query):
    """
    Parent id query: the child documents of a parent document (join field), `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-parent-id-query>`__

    ::

        ParentId('answer', '1')  # {'parent_id': {'type': 'answer', 'id': '1'}}

    :param type: Name of the child relationship
    :param id: The id of the parent document
    :param ignore_unmapped: Ignore an unmapped type instead of returning an error
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, type: str, id: str, *,
                 ignore_unmapped: Optional[bool] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.type = type
        """ Name of the child relationship """
        self.id = id
        """ The id of the parent document """
        self.params = _params(ignore_unmapped=ignore_unmapped, boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'parent_id': {'type': self.type, 'id': self.id, **self.params}})


#
# Geo queries
#

class GeoDistance(Query):
    """
    Geo distance query: the geo point is within the distance, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-geo-distance-query>`__

    ::

        GeoDistance('location', {'lat': 47.49, 'lon': 19.04}, '10km')

    Result: ``{'geo_distance': {'distance': '10km', 'location': {'lat': 47.49, 'lon': 19.04}}}``

    :param field: The geo_point field name
    :param point: The center point: ``{'lat': .., 'lon': ..}``, ``[lon, lat]``, ``"lat,lon"``, geohash or WKT
    :param distance: The radius, e.g. ``'10km'``, ``'200m'``, a number is in meters
    :param distance_type: How the distance is computed, ``'arc'`` (default) or ``'plane'`` (faster, less accurate)
    :param validation_method: How invalid coordinates are handled
    :param ignore_unmapped: Ignore an unmapped field instead of returning an error
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, field: str, point: ESGeoPoint, distance: Union[str, float], *,
                 distance_type: Optional[Literal['arc', 'plane']] = None,
                 validation_method: Optional[_ValidationMethod] = None,
                 ignore_unmapped: Optional[bool] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.field = field
        """ The geo_point field name """
        self.point = point
        """ The center point """
        self.distance = distance
        """ The radius """
        self.params = _params(distance_type=distance_type, validation_method=validation_method,
                              ignore_unmapped=ignore_unmapped, boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'geo_distance': {'distance': self.distance, self.field: _list(self.point),
                                               **self.params}})


class GeoBoundingBox(Query):
    """
    Geo bounding box query: the geo point is in the box, `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-geo-bounding-box-query>`__

    The box is given by 2 corners, by the 4 sides or as WKT::

        GeoBoundingBox('location', top_left={'lat': 47.6, 'lon': 18.9}, bottom_right={'lat': 47.4, 'lon': 19.2})

    :param field: The geo_point field name
    :param top_left: Top left corner
    :param bottom_right: Bottom right corner
    :param top_right: Top right corner
    :param bottom_left: Bottom left corner
    :param top: Top latitude
    :param left: Left longitude
    :param bottom: Bottom latitude
    :param right: Right longitude
    :param wkt: The box as WKT, e.g. ``'BBOX (18.9, 19.2, 47.6, 47.4)'``
    :param validation_method: How invalid coordinates are handled
    :param ignore_unmapped: Ignore an unmapped field instead of returning an error
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, field: str, *,
                 top_left: Optional[ESGeoPoint] = None,
                 bottom_right: Optional[ESGeoPoint] = None,
                 top_right: Optional[ESGeoPoint] = None,
                 bottom_left: Optional[ESGeoPoint] = None,
                 top: Optional[float] = None,
                 left: Optional[float] = None,
                 bottom: Optional[float] = None,
                 right: Optional[float] = None,
                 wkt: Optional[str] = None,
                 validation_method: Optional[_ValidationMethod] = None,
                 ignore_unmapped: Optional[bool] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.field = field
        """ The geo_point field name """
        self.box = {k: _list(v) for k, v in _params(top_left=top_left, bottom_right=bottom_right,
                                                    top_right=top_right, bottom_left=bottom_left, top=top,
                                                    left=left, bottom=bottom, right=right, wkt=wkt).items()}
        """ The bounding box (as in ES) """
        self.params = _params(validation_method=validation_method, ignore_unmapped=ignore_unmapped, boost=boost,
                              name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def to_query(self) -> ESQuery:
        return cast(ESQuery, {'geo_bounding_box': {self.field: dict(self.box), **self.params}})


#
# Vector queries
#

class Knn(_Compound):
    """
    kNN query: the nearest neighbors of a vector (dense_vector field), `ES docs
    <https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-knn-query>`__

    ::

        with Knn('embedding', [0.1, 0.2, 0.3], k=10, num_candidates=100):
            Term('lang', 'en')  # filter

    The query clauses created in the ``with Knn()`` block are the filters, the documents must match all of them.

    :param field: The dense_vector field name
    :param query_vector: The query vector
    :param query_vector_builder: Configuration to build the query vector from a model
    :param k: The number of neighbors to return from each shard
    :param num_candidates: The number of candidates to consider on each shard
    :param filter: Filter the documents before the vector search
    :param similarity: Minimum similarity of the results
    :param rescore_vector: Rescoring options of quantized vectors (e.g. ``{'oversample': 2.0}``)
    :param boost: Boosting value of the query
    :param name: Name of the query, it is returned in ``matched_queries`` (``_name``)
    """

    def __init__(self, field: str, query_vector: Optional[Sequence[float]] = None, *,
                 query_vector_builder: Optional[Dict[str, Any]] = None,
                 k: Optional[int] = None,
                 num_candidates: Optional[int] = None,
                 filter: Optional[ClausesInput] = None,
                 similarity: Optional[float] = None,
                 rescore_vector: Optional[Dict[str, Any]] = None,
                 boost: Optional[float] = None,
                 name: Optional[str] = None) -> None:
        self.field = field
        """ The dense_vector field name """
        self.filter = _clauses(filter)
        """ Filter the documents before the vector search """
        self.params = _params(query_vector=_list(query_vector), query_vector_builder=query_vector_builder, k=k,
                              num_candidates=num_candidates, similarity=similarity, rescore_vector=rescore_vector,
                              boost=boost, name=name)
        """ The other parameters of the query (as in ES) """
        super().__init__()

    def _default(self) -> _Collector:
        return self.filter

    def to_query(self) -> ESQuery:
        body: Dict[str, Any] = {'field': self.field, **self.params}
        if self.filter:
            filters = self.filter.to_list()
            body['filter'] = filters[0] if len(filters) == 1 else filters
        return cast(ESQuery, {'knn': body})


#
# Any other query
#

class Raw(Query):
    """
    A query dict as a builder query

    Query dicts can be given anywhere as query clauses, but only builder queries are collected by ``with`` blocks.
    With Raw any query can be used in a ``with`` block, or with the operators::

        with Bool() as q:
            Raw({'span_term': {'text': 'fox'}})

    :param query: The query dict, it is in the result as it is (not copied)
    """

    def __init__(self, query: ESQuery) -> None:
        self.query = query
        """ The query dict """
        super().__init__()

    def to_query(self) -> ESQuery:
        return self.query
