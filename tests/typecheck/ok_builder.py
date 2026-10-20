# pyright: standard
"""
Correct usage of the query builder, type checkers must not report anything here
"""
from typing import Dict, List, Optional
from datetime import date, datetime
from uuid import UUID

from esorm import ESModel, Pagination, Sort
from esorm.query import ESQuery
from esorm.aggs import ESAggs
from esorm.builder import (Query, QueryLike, Clauses, Bool, Boosting, ConstantScore, DisMax, FunctionScore,
                           Term, Terms, TermsSet, Range, Exists, Ids, Prefix, Wildcard, Regexp, Fuzzy,
                           Match, MatchPhrase, MatchPhrasePrefix, MatchBoolPrefix, MultiMatch, QueryString,
                           SimpleQueryString, MatchAll, MatchNone, Nested, HasChild, HasParent, ParentId,
                           GeoDistance, GeoBoundingBox, Knn, Raw)


class User(ESModel):
    name: str
    age: int = 18
    active: bool = True


async def parameters(uid: UUID) -> None:
    country_hu: ESQuery = {'term': {'country': 'HU'}}
    q = Bool(
        must=[Match('name', 'John'), Range('age', gte=18, lt=65)],
        filter=[Term('active', True), Raw({'term': {'country': 'HU'}}), country_hu],  # query dicts are clauses too
        must_not=Exists('deleted_at'),  # a single builder query
        minimum_should_match=1,
    )
    query: ESQuery = q.to_query()
    users = await User.search(query)
    for user in users:
        print(user.name)
    await User.search(q.to_query(), page_size=10)
    user1 = await User.search_one(q.to_query())
    if user1 is not None:
        print(user1.name)
    await User.count(q.to_query())
    await User.aggregate({'ages': {'terms': {'field': 'age'}}}, query=q.to_query())
    async for user in User.scan(q.to_query()):
        print(user.name)

    # Builder queries can be given directly too
    users2: List[User] = await User.search(q)
    users3: List[User] = await User.search(q, page_size=10, page=2, sort='age', res_dict=False)
    by_id: Dict[str, User] = await User.search(Term('name', 'John'), res_dict=True)
    print(users2[0].name, users3, by_id['1'].age)
    user2 = await User.search_one(Match('name', 'John'))
    if user2 is not None:
        print(user2.name)
    count: int = await User.count(Term('active', True))
    print(count)
    aggs_def: ESAggs = {'ages': {'terms': {'field': 'age'}}}
    await User.aggregate(aggs_def, query=q)
    aggs = await User.aggregate({'ages': {'terms': {'field': 'age'}}}, query=q)
    for bucket in aggs['ages']['buckets']:
        print(bucket['key'])
    async for user in User.scan(q, batch_size=100):
        print(user.name)
    async for batch in User.scan_batches(Range('age', gte=18)):
        print(len(batch))
    # Query dicts still work as before
    await User.search({'term': {'name': 'John'}})
    await User.count({'term': {'name': 'John'}})
    # Decorated models
    paginated = Pagination(page=1, page_size=10)(User)
    sorted_users = Sort(sort='age')(User)
    await paginated.search(q)
    await sorted_users.search(q)

    # Values
    Term('id', uid)
    Term('created', date(2020, 1, 1))
    Term('tag', 'python', boost=2.0, case_insensitive=True, name='tag_query')
    Range('created', gte=datetime(2020, 1, 1), lte='now/d', time_zone='+01:00', format='yyyy-MM-dd')
    Terms('tag', ['a', 'b'])
    Terms('tag', ('a', 'b'))
    Terms('tag', {'index': 'users', 'id': '1', 'path': 'tags'})
    TermsSet('skills', ['python', 'es'], minimum_should_match_field='required')
    Ids(['1', '2'])
    Prefix('name', 'Jo', case_insensitive=True)
    Wildcard('name', 'J*n')
    Regexp('code', 'ab[0-9]+', flags='ALL')
    Fuzzy('name', 'jonh', fuzziness='AUTO')
    Fuzzy('name', 'jonh', fuzziness=2)
    Match('name', 'John Doe', operator='and', minimum_should_match='75%')
    MatchPhrase('text', 'quick brown fox', slop=1)
    MatchPhrasePrefix('title', 'quick brown f')
    MatchBoolPrefix('title', 'quick brown f')
    MultiMatch('quick fox', ['title^3', 'body'], type='best_fields', tie_breaker=0.3)
    QueryString('(new york) OR (big apple)', default_field='content', default_operator='AND')
    SimpleQueryString('"fried eggs" +potato', fields=['title^5', 'body'])
    MatchAll(boost=1.5)
    MatchNone()
    Nested('comments', Term('comments.author', 'john'), score_mode='avg', inner_hits={})
    HasChild('answer', Match('body', 'es'), min_children=2)
    HasParent('question', Term('tags', 'python'), score=True)
    ParentId('answer', '1')
    GeoDistance('location', {'lat': 47.49, 'lon': 19.04}, '10km')
    GeoDistance('location', [19.04, 47.49], 1000, distance_type='plane')
    GeoBoundingBox('location', top_left={'lat': 47.6, 'lon': 18.9}, bottom_right='47.4,19.2')
    Knn('embedding', [0.1, 0.2], k=10, num_candidates=100, filter=[Term('lang', 'en')])
    Boosting(Match('text', 'apple'), Match('text', 'pie'), negative_boost=0.5)
    ConstantScore(Term('active', True), boost=1.2)
    DisMax([Match('title', 'fox'), Match('body', 'fox')], tie_breaker=0.7)
    FunctionScore(Match('title', 'es'), field_value_factor={'field': 'likes', 'modifier': 'log1p'},
                  boost_mode='sum')
    Raw({'span_term': {'text': 'fox'}})


def with_blocks() -> ESQuery:
    with Bool() as q:
        Term('active', True)
        with q.should:
            Term('tag', 'python')
            with Nested('comments'):
                Match('comments.text', 'great')
            with Bool() as inner:
                with inner.filter:
                    Range('stars', gte=100)
        with q.must_not as must_not:
            Exists('deleted_at')
            print(len(must_not))
        q.minimum_should_match = 1
    with Boosting(negative_boost=0.5) as b:
        with b.positive:
            Match('text', 'apple')
        with b.negative:
            Match('text', 'pie')
    with DisMax() as d:
        Match('title', 'fox')
    with Knn('embedding', [0.1, 0.2], k=10):
        Term('lang', 'en')
    with ConstantScore():
        Term('active', True)
    with FunctionScore(weight=2):
        Term('active', True)
    with HasChild('answer'):
        Term('a', 1)
    with HasParent('question'):
        Term('a', 1)
    print(b.to_query(), d.to_query())
    return q.to_query()


def operators() -> List[ESQuery]:
    a = Term('a', 1) & Term('b', 2)
    o = Term('a', 1) | Term('b', 2) | Raw({'match_all': {}})
    n = ~Exists('deleted_at')
    combined: Bool = (a | n) & ~o
    return [a.to_query(), o.to_query(), n.to_query(), combined.to_query()]


def dynamic(name: Optional[str], min_age: Optional[int], tags: List[str]) -> ESQuery:
    q = Bool()
    if name:
        q.must.append(Match('name', name))
    if min_age is not None:
        q.filter += Range('age', gte=min_age)
    if tags:
        q.filter += [Terms('tag', tags), Exists('tag')]
    q.should = Clauses([Term('featured', True)])
    q.must.append({'term': {'country': 'HU'}})
    q.must_not += Exists('deleted_at')
    n = Nested('comments')
    n.query.set({'term': {'comments.author': 'john'}})
    n.query.set(Term('comments.author', 'john'))
    clauses: Clauses = q.filter
    first: QueryLike = clauses[0]
    print(first, len(q.must), bool(q.should))
    for clause in q.filter:
        print(clause)
    return q.to_query()


class Published(Query):
    """ A custom query clause """

    def __init__(self) -> None:
        super().__init__()

    def to_query(self) -> ESQuery:
        return {'term': {'published': True}}


def custom() -> ESQuery:
    return (Published() & Term('a', 1)).to_query()
