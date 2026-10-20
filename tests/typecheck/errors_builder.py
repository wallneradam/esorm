# pyright: standard
"""
Incorrect usage of the query builder: type checkers must report every line marked with "# E", and nothing else
"""
from esorm import ESModel
from esorm.query import ESQuery
from esorm.builder import Bool, Term, Terms, Range, Match, MultiMatch, Nested, Knn, Boosting, GeoDistance


class User(ESModel):
    name: str
    age: int = 18


async def builder_errors() -> None:
    Term('name')  # E: missing value
    Term('name', object())  # E: wrong value type
    Term('name', 'x', bost=2.0)  # E: unknown parameter
    Terms('tag', 'abc')  # E: terms needs a list
    Range('age', gte=18, lesser=65)  # E: unknown range parameter
    Range('age', 18)  # E: the boundaries are keyword arguments
    Match('name', 'x', operator='adn')  # E: invalid operator
    MultiMatch('x', 'title')  # E: fields must be a list
    Nested(path='comments', query='x')  # E: the query must be a query
    Knn('embedding', 'abc')  # E: the query vector must be a list of floats
    Boosting(Term('a', 1), Term('b', 2))  # E: negative_boost is required
    GeoDistance('location', {'lat': 47.49}, '10km')  # E: lon is missing
    Bool(must=[{'trem': {'name': 'x'}}])  # E: typo in a query dict clause
    Bool(must=['x'])  # E: a string is not a query
    Bool(shuld=[Term('a', 1)])  # E: typo in bool
    Bool(minimum_should_match=[1])  # E: wrong type
    await User.search(Term('name', 'x'))  # E: use to_query()
    query: ESQuery = Bool()  # E: use to_query()
    print(query)
    q = Bool()
    q.must.append('x')  # E: a string is not a query
    q.mustt.append(Term('a', 1))  # E: unknown clause list
    q.must = [Term('a', 1)]  # E: use Clauses([...])
