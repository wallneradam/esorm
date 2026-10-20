# pyright: standard
"""
Incorrect usage: type checkers must report every line marked with "# E", and nothing else
"""
from typing import List
from esorm import ESModel, ESBaseModel
from esorm.fields import TextField, NumericField, Field
from esorm.query import ESQuery
from esorm.aggs import ESAggs


class Address(ESBaseModel):
    city: str


class User(ESModel):
    name: str
    age: int = 18
    address: Address
    bad_text: int = TextField('x')  # E: str default for an int field
    bad_num: str = NumericField(1)  # E: int default for a str field
    bad_field: int = Field('x')  # E: str default for an int field


async def queries(name: str) -> None:
    await User.search({'trem': {'name': 'x'}})  # E: unknown query type
    await User.search({'term': {'name': {'valeu': 'x'}}})  # E: typo in term params
    await User.search({'term': {'name': {'boost': 2}}})  # E: missing required value
    await User.search({'match': {'name': {'query': 'x', 'operator': 'adn'}}})  # E: invalid operator
    await User.search({'match': {'name': {'qeury': 'x'}}})  # E: typo in match params
    await User.search({'terms': {'name': 'x'}})  # E: terms needs a list
    await User.search({'range': {'age': {'gte': 1, 'lesser': 2}}})  # E: unknown range param
    await User.search({'range': {'age': 18}})  # E: range needs params
    await User.search({'bool': {'must': [{'match': {'name': 'x'}}], 'shuld': []}})  # E: typo in bool
    await User.search({'bool': {'must': [{'mtach': {'name': 'x'}}]}})  # E: typo in a clause
    await User.search({'bool': {'filter': [{'bool': {'must_not': [{'exits': {'field': 'x'}}]}}]}})  # E: deep typo
    await User.search({'nested': {'path': 'a', 'query': {'bool': {'must': [{'trem': {'a': 1}}]}}}})  # E: deep typo
    await User.search({'bool': {'must': {'match': {'name': 'x'}}}})  # E: bool clauses must be in a list
    await User.search({'bool': {'minimum_should_match': [1]}})  # E: wrong type
    await User.search({'nested': {'path': 'tags'}})  # E: nested needs a query
    await User.search({'exists': {'name': 'x'}})  # E: exists needs field
    await User.search({'ids': {'values': '1'}})  # E: ids needs a list
    await User.search({'knn': {'field': 'v', 'query_vector': 'x'}})  # E: vector is not a string
    await User.search({'multi_match': {'query': 'x', 'type': 'best_field'}})  # E: invalid type literal
    await User.search({'match_all': {'boost': 'x'}})  # E: boost is a number
    await User.count({'term': {'age': [1, 2]}})  # E: term with a list
    async for _ in User.scan({'rnage': {'age': {'gte': 18}}}):  # E: typo in scan query
        pass

    must: List[ESQuery] = []
    must.append({'match': {'name': {'query': name, 'fuzzyness': 1}}})  # E: typo in an appended clause
    query: ESQuery = {'bool': {'must': must, 'boost': 'high'}}  # E: wrong boost type
    await User.search(query)


async def aggregations() -> None:
    await User.aggregate({'avg_age': {'avgg': {'field': 'age'}}})  # E: unknown aggregation
    await User.aggregate({'countries': {'terms': {'field': 'country', 'order': {'_count': 'dsc'}}}})  # E: order
    await User.aggregate({'countries': {'terms': {'field': 'country'}, 'agg': {}}})  # E: typo in sub-aggs key
    await User.aggregate({'c': {'terms': {'field': 'country'}, 'aggs': {'a': {'avgg': {'field': 'age'}}}}})  # E: sub-agg
    deep_aggs = {'a': {'terms': {'field': 'b'}, 'aggs': {'x': {'mx': {}}}}}
    await User.aggregate({'c': {'terms': {'field': 'c'}, 'aggs': deep_aggs}})  # E: deep sub-agg (unannotated)
    await User.aggregate({'c': {'terms': {'field': 'c'}, 'aggs': {'a': {'aggs': {'x': {'mx': {}}}}}}})  # E: deep
    await User.aggregate({'avg_age': {'avg': {'field': 'age'}}}, query={'trem': {'a': 1}})  # E: typo in query
    aggs_def: ESAggs = {'ranges': {'range': {'field': 'age', 'ranges': [{'from': 1, 'too': 2}]}}}  # E: typo
    aggs = await User.aggregate(aggs_def)
    for bucket in aggs['countries']['buckets']:
        count: str = bucket['doc_count']  # E: doc_count is int
        print(count)
    avg: float = aggs['avg_age']['value']  # E: value can be None
    print(avg)


async def results() -> None:
    users = await User.search({'match_all': {}})
    users.items()  # E: list has no items
    by_id = await User.search({'match_all': {}}, res_dict=True)
    by_id.append(None)  # E: dict has no append
    user = await User.search_one({'match_all': {}})
    print(user.name)  # E: user may be None
    user2 = await User.get('1')
    print(user2.nmae)  # E: unknown attribute
    user2.agee = 1  # E: unknown attribute
    print(user2.address.ctiy)  # E: unknown attribute of a nested model
    user2.age = 'x'  # E: wrong type
