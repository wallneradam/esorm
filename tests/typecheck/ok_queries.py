# pyright: standard
"""
Correct queries, type checkers must not report anything here
"""
from typing import Dict, List, Optional
from contextlib import aclosing
from datetime import datetime, date, timedelta
from uuid import UUID

from esorm import ESModel
from esorm.query import ESQuery
from esorm.watcher import DeleteWatcher


class User(ESModel):
    name: str
    age: int = 18
    country: str = 'HU'
    is_available: bool = True
    created: datetime = datetime(2020, 1, 1)


async def inline_queries(countries: List[str], ages: List[int], uid: UUID) -> None:
    # Bool query with a list of clauses
    users = await User.search(query={'bool': {'must': [{'range': {'age': {'gte': 18}}}]}})
    for user in users:
        print(user.name)
    # The result of search_one may be None
    user1 = await User.search_one({'bool': {'must': [{'match': {'name': {'query': 'John Doe'}}}]}})
    if user1 is not None:
        print(user1.name)
    # Shorthand forms
    await User.search({'term': {'country': 'HU'}})
    await User.search({'term': {'is_available': True}})
    await User.search({'term': {'created': date(2020, 1, 1)}})
    await User.search({'term': {'name.keyword': uid}})
    await User.search({'match': {'name': 'John'}})
    await User.search({'match_phrase': {'name': 'John Doe'}})
    await User.search({'prefix': {'name': 'Jo'}})
    await User.search({'wildcard': {'name': 'J*n'}})
    await User.search({'regexp': {'name': 'jo.*'}})
    await User.search({'fuzzy': {'name': 'jhon'}})
    # Term level queries
    await User.search({'term': {'country': {'value': 'HU', 'boost': 2.0, 'case_insensitive': True}}})
    await User.search({'terms': {'country': ['HU', 'AT']}})
    await User.search({'terms': {'country': countries}})
    await User.search({'terms': {'age': ages}})
    await User.search({'terms': {'age': (18, 19, 20)}})
    await User.search({'terms': {'country': {'index': 'countries', 'id': '1', 'path': 'codes'}}})
    await User.search({'range': {'created': {'gte': datetime(2020, 1, 1), 'lt': 'now-1d/d',
                                             'format': 'strict_date_optional_time', 'time_zone': '+01:00'}}})
    await User.search({'range': {'age': {'gte': 18, 'lte': 65.5}}})
    await User.search({'exists': {'field': 'name'}})
    await User.search({'ids': {'values': ['1', '2']}})
    await User.search({'regexp': {'name': {'value': 'jo.*', 'flags': 'ALL'}}})
    await User.search({'terms_set': {'tags': {'terms': ['a', 'b'], 'minimum_should_match_field': 'required'}}})
    # Full text queries
    await User.search({'match': {'name': {'query': 'John', 'operator': 'and', 'fuzziness': 'AUTO'}}})
    await User.search({'match_phrase_prefix': {'name': {'query': 'joh', 'max_expansions': 10}}})
    await User.search({'multi_match': {'query': 'john', 'fields': ['name^2', 'title'], 'type': 'best_fields'}})
    await User.search({'query_string': {'query': 'name:john AND age:>18', 'default_operator': 'AND'}})
    await User.search({'simple_query_string': {'query': 'john + doe', 'fields': ['name']}})
    await User.search({'combined_fields': {'query': 'john', 'fields': ['name', 'title']}})
    # Compound queries
    await User.search({'bool': {'must': {'match': {'name': {'query': 'x'}}}}})
    await User.search({'bool': {
        'filter': [{'term': {'country': {'value': 'HU'}}}],
        'must_not': [{'exists': {'field': 'nick'}}],
        'should': [{'match_phrase': {'name': {'query': 'john doe', 'slop': 1}}}],
        'minimum_should_match': 1,
        'boost': 1.5,
    }})
    await User.search({'bool': {'must': [{'bool': {'should': [{'term': {'country': 'HU'}}]}}]}})
    await User.search({'constant_score': {'filter': {'term': {'country': 'HU'}}, 'boost': 1.2}})
    await User.search({'dis_max': {'queries': [{'match': {'name': 'a'}}, {'match': {'title': 'a'}}]}})
    await User.search({'boosting': {'positive': {'match': {'name': 'a'}}, 'negative': {'term': {'country': 'AT'}},
                                    'negative_boost': 0.5}})
    await User.search({'function_score': {'query': {'match_all': {}},
                                          'functions': [{'filter': {'term': {'country': 'HU'}}, 'weight': 2}],
                                          'score_mode': 'sum', 'boost_mode': 'multiply'}})
    await User.search({'function_score': {'field_value_factor': {'field': 'age', 'modifier': 'log1p'}}})
    # Joining queries
    await User.search({'nested': {'path': 'tags', 'query': {'term': {'tags.name': 'x'}}, 'score_mode': 'avg'}})
    await User.search({'has_child': {'type': 'answer', 'query': {'match_all': {}}}})
    await User.search({'has_parent': {'parent_type': 'question', 'query': {'match_all': {}}}})
    # Geo queries
    await User.search({'geo_distance': {'distance': '10km', 'location': {'lat': 47.5, 'lon': 19.0}}})
    await User.search({'geo_bounding_box': {'location': {'top_left': [19.0, 48.0], 'bottom_right': '47.0,20.0'}}})
    # Vector and other queries
    await User.search({'knn': {'field': 'embedding', 'query_vector': [0.1, 0.2, 0.3], 'k': 10,
                               'num_candidates': 100, 'filter': {'term': {'country': 'HU'}}}})
    await User.search({'script_score': {'query': {'match_all': {}},
                                        'script': {'source': "cosineSimilarity(params.v, 'embedding') + 1.0",
                                                   'params': {'v': [0.1, 0.2, 0.3]}}}})
    await User.search({'script': {'script': "doc['age'].value > 18"}})
    await User.search({'more_like_this': {'fields': ['name'], 'like': 'John', 'min_term_freq': 1}})
    await User.search({'match_all': {}})
    await User.search({'match_none': {}})
    await User.search({'match_all': {'boost': 1.2, '_name': 'all'}})
    # Other methods
    await User.count({'term': {'country': 'HU'}})
    await User.count()
    async for user in User.scan({'range': {'age': {'gte': 18}}}, sort='age', batch_size=500):
        print(user.name)
    async for batch in User.scan_batches({'term': {'country': 'HU'}}):
        for user in batch:
            print(user.name)
    for user in await User.all():
        print(user.name)
    for user in await User.search_by_fields({'country': 'HU'}):
        print(user.name)
    async with aclosing(User.scan(sort='name')) as it:
        async for user in it:
            if user.name == 'admin':
                break


async def search_by_fields(first_name: Optional[str] = None, age: Optional[int] = None) -> None:
    fields: Dict[str, str] = {'first_name': 'John'}
    await User.search_by_fields(fields)
    filters = {'first_name': first_name, 'age': age}
    await User.search_by_fields({k: v for k, v in filters.items() if v is not None})
    await User.search_one_by_fields({'age': 18, 'is_available': True})


class PurgeDraft(DeleteWatcher):
    trigger = {"schedule": {"interval": "30s"}}
    _index = "draft"
    _query: ESQuery = {"bool": {"must": [{"wildcard": {"id": {"value": "_*"}}},
                                         {"range": {"created_at": {"lt": "now-1h"}}}]}}


class PurgeOld(DeleteWatcher):
    _index = "old"
    _query = {"range": {"created_at": {"lt": "now-1y"}}}


async def results(as_dict: bool) -> None:
    users = await User.search({'match_all': {}}, page_size=10, page=2, sort='name')
    print(users[0].name, len(users))
    by_id = await User.search({'match_all': {}}, res_dict=True)
    for doc_id, doc in by_id.items():
        print(doc_id, doc.name)
    by_id2 = await User.search_by_fields({'country': 'HU'}, res_dict=True)
    print(by_id2['1'].name)
    res = await User.search({'match_all': {}}, res_dict=as_dict)
    print(len(res))


async def query_variables(name: str, min_age: int) -> None:
    # Annotated query variable
    query: ESQuery = {'bool': {'must': [{'match': {'name': {'query': name}}}]}}
    await User.search(query)
    # Dynamically built query
    must: List[ESQuery] = []
    if name:
        must.append({'match': {'name': name}})
    if min_age:
        must.append({'range': {'age': {'gte': min_age}}})
    query2: ESQuery = {'bool': {'must': must}}
    await User.search(query2)
    # Construction with the TypedDict class
    query3 = ESQuery(bool={'filter': [{'term': {'country': 'HU'}}]})
    await User.search(query3)
    # Dict of field filters
    filters: Dict[str, str] = {'country': 'HU'}
    await User.search({'bool': {'filter': [{'term': {k: v}} for k, v in filters.items()]}})
    await User.search({'term': filters})
    # Date math and timedelta
    since = datetime.now() - timedelta(days=1)
    await User.search({'range': {'created': {'gte': since}}})
