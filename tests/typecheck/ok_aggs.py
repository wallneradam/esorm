# pyright: standard
"""
Correct aggregations, type checkers must not report anything here
"""
from typing import Optional

from esorm import ESModel
from esorm.aggs import ESAggs, ESAggsResponse


class User(ESModel):
    name: str
    age: int = 18
    country: str = 'HU'


async def aggregations() -> None:
    # Annotated aggregation definition
    aggs_def: ESAggs = {'avg_age': {'avg': {'field': 'age'}}}
    aggs = await User.aggregate(aggs_def)
    avg: Optional[float] = aggs['avg_age']['value']
    print(avg)
    # Inline definition, with query
    aggs = await User.aggregate({'countries': {'terms': {'field': 'country', 'size': 10}}},
                                query={'match': {'country': {'query': 'HU'}}})
    for bucket in aggs['countries']['buckets']:
        print(bucket['key'], bucket['doc_count'] + 1)
    # Sub-aggregations
    aggs = await User.aggregate({'countries': {'terms': {'field': 'country', 'order': {'_count': 'desc'}},
                                               'aggs': {'avg_age': {'avg': {'field': 'age'}}}}})
    for bucket in aggs['countries']['buckets']:
        print(bucket['key'], bucket['avg_age']['value'])
    # Single bucket aggregation with sub-aggregation
    res: ESAggsResponse = await User.aggregate({
        'adults': {'filter': {'range': {'age': {'gte': 18}}}, 'aggs': {'avg_age': {'avg': {'field': 'age'}}}},
    })
    print(res['adults']['doc_count'], res['adults']['avg_age']['value'])
    # Stats, percentiles, cardinality
    aggs = await User.aggregate({
        'age_stats': {'stats': {'field': 'age'}},
        'age_percentiles': {'percentiles': {'field': 'age', 'percents': [50, 95]}},
        'countries': {'cardinality': {'field': 'country'}},
    })
    print(aggs['age_stats']['count'], aggs['age_stats']['avg'], aggs['age_stats']['sum'])
    print(aggs['age_percentiles']['values']['50.0'], aggs['countries']['value'])
    # Histograms and ranges
    aggs = await User.aggregate({
        'ages': {'histogram': {'field': 'age', 'interval': 10, 'min_doc_count': 1}},
        'groups': {'range': {'field': 'age', 'ranges': [{'to': 18, 'key': 'child'}, {'from': 18}]}},
        'created': {'date_histogram': {'field': 'created', 'calendar_interval': 'month', 'format': 'yyyy-MM'}},
    })
    for bucket in aggs['groups']['buckets']:
        print(bucket['key'], bucket['from'], bucket['doc_count'])
    for bucket in aggs['created']['buckets']:
        print(bucket['key_as_string'])
    # Pipeline aggregation
    await User.aggregate({
        'per_country': {'terms': {'field': 'country'}, 'aggs': {'avg_age': {'avg': {'field': 'age'}}}},
        'max_avg_age': {'max_bucket': {'buckets_path': 'per_country>avg_age'}},
    })
    # Composite and top hits
    aggs = await User.aggregate({
        'pages': {'composite': {'sources': [{'country': {'terms': {'field': 'country'}}}], 'size': 100}},
        'top': {'terms': {'field': 'country'}, 'aggs': {'oldest': {'top_hits': {'size': 1, 'sort': [{'age': 'desc'}]}}}},
    })
    print(aggs['pages']['after_key'])
    # Filters aggregation
    await User.aggregate({'groups': {'filters': {'filters': {'hu': {'term': {'country': 'HU'}},
                                                             'at': {'term': {'country': 'AT'}}}}}})
