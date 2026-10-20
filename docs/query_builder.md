<a id="query-builder"></a>
# 🏗 Query builder

The query builder (`esorm.builder`) builds ElasticSearch queries from classes instead of nested dicts. It produces
**exactly the same query dict** (`ESQuery`) as you would write by hand. The search methods of the models accept the
builder queries directly, and `to_query()` returns the query dict for everything else. The builder doesn't run
anything, there is no `await` in it: it only builds the query.

- [Why a query builder](#why-a-query-builder)
- [Quick start](#builder-quick-start)
- [Two ways to build a query: parameters and `with` blocks](#two-ways-to-build-a-query)
- [How `with` blocks work](#how-with-blocks-work)
- [Combining queries with operators](#combining-queries-with-operators)
- [Using the built query](#using-the-built-query)
- [Building queries dynamically](#building-queries-dynamically)
- [Query clauses](#query-clauses)
    - [Compound queries](#builder-compound-queries)
    - [Term level queries](#builder-term-level-queries)
    - [Full text queries](#builder-full-text-queries)
    - [Joining queries](#builder-joining-queries)
    - [Geo queries](#builder-geo-queries)
    - [Vector queries](#builder-vector-queries)
    - [Any other query](#builder-any-other-query)
    - [Common parameters](#builder-common-parameters)
- [Custom query clauses](#custom-query-clauses)
- [Type checking](#builder-type-checking)
- [Tips and pitfalls](#builder-tips-and-pitfalls)

<a id="why-a-query-builder"></a>
## Why a query builder

Query dicts are simple, but real queries quickly become deep: lists in dicts in lists, and every clause needs its
own braces. This is a query for active adult users named John, who are not deleted:

```python
from esorm.query import ESQuery

query: ESQuery = {
    'bool': {
        'must': [
            {'match': {'name': 'John'}},
        ],
        'filter': [
            {'range': {'age': {'gte': 18}}},
            {'term': {'active': True}},
        ],
        'must_not': [
            {'exists': {'field': 'deleted_at'}},
        ],
    }
}
```

The same with the query builder:

```python
from esorm.builder import Bool, Match, Range, Term, Exists

with Bool() as q:
    Match('name', 'John')
    with q.filter:
        Range('age', gte=18)
        Term('active', True)
    with q.must_not:
        Exists('deleted_at')
```

```json
{
  "bool": {
    "must": [{"match": {"name": "John"}}],
    "filter": [{"range": {"age": {"gte": 18}}}, {"term": {"active": true}}],
    "must_not": [{"exists": {"field": "deleted_at"}}]
  }
}
```

(The JSON blocks in this chapter show the result of `q.to_query()`, they are checked by the tests.)

What you get:

- **Less noise**: no braces and lists, the nesting of the `with` blocks follows the nesting of the query.
- **Every parameter is checked**: the clauses are ordinary classes with typed parameters, so your IDE autocompletes
  them, and typos, wrong value types and missing arguments are reported by the type checkers.
- **Building queries in code is easy**: clauses can be appended conditionally, combined with `&`, `|` and `~`, and
  reused.
- **Nothing to migrate**: the result is the same query dict, builder queries and query dicts can be mixed freely.

<a id="builder-quick-start"></a>
## Quick start

```python
from esorm import ESModel
from esorm.fields import text, keyword
from esorm.builder import Bool, Match, Range, Term


class User(ESModel):
    name: text
    country: keyword
    age: int
    active: bool = True


async def search_users():
    with Bool() as q:
        Match('name', 'John')  # directly in the Bool block: must
        with q.filter:
            Range('age', gte=18)
            Term('country', 'HU')

    users = await User.search(q)
    for user in users:
        print(user.name)
```

1. Import the query clauses from `esorm.builder`.
2. Build the query with parameters, `with` blocks or operators.
3. Give it to `search()`, `count()` or any other method of the model. If you need the query dict itself, call
   `to_query()`.

<a id="two-ways-to-build-a-query"></a>
## Two ways to build a query: parameters and `with` blocks

<a id="builder-parameters"></a>
### Parameters

Every clause can be given as a parameter, like in a dict, just with classes:

```python
from esorm.builder import Bool, Match, Range, Term, Exists

q = Bool(
    must=[Match('name', 'John')],
    filter=[Range('age', gte=18), Term('active', True)],
    must_not=Exists('deleted_at'),  # a single clause doesn't need a list
)
```

```json
{
  "bool": {
    "must": [{"match": {"name": "John"}}],
    "filter": [{"range": {"age": {"gte": 18}}}, {"term": {"active": true}}],
    "must_not": [{"exists": {"field": "deleted_at"}}]
  }
}
```

This is the best for short queries, which fit in one or two lines:

```python
from esorm.builder import Bool, Term, Range

q = Bool(filter=[Term('country', 'HU'), Range('age', gte=18)])
```

```json
{"bool": {"filter": [{"term": {"country": "HU"}}, {"range": {"age": {"gte": 18}}}]}}
```

<a id="builder-with-blocks"></a>
### `with` blocks

A compound query (and its clause lists, e.g. `q.must`) can be used in a `with` block. Every query clause which is
created inside the block is added to it, you don't have to collect them:

```python
from esorm.builder import Bool, Match, Range, Term, Exists

with Bool() as q:
    with q.must:
        Match('name', 'John')
    with q.filter:
        Range('age', gte=18)
        Term('active', True)
    with q.must_not:
        Exists('deleted_at')
```

```json
{
  "bool": {
    "must": [{"match": {"name": "John"}}],
    "filter": [{"range": {"age": {"gte": 18}}}, {"term": {"active": true}}],
    "must_not": [{"exists": {"field": "deleted_at"}}]
  }
}
```

The clauses created directly in the `with Bool()` block are added to `must`, so the `with q.must:` line can be left
out:

```python
from esorm.builder import Bool, Match, Range

with Bool() as q:
    Match('name', 'John')
    with q.filter:
        Range('age', gte=18)
```

```json
{
  "bool": {
    "must": [{"match": {"name": "John"}}],
    "filter": [{"range": {"age": {"gte": 18}}}]
  }
}
```

The `with` blocks are the best for big queries with deep nesting, because the indentation shows the structure:

```python
from esorm.builder import Bool, Match, Term, Range, Nested

with Bool() as q:
    Term('active', True)
    with q.should:
        Match('title', 'elasticsearch')
        with Nested('comments'):
            Match('comments.text', 'elasticsearch')
            Range('comments.stars', gte=4)
    q.minimum_should_match = 1
```

```json
{
  "bool": {
    "must": [{"term": {"active": true}}],
    "should": [
      {"match": {"title": "elasticsearch"}},
      {
        "nested": {
          "path": "comments",
          "query": {
            "bool": {
              "must": [
                {"match": {"comments.text": "elasticsearch"}},
                {"range": {"comments.stars": {"gte": 4}}}
              ]
            }
          }
        }
      }
    ],
    "minimum_should_match": 1
  }
}
```

The parameters of a compound query can be given in the constructor (`with Bool(minimum_should_match=1) as q:`), or set
as attributes, also inside the block (`q.minimum_should_match = 1`).

<a id="builder-mixing-styles"></a>
### Mixing them

The two styles can be mixed in any way. A clause given as a parameter inside a `with` block is not added to the block
twice: it belongs to the query it was given to.

```python
from esorm.builder import Bool, Term, Range, Match

with Bool() as q:
    Match('name', 'John')
    with q.should:
        Term('tag', 'python')
        Bool(must=[Term('tag', 'es'), Range('stars', gte=100)])
```

```json
{
  "bool": {
    "must": [{"match": {"name": "John"}}],
    "should": [
      {"term": {"tag": "python"}},
      {"bool": {"must": [{"term": {"tag": "es"}}, {"range": {"stars": {"gte": 100}}}]}}
    ]
  }
}
```

<a id="how-with-blocks-work"></a>
## How `with` blocks work

The rules are simple, but it is worth knowing them exactly:

1. **Every builder query created inside a `with` block is added to the innermost block**, in the order of creation.
   It doesn't matter how it is created: a statement, a variable assignment or a function call that creates
   clauses all add them.

2. **Each compound query has a default place** for the clauses created directly in its block:

   | Block | The clauses go to |
   |---|---|
   | `with Bool() as q:` | `q.must` |
   | `with q.must:`, `with q.filter:`, `with q.should:`, `with q.must_not:` | that list |
   | `with Nested(...):`, `with HasChild(...):`, `with HasParent(...):` | `query` |
   | `with FunctionScore(...):` | `query` |
   | `with ConstantScore():` | `filter` |
   | `with Boosting(...) as q:` | `q.positive` (use `with q.negative:` for the other one) |
   | `with DisMax() as q:` | `q.queries` |
   | `with Knn(...) as q:` | `q.filter` |

   The single query places (e.g. the `query` of `Nested`) accept more clauses too: all of them must match, so they
   are combined into a `bool.must` query.

3. **A clause given to another query explicitly is moved there**: if you pass it as a parameter, combine it with an
   operator, or `append()` it to a clause list inside the block, it is removed from the block. So nothing is
   duplicated, and the order is kept: the inner clauses are created first, then the outer query takes them, and only
   the outer query stays in the block.

   ```python
   from esorm.builder import Bool, Term

   with Bool() as q:
       Term('a', 1)
       Bool(should=[Term('b', 2), Term('c', 3)])  # one bool clause, not three clauses
       q.filter += Term('d', 4) | Term('e', 5)  # one clause in filter
   ```

   ```json
   {
     "bool": {
       "must": [
         {"term": {"a": 1}},
         {"bool": {"should": [{"term": {"b": 2}}, {"term": {"c": 3}}]}}
       ],
       "filter": [
         {
           "bool": {
             "should": [{"term": {"d": 4}}, {"term": {"e": 5}}],
             "minimum_should_match": 1
           }
         }
       ]
     }
   }
   ```

4. **When the block ends, its clauses are final.** Using a clause in another query after the block doesn't remove it
   from the block, the clause is simply used in both places.

5. **Query dicts are not collected**, only builder queries are (a dict doesn't know where it was created). Wrap a
   query dict into [`Raw`](#builder-any-other-query), or `append()` it:

   ```python
   from esorm.builder import Bool, Raw

   with Bool() as q:
       Raw({'span_term': {'text': 'fox'}})
       q.filter.append({'term': {'active': True}})
   ```

   ```json
   {
     "bool": {
       "must": [{"span_term": {"text": "fox"}}],
       "filter": [{"term": {"active": true}}]
     }
   }
   ```

6. **Exceptions close the blocks normally**: if an exception is raised inside a block, the block ends, and the
   clauses created before the exception stay in the query.

7. **Threads and async tasks have their own blocks**: the current block is stored in a context variable, so queries
   built at the same time in different threads or async tasks don't mix, even if there is an `await` inside the
   block.

8. **These are ordinary (synchronous) `with` blocks**, use `with`, not `async with`. Entering and leaving a block
   doesn't do anything with ElasticSearch.

Outside of `with` blocks a created clause is not added anywhere, it is just an object, like any other.

<a id="combining-queries-with-operators"></a>
## Combining queries with operators

Queries can be combined with the `&`, `|` and `~` operators, the result is a `Bool` query:

| Operator | Meaning | Result |
|---|---|---|
| `a & b` | both must match | `{'bool': {'must': [a, b]}}` |
| `a \| b` | at least one of them must match | `{'bool': {'should': [a, b], 'minimum_should_match': 1}}` |
| `~a` | must not match | `{'bool': {'must_not': [a]}}` |

```python
from esorm.builder import Term, Exists

q = (Term('tag', 'python') | Term('tag', 'es')) & ~Exists('deleted_at')
```

```json
{
  "bool": {
    "must": [
      {
        "bool": {
          "should": [{"term": {"tag": "python"}}, {"term": {"tag": "es"}}],
          "minimum_should_match": 1
        }
      },
      {"bool": {"must_not": [{"exists": {"field": "deleted_at"}}]}}
    ]
  }
}
```

Chains are flattened: `a & b & c` is one bool query with 3 must clauses, not nested bool queries. A bool query is
flattened only if it has nothing else (no other clauses and no parameters), so the meaning never changes:

```python
from esorm.builder import Term

q = Term('a', 1) & Term('b', 2) & Term('c', 3)
```

```json
{"bool": {"must": [{"term": {"a": 1}}, {"term": {"b": 2}}, {"term": {"c": 3}}]}}
```

Some details:

- `|` always sets `minimum_should_match: 1`. A bool query with only should clauses requires one match anyway, but if
  you add must or filter clauses to it later, the should clauses would become optional without it.
- The precedence is the same as in Python: `~` first, then `&`, then `|`. So `a | b & c` means `a | (b & c)`. Use
  parentheses to make it clear.
- The operands are not changed, a new query is created. The operands can be query dicts too
  (`Term('a', 1) & {'match_all': {}}`), but for type checkers start with a builder query, or wrap the dict into
  `Raw()`.
- In a `with` block the result of an operator is collected (as one clause), the operands are not. But don't write
  an operator expression alone as a statement in a block (`Term('a', 1) | Term('b', 2)`): it works, but pyright
  (VS Code) and PyCharm report it as an expression without effect, because they don't know that creating the query
  has an effect. Add it to a clause list instead, it is also more explicit:

  ```python
  from esorm.builder import Bool, Term, Exists

  with Bool() as q:
      Term('active', True)
      q.must += Term('tag', 'python') | Term('tag', 'es')
      q.filter += ~Exists('deleted_at')
  ```

  ```json
  {
    "bool": {
      "must": [
        {"term": {"active": true}},
        {
          "bool": {
            "should": [{"term": {"tag": "python"}}, {"term": {"tag": "es"}}],
            "minimum_should_match": 1
          }
        }
      ],
      "filter": [{"bool": {"must_not": [{"exists": {"field": "deleted_at"}}]}}]
    }
  }
  ```

<a id="using-the-built-query"></a>
## Using the built query

Every method of the models which has a query parameter accepts builder queries too: `search()`, `search_one()`,
`count()`, `aggregate()`, `scan()` and `scan_batches()`, also on models decorated with `Pagination` or `Sort`:

```python
from esorm import ESModel
from esorm.fields import text, keyword
from esorm.builder import Bool, Match, Range, Term


class User(ESModel):
    name: text
    country: keyword
    age: int


async def examples():
    q = Bool(must=[Match('name', 'John')], filter=[Range('age', gte=18)])

    users = await User.search(q, page_size=10, sort='age')
    user = await User.search_one(q)
    count = await User.count(q)
    aggs = await User.aggregate({'countries': {'terms': {'field': 'country'}}}, query=q)
    async for user in User.scan(q):
        print(user.name)
    print(users, user, count, aggs)
```

`to_query()` returns the query dict (`ESQuery`), the methods call it too. Use it if you need the dict itself, e.g. to
put it into a query dict, to log it, or to give it to the ES client directly (`esorm.es.search(query=...)`).

Things worth knowing:

- **Query dicts can be clauses**: wherever a clause is expected, a query dict is accepted too, it is put into the
  result as it is. So existing query dicts can be reused in the builder:

  ```python
  from esorm.query import ESQuery
  from esorm.builder import Bool, Match

  only_active: ESQuery = {'term': {'active': True}}
  q = Bool(must=[Match('name', 'John')], filter=[only_active])
  ```

  ```json
  {"bool": {"must": [{"match": {"name": "John"}}], "filter": [{"term": {"active": true}}]}}
  ```

- **A built query can be a part of a query dict**: `{'bool': {'filter': [q.to_query(), ...]}}`.
- **The result is a new dict** every time `to_query()` is called (except the query dicts given as clauses, which are
  in the result as they are). It can be modified, it doesn't change the builder query.
- **For debugging** print the query: `repr()` shows the query dict, e.g. `<Term {'term': {'age': 18}}>`. To try it in
  Kibana Dev Tools, print it as JSON: `json.dumps(q.to_query(), default=str)` (`default=str` is needed for dates).
- **The query can be stored and reused**: a builder query can be used in more queries, and `to_query()` can be
  called any number of times.

<a id="building-queries-dynamically"></a>
## Building queries dynamically

Search forms and API filters usually have optional fields. The clause lists of a `Bool` query work like Python lists,
so clauses can be added conditionally:

```python
from typing import List, Optional

from esorm import ESModel
from esorm.fields import text, keyword
from esorm.builder import Bool, Match, Range, Terms


class Product(ESModel):
    name: text
    category: keyword
    price: float


async def search_products(search: Optional[str] = None,
                          min_price: Optional[float] = None,
                          max_price: Optional[float] = None,
                          categories: Optional[List[str]] = None):
    q = Bool()
    if search:
        q.must.append(Match('name', search, operator='and'))
    if min_price is not None or max_price is not None:
        q.filter += Range('price', gte=min_price, lte=max_price)  # None is left out
    if categories:
        q.filter += Terms('category', categories)
    return await Product.search(q)
```

If no clause is added, the result is `{'bool': {}}`, which matches all documents, so the empty search form returns
everything.

The same with `with` blocks and `if` statements inside them:

```python
from esorm.builder import Bool, Match, Range

search, min_price = 'apple', 10

with Bool() as q:
    if search:
        Match('name', search)
    with q.filter:
        if min_price is not None:
            Range('price', gte=min_price)
```

```json
{
  "bool": {
    "must": [{"match": {"name": "apple"}}],
    "filter": [{"range": {"price": {"gte": 10}}}]
  }
}
```

Clause lists (`q.must`, `q.filter`, `q.should`, `q.must_not` of `Bool`, `q.queries` of `DisMax` and `q.filter` of
`Knn`) are `Clauses` objects:

| Operation | Example |
|---|---|
| Add a clause | `q.must.append(Term('a', 1))`, `q.must += Term('a', 1)` |
| Add more clauses | `q.must.extend([...])`, `q.must += [...]` |
| Remove all | `q.must.clear()` |
| Replace | `q.must = Clauses([...])` |
| Is it empty | `if q.must:`, `len(q.must)` |
| Read | `q.must[0]`, `for clause in q.must:` |
| Collect in a block | `with q.must:` |

The single clause places (`q.query` of `Nested`, `q.positive` of `Boosting`, ...) are `Clause` objects, they can be
set by `set()`, e.g. `q.positive.set(Match('text', 'apple'))`, or in a `with` block.

The other parameters of the leaf clauses are in their `params` dict (in the form ES uses them), it can be changed
too: `t.params['boost'] = 2.0`.

<a id="query-clauses"></a>
## Query clauses

Every query clause of the builder creates the ES query of the same name. The links point to the ElasticSearch
documentation of the query, the parameters have the same names and meaning, except `name`, which is `_name` in ES
(see [Common parameters](#builder-common-parameters)). The API reference has the full parameter lists.

The leaf clauses use the shortest form ES accepts, like you would write them by hand: `{'term': {'age': 18}}` instead
of `{'term': {'age': {'value': 18}}}`, if there are no other parameters.

<a id="builder-compound-queries"></a>
### Compound queries

| Class | ES query | `with` block collects to |
|---|---|---|
| `Bool(must, filter, should, must_not, *, minimum_should_match, boost, name)` | [bool](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-bool-query) | `must` |
| `Boosting(positive, negative, *, negative_boost, ...)` | [boosting](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-boosting-query) | `positive` |
| `ConstantScore(filter, *, boost, name)` | [constant_score](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-constant-score-query) | `filter` |
| `DisMax(queries, *, tie_breaker, ...)` | [dis_max](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-dis-max-query) | `queries` |
| `FunctionScore(query, *, functions, weight, field_value_factor, ...)` | [function_score](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-function-score-query) | `query` |

The four clause lists of `Bool`:

- `must`: must match, and contributes to the score
- `filter`: must match, but doesn't contribute to the score, so ES can cache it. Use it for exact conditions (terms,
  ranges, ...) which don't need scoring.
- `should`: should match. If there are no must or filter clauses, at least one of them must match, otherwise they only
  increase the score, unless `minimum_should_match` is set.
- `must_not`: must not match (in filter context, no scoring)

Boosting: the documents matching the negative query are not excluded, only their score is decreased:

```python
from esorm.builder import Boosting, Match

with Boosting(negative_boost=0.5) as q:
    Match('text', 'apple')  # positive
    with q.negative:
        Match('text', 'pie')
```

```json
{
  "boosting": {
    "positive": {"match": {"text": "apple"}},
    "negative": {"match": {"text": "pie"}},
    "negative_boost": 0.5
  }
}
```

Constant score: every matching document gets the same score:

```python
from esorm.builder import ConstantScore, Term

q = ConstantScore(Term('active', True), boost=1.2)
```

```json
{"constant_score": {"filter": {"term": {"active": true}}, "boost": 1.2}}
```

Disjunction max: the score is the score of the best matching clause:

```python
from esorm.builder import DisMax, Match

with DisMax(tie_breaker=0.7) as q:
    Match('title', 'quick fox')
    Match('body', 'quick fox')
```

```json
{
  "dis_max": {
    "queries": [{"match": {"title": "quick fox"}}, {"match": {"body": "quick fox"}}],
    "tie_breaker": 0.7
  }
}
```

Function score: modify the score, e.g. by a field value. The score functions are dicts (`ESScoreFunction`), as in ES:

```python
from esorm.builder import FunctionScore, Match

q = FunctionScore(Match('title', 'elasticsearch'),
                  field_value_factor={'field': 'likes', 'modifier': 'log1p'},
                  boost_mode='sum')
```

```json
{
  "function_score": {
    "query": {"match": {"title": "elasticsearch"}},
    "field_value_factor": {"field": "likes", "modifier": "log1p"},
    "boost_mode": "sum"
  }
}
```

If a function needs a filter query, give it as a dict: `{'filter': Term('a', 1).to_query(), 'weight': 2}`.

<a id="builder-term-level-queries"></a>
### Term level queries

Term level queries search for exact values, they are not analyzed. Use them on `keyword`, numeric, date and boolean
fields, mostly in filter context.

| Class | ES query | Example |
|---|---|---|
| `Term(field, value)` | [term](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-term-query) | `Term('country', 'HU')` |
| `Terms(field, values)` | [terms](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-terms-query) | `Terms('tag', ['python', 'es'])` |
| `TermsSet(field, terms, *, minimum_should_match_field, ...)` | [terms_set](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-terms-set-query) | `TermsSet('skills', ['a', 'b'], minimum_should_match_field='required')` |
| `Range(field, *, gt, gte, lt, lte, format, time_zone, relation)` | [range](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-range-query) | `Range('age', gte=18, lt=65)` |
| `Exists(field)` | [exists](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-exists-query) | `Exists('email')` |
| `Ids(values)` | [ids](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-ids-query) | `Ids(['1', '2'])` |
| `Prefix(field, value)` | [prefix](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-prefix-query) | `Prefix('code', 'HU-')` |
| `Wildcard(field, value)` | [wildcard](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-wildcard-query) | `Wildcard('code', 'HU-*-01')` |
| `Regexp(field, value, *, flags, ...)` | [regexp](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-regexp-query) | `Regexp('code', 'HU-[0-9]+')` |
| `Fuzzy(field, value, *, fuzziness, ...)` | [fuzzy](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-fuzzy-query) | `Fuzzy('name', 'jonh', fuzziness='AUTO')` |

The values can be strings, numbers, booleans, dates, datetimes, UUIDs and Decimals, the ES client serializes them.
Range boundaries can be date math strings too (`'now-1d/d'`):

```python
from datetime import date
from esorm.builder import Bool, Term, Terms, Range, Exists

with Bool() as q:
    with q.filter:
        Term('country', 'HU')
        Terms('tag', ['python', 'es'])
        Range('created', gte=date(2024, 1, 1), lt='now/d')
    with q.must_not:
        Exists('deleted_at')
```

```json
{
  "bool": {
    "filter": [
      {"term": {"country": "HU"}},
      {"terms": {"tag": ["python", "es"]}},
      {"range": {"created": {"gte": "2024-01-01", "lt": "now/d"}}}
    ],
    "must_not": [{"exists": {"field": "deleted_at"}}]
  }
}
```

(The JSON shows the date as ES gets it, the query dict has a `date` object there.)

With parameters the long form is used:

```python
from esorm.builder import Term

q = Term('tag', 'Python', case_insensitive=True)
```

```json
{"term": {"tag": {"value": "Python", "case_insensitive": true}}}
```

`Terms` accepts a [terms lookup](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-terms-query#query-dsl-terms-lookup)
too, to get the values from a field of another document:

```python
from esorm.builder import Terms

q = Terms('color', {'index': 'my-index', 'id': '2', 'path': 'color'})
```

```json
{"terms": {"color": {"index": "my-index", "id": "2", "path": "color"}}}
```

<a id="builder-full-text-queries"></a>
### Full text queries

Full text queries analyze the query text the same way as the field was analyzed at indexing. Use them on `text`
fields (in ESORM a `str` field is `keyword`, use `esorm.fields.text` or `TextField` for full text search).

| Class | ES query | Example |
|---|---|---|
| `Match(field, query, *, operator, minimum_should_match, fuzziness, ...)` | [match](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-query) | `Match('name', 'John Doe')` |
| `MatchPhrase(field, query, *, slop, ...)` | [match_phrase](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-query-phrase) | `MatchPhrase('text', 'quick brown fox')` |
| `MatchPhrasePrefix(field, query, ...)` | [match_phrase_prefix](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-query-phrase-prefix) | `MatchPhrasePrefix('title', 'quick brown f')` |
| `MatchBoolPrefix(field, query, ...)` | [match_bool_prefix](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-bool-prefix-query) | `MatchBoolPrefix('title', 'quick brown f')` |
| `MultiMatch(query, fields, *, type, ...)` | [multi_match](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-multi-match-query) | `MultiMatch('fox', ['title^3', 'body'])` |
| `QueryString(query, *, default_field, fields, ...)` | [query_string](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-query-string-query) | `QueryString('(new york) OR (big apple)')` |
| `SimpleQueryString(query, *, fields, ...)` | [simple_query_string](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-simple-query-string-query) | `SimpleQueryString('"fried eggs" +potato')` |
| `MatchAll()` | [match_all](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-all-query) | `MatchAll()` |
| `MatchNone()` | [match_none](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-all-query#query-dsl-match-none-query) | `MatchNone()` |

```python
from esorm.builder import Bool, Match, MatchPhrase, MultiMatch

with Bool() as q:
    MultiMatch('python elasticsearch', ['title^3', 'body'], type='best_fields')
    with q.should:
        MatchPhrase('title', 'python elasticsearch', boost=2.0)
        Match('tags', 'orm', operator='and')
```

```json
{
  "bool": {
    "must": [
      {
        "multi_match": {
          "query": "python elasticsearch",
          "fields": ["title^3", "body"],
          "type": "best_fields"
        }
      }
    ],
    "should": [
      {"match_phrase": {"title": {"query": "python elasticsearch", "boost": 2.0}}},
      {"match": {"tags": {"query": "orm", "operator": "and"}}}
    ]
  }
}
```

For search boxes use `SimpleQueryString`: its syntax is simple and it never returns an error. `QueryString` has the
full Lucene syntax, but it returns an error for an invalid query, so it is not for user input.

<a id="builder-joining-queries"></a>
### Joining queries

| Class | ES query | `with` block collects to |
|---|---|---|
| `Nested(path, query, *, score_mode, inner_hits, ...)` | [nested](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-nested-query) | `query` |
| `HasChild(type, query, *, min_children, max_children, ...)` | [has_child](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-has-child-query) | `query` |
| `HasParent(parent_type, query, *, score, ...)` | [has_parent](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-has-parent-query) | `query` |
| `ParentId(type, id)` | [parent_id](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-parent-id-query) | |

`List[SubModel]` fields are `nested` fields in ESORM. A nested query matches if one nested document matches all its
clauses. The field names are full paths:

```python
from typing import List

from esorm import ESModel, ESBaseModel
from esorm.fields import keyword
from esorm.builder import Nested, Term, Range


class Comment(ESBaseModel):
    author: keyword
    stars: int


class Post(ESModel):
    title: str
    comments: List[Comment] = []


# Posts which have a comment from John with at least 4 stars
with Nested('comments') as q:
    Term('comments.author', 'john')
    Range('comments.stars', gte=4)
```

```json
{
  "nested": {
    "path": "comments",
    "query": {
      "bool": {
        "must": [
          {"term": {"comments.author": "john"}},
          {"range": {"comments.stars": {"gte": 4}}}
        ]
      }
    }
  }
}
```

Without `Nested` these clauses don't match anything, because the nested documents are stored as separate (hidden)
documents. A single `SubModel` field (not a list) is an `object` field, it doesn't need `Nested`: its fields can be
queried directly, e.g. `Term('address.city', 'Budapest')`.

`inner_hits={}` returns the matching nested documents in the raw ES response, the models returned by `search()`
don't contain them.

<a id="builder-geo-queries"></a>
### Geo queries

| Class | ES query |
|---|---|
| `GeoDistance(field, point, distance, *, distance_type, ...)` | [geo_distance](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-geo-distance-query) |
| `GeoBoundingBox(field, *, top_left, bottom_right, ...)` | [geo_bounding_box](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-geo-bounding-box-query) |

Points can be `{'lat': .., 'lon': ..}` dicts, `[lon, lat]` lists, `"lat,lon"` strings, geohashes or WKT strings:

```python
from esorm.builder import Bool, GeoDistance, GeoBoundingBox

with Bool() as q:
    with q.filter:
        GeoDistance('location', {'lat': 47.4979, 'lon': 19.0402}, '10km')
        GeoBoundingBox('location',
                       top_left={'lat': 47.6, 'lon': 18.9},
                       bottom_right={'lat': 47.4, 'lon': 19.2})
```

```json
{
  "bool": {
    "filter": [
      {"geo_distance": {"distance": "10km", "location": {"lat": 47.4979, "lon": 19.0402}}},
      {
        "geo_bounding_box": {
          "location": {
            "top_left": {"lat": 47.6, "lon": 18.9},
            "bottom_right": {"lat": 47.4, "lon": 19.2}
          }
        }
      }
    ]
  }
}
```

<a id="builder-vector-queries"></a>
### Vector queries

| Class | ES query | `with` block collects to |
|---|---|---|
| `Knn(field, query_vector, *, k, num_candidates, filter, similarity, ...)` | [knn](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-knn-query) | `filter` |

The filter is applied during the vector search, so `k` matching results are returned (if there are enough), even if
most documents don't match the filter:

```python
from esorm.builder import Knn, Term

with Knn('embedding', [0.12, 0.85, 0.33], k=10, num_candidates=100) as q:
    Term('lang', 'en')
```

```json
{
  "knn": {
    "field": "embedding",
    "query_vector": [0.12, 0.85, 0.33],
    "k": 10,
    "num_candidates": 100,
    "filter": {"term": {"lang": "en"}}
  }
}
```

Hybrid search (full text and vector similarity together) is a bool query with both:

```python
from esorm.builder import Bool, Knn, Match

q = (Match('title', 'red shoes')
     | Knn('embedding', [0.12, 0.85, 0.33], k=10, num_candidates=100))
```

```json
{
  "bool": {
    "should": [
      {"match": {"title": "red shoes"}},
      {
        "knn": {
          "field": "embedding",
          "query_vector": [0.12, 0.85, 0.33],
          "k": 10,
          "num_candidates": 100
        }
      }
    ],
    "minimum_should_match": 1
  }
}
```

<a id="builder-any-other-query"></a>
### Any other query

The builder has the most often used queries. For the others (span, percolate, script, ...) use their query dicts:

- as a parameter or with `append()`, a query dict can be used directly,
- in a `with` block or with an operator, wrap it into `Raw`:

```python
from esorm.builder import Bool, Raw, Match

with Bool() as q:
    Match('title', 'fox')
    Raw({'span_near': {
        'clauses': [{'span_term': {'text': 'quick'}}, {'span_term': {'text': 'fox'}}],
        'slop': 2,
    }})
```

```json
{
  "bool": {
    "must": [
      {"match": {"title": "fox"}},
      {
        "span_near": {
          "clauses": [{"span_term": {"text": "quick"}}, {"span_term": {"text": "fox"}}],
          "slop": 2
        }
      }
    ]
  }
}
```

`Raw` accepts an `ESQuery`, so the query dict is type checked too.

<a id="builder-common-parameters"></a>
### Common parameters

- `boost`: boosting value of the query, it multiplies the score.
- `name`: name of the query, it is `_name` in ES. The names of the matching queries are returned in the
  `matched_queries` of the hits, so you can see which clauses matched:

  ```python
  from esorm.builder import Bool, Match

  q = Bool(should=[Match('title', 'fox', name='title'), Match('body', 'fox', name='body')])
  ```

  ```json
  {
    "bool": {
      "should": [
        {"match": {"title": {"query": "fox", "_name": "title"}}},
        {"match": {"body": {"query": "fox", "_name": "body"}}}
      ]
    }
  }
  ```

- The parameters which are not given (`None`) are left out of the query, so ES uses its defaults.
- Leaf clause parameters are keyword-only: `Range('age', gte=18)`, not `Range('age', 18)`.

<a id="custom-query-clauses"></a>
## Custom query clauses

If you often use the same query, or a query which is not in the builder, write your own clause: subclass `Query`,
implement `to_query()`, and call `super().__init__()` at the end of `__init__()` (that's where the clause is added to
the current `with` block, so it must be complete by then):

```python
from esorm.query import ESQuery
from esorm.builder import Query, Bool, Match


class Published(Query):
    """ Only the published documents """

    def __init__(self, published: bool = True):
        self.published = published
        super().__init__()

    def to_query(self) -> ESQuery:
        return {'term': {'published': self.published}}


with Bool() as q:
    Match('title', 'fox')
    with q.filter:
        Published()
```

```json
{"bool": {"must": [{"match": {"title": "fox"}}], "filter": [{"term": {"published": true}}]}}
```

Custom clauses work in `with` blocks and with the operators, like the built-in ones.

<a id="builder-type-checking"></a>
## Type checking

The query clauses are ordinary classes with typed parameters, so the IDE autocompletes and checks them, and they
don't depend on how well a type checker understands `TypedDict` literals. They are tested with pyright (VS Code /
Pylance) and mypy in the CI:

<!-- no-check: every line is a mistake -->
```python
from esorm import ESModel
from esorm.builder import Bool, Term, Terms, Range, Match, Boosting


class User(ESModel):
    name: str


async def mistakes():
    Term('name')  # Error: missing value
    Term('name', 'x', bost=2.0)  # Error: no parameter named "bost"
    Terms('tag', 'python')  # Error: terms needs a list, not a string
    Range('age', gte=18, lesser=65)  # Error: no parameter named "lesser"
    Match('name', 'x', operator='adn')  # Error: invalid operator
    Boosting(Term('a', 1), Term('b', 2))  # Error: negative_boost is required
    Bool(must=[{'trem': {'name': 'x'}}])  # Error: query dicts are checked too
    Bool(shuld=[Term('a', 1)])  # Error: no parameter named "shuld"
    await User.count('x')  # Error: neither a query dict nor a builder query
```

Notes:

- The search methods have separate overloads for query dicts and builder queries (not a union), because PyCharm
  doesn't check a dict literal against a union of a `TypedDict` and something else. So the query dicts are checked
  as before, and the results are typed (`search()` returns a list of models) with both.
- `to_query()` returns `ESQuery`, the same type the query dicts have. A builder query where a query dict is needed
  (e.g. `query: ESQuery = Bool()`) is reported, use `to_query()` there.
- Field names are strings, so a typo in a field name is not reported, like in query dicts. ES doesn't report it
  either, the query just doesn't match anything.
- To replace a clause list use `q.must = Clauses([...])` (or `clear()` and `extend()`), not a plain list: mypy can't
  handle `+=` on a property whose setter accepts other types than its getter returns, so the setters accept only
  `Clauses` objects.

<a id="builder-tips-and-pitfalls"></a>
## Tips and pitfalls

- **Clauses created for later use inside a block are collected too.** Create them before the block, or give them to a
  query explicitly in the block (then they are moved):

  ```python
  from esorm.builder import Bool, Term, Match

  active = Term('active', True)  # outside of the block: not collected

  with Bool() as q:
      Match('name', 'John')
      q.filter += active
  ```

  ```json
  {"bool": {"must": [{"match": {"name": "John"}}], "filter": [{"term": {"active": true}}]}}
  ```

- **`str` fields are `keyword` fields in ESORM.** `Match` on a keyword field matches only the exact value. Use
  `esorm.fields.text` or `TextField` for full text search, and `Term` / `Terms` for keyword fields.
- **Use `filter` for exact conditions.** Term level queries and ranges usually don't need scoring, in `filter` they
  are faster and cached.
- **`should` clauses next to `must` or `filter` are optional**, they only increase the score. Set
  `minimum_should_match` if at least one of them must match. The `|` operator does it automatically.
- **Nested fields need `Nested`**, see [Joining queries](#builder-joining-queries).
- **Start operator expressions with a builder query** (or `Raw`): `{'match_all': {}} | Term(...)` works at runtime,
  but type checkers don't know the type of the dict literal there.
- **Don't write an operator expression alone in a `with` block**, use `q.must += a | b`, see
  [operators](#combining-queries-with-operators).
- **An empty `Bool()` matches all documents**, which is usually what an empty search form should do. If it shouldn't,
  check the clauses before searching (e.g. `if not q.must and not q.filter: return []`).
