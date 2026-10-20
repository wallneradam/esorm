<a id="advanced-usage"></a>
# 🔬 Advanced usage


<a id="optimistic-concurrency-control"></a>
## Optimistic concurrency control

ESORM uses optimistic concurrency control automatically, to prevent race conditions and data losses 
when multiple updates are made to the same document at the same time. 

When you save a document, ESORM checks if the document has been changed since it was loaded and 
raises an exception if it has. In the background it uses the `_seq_no` and `_primary_term` fields
for checking. More information in [ES documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/optimistic-concurrency-control.html).

This is how to handle race conditions:
```python
from esorm import ESModel


class User(ESModel):
    first_name: str
    last_name: str

    
async def test_race_condition():
    """ Update the user """
    from elasticsearch import ConflictError
    
    # Load the user
    user_load1 = await User.get(id = 1)
    # Load again
    user_load2 = await User.get(id = 1)
    # Here both have the same _seq_no and _primary_term
    
    try:
        # Update the 1st loaded user
        user_load1.first_name = "John"
        await user_load1.save()
        # Update the 2nd loaded user
        user_load2.first_name = "Jane"
        await user_load2.save()   # This will raise a ConflictError         
    except ConflictError:
        print("Conflict error")    
        
    # Load the user
    user_load1 = await User.get(id = 1)
    # Load again
    user_load2 = await User.get(id = 1)    
    # Here both have the same _seq_no and _primary_term
    
    # Delete operation
    try:
        user_load1.first_name = "John"
        await user_load1.save()
        await user_load2.delete()  # This will raise a ConflictError
    except ConflictError:
        print("Conflict error")
```

This is how to use it in bulk operations:
```python
from esorm import ESModel, ESBulk, error


class User(ESModel):
    first_name: str
    last_name: str


async def test_bulk():
    # Load the user
    user_load1 = await User.get(id = 1)
    # Load again
    user_load2 = await User.get(id = 1)    
    # Here both have the same _seq_no and _primary_term
    try:
        async with ESBulk(wait_for=True) as bulk:
            user_load1.first_name = "John"
            await bulk.save(user_load1)
            user_load2.first_name = "Jane"
            await bulk.save(user_load2)  #
    except error.BulkError as e:
        # You can get the failed operations from e.failed_operations
        print("Bulk error:", e.failed_operations)
```

### Retry on conflict

You can use the `retry_on_conflict` decorator to automatically retry the operation(s) on conflict:
```python
import asyncio
from esorm import ESModel, retry_on_conflict


class User(ESModel):
    first_name: str
    last_name: str
    logins: int = 0


async def test_retry_on_conflict(user: User):    
    @retry_on_conflict(3)  # Retry 3 times on conflict
    async def login(user_id):
        _user = await User.get(id=user_id)
        _user.logins += 1
    
    # This won't raise a ConflictError   
    await asyncio.gather(
        login(user.__id__),
        login(user.__id__),
        login(user.__id__),
    )
```

#### If it is a model method

If you retry a model method, it will reload the model automaticaly by default before calling the method again:
```python
import asyncio
from esorm import ESModel, retry_on_conflict


class User(ESModel):
    first_name: str
    last_name: str
    logins: int = 0
    
    @retry_on_conflict(3, reload_on_conflict=True)  # reload_on_conflict is True by default 
    async def login(self):
        self.logins += 1
        await self.save()
        
async def test_retry_on_conflict(user: User):
    user_1 = await User.get(user.__id__)
    user_2 = await User.get(user.__id__)
    user_3 = await User.get(user.__id__)
    # This won't raise a ConflictError, becaue the model reloads and retries the method on conflict   
    await asyncio.gather(
        user_1.login(),
        user_2.login(),
        user_3.login(),
    )
```

<a id="lazy-properties"></a>
## Lazy properties

ESORM is based on [pydantic](https://pydantic-docs.helpmanual.io/) which is fully synchronous, so
it's not possible to use async functions to calculate property values. Because of this, you also 
can't query another model in a computed field. To solve this problem, ESORM provides lazy properties.

You can create lazy properties like this:
```python
from typing import List
from esorm import ESModel, lazy_property
from pydantic import computed_field

class User(ESModel):
    first_name: str
    last_name: str

    # This is classic pydantic computed field, which can be only synchronous
    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
        
    # This is lazy property, which can be async
    @lazy_property
    async def same_first_name(self) -> List["User"]:
        return await User.search_by_fields({'first_name': self.first_name})
```

Lazy properties in the background work like the following: 
- they are registered in the model by storing the async function
- replace it to a real property on model creation
- after a query is executed (which is always async), lazy properties are calculated and stored in the model
- these stored values are used when accessing the property, e.g, when they are serialized to JSON

Lazy properties are computed parallelly, you can configure the number of parallel query tasks by 
`set_max_lazy_property_concurrency` function:

```python
from esorm.model import set_max_lazy_property_concurrency

set_max_lazy_property_concurrency(10)  # Set the number of parallel tasks to 10
```

The above example is recursive, because `User` model is used in the `same_first_name` property. This
could lead to an infinite loop. Because of this, ESORM restricts the depth of recursion to 1 by default.

If you want to change the recursion depth, you can do it by setting the `max_lazy_property_depth` in 
the `ESConfig`:

```python
from esorm import ESModel


class User(ESModel):
    class ESConfig:
        lazy_property_max_recursion_depth = 2  # Set the recursion depth to 2
   
    first_name: str
    last_name: str
    ...
```

<a id="shard-routing"></a>
## Shard routing

Shard routing is a feature of Elasticsearch which allows you to store documents in a specific shard.
This can be useful if you want to store documents of a specific type in a specific shard, e.g, you 
want to store all documents of a specific region. When using shard routing, ElasticSearch does not 
need to search all shards, but only the shards which contain the documents you are looking for.

More info: https://www.elastic.co/guide/en/elasticsearch/reference/current/search-shard-routing.html

In ESORM shard routing looks like this:

```python
from typing import List
from esorm import ESModel

class User(ESModel):
    first_name: str
    last_name: str
    region: str
    
    @property
    def __routing__(self) -> str:
        """ Return the routing value for this document """
        return self.region + '_routing'  # Calculate the routing value from the region field
        
async def get_user_by_region(region: str = 'europe') -> List[User]:
    """ Search for users by region using shard routing """
    return await User.search_by_fields({'region': region}, routing=f"{region}_routing")
```

<a id="retreive-selected-fields-only"></a>
## Retreive Selected Fields Only

When you query documents from Elasticsearch, you can specify which fields you want to get back.
For this you can use teh `_source` argument in `search` methods and `get` method.
This will only work if you specify default values for all the fields you may skip:

```python
import esorm

class Model(esorm.ESModel):
    f_int: int = 0
    f_str: str = 'a'


async def test_source():    
    doc = Model(f_int=1, f_str='b')
    doc_id = await doc.save()

    doc = await Model.get(doc_id)
    assert doc.f_str == 'b'
    assert doc.f_int == 1

    doc = await Model.search_one_by_fields(dict(_id=doc_id), _source=['f_str'])
    assert doc.f_str == 'b'
    assert doc.f_int == 0

    doc = await Model.search_one_by_fields(dict(_id=doc_id), _source=['f_int'])
    assert doc.f_str == 'a'
    assert doc.f_int == 1

    doc = await Model.get(doc_id, _source=['f_str'])
    assert doc.f_str == 'b'
    assert doc.f_int == 0

    doc = await Model.get(doc_id, _source=['f_int'])
    assert doc.f_str == 'a'
    assert doc.f_int == 1
```

<a id="iterating-through-large-result-sets"></a>
## Iterating through large result sets

`search()` with pagination is made for showing results to users: it returns a page of results, and
it can't go deeper than 10,000 results by default (`index.max_result_window`), because the deeper the
page, the more documents every shard needs to sort and return. When you need to process *all* results
of a query, e.g. to export, migrate or reindex data, or to run batch jobs, use `scan()` or
`scan_batches()`. They can iterate through any number of documents (even millions) efficiently.

| | `search()` + `Pagination` | `scan()` / `scan_batches()` |
|---|---|---|
| Use case | Pages for users (UI, API) | Processing all results (export, batch jobs) |
| Number of results | Max. 10,000 by default | Unlimited |
| Jump to any page | Yes | No, forward only |
| Consistent results | No, changes between pages are visible | Yes, a snapshot of the index |
| Returns | A list of models | An async iterator of models (or lists of models) |

### Basic usage

`scan()` returns an async iterator, which yields the models one by one, while in the background it
fetches them in batches:

```python
from esorm import ESModel


class User(ESModel):
    name: str
    age: int
    country: str


async def export_adults():
    async for user in User.scan({'range': {'age': {'gte': 18}}}):
        print(user.name)
```

Without query, all documents are returned:

```python
async def count_by_country():
    counts = {}
    async for user in User.scan():
        counts[user.country] = counts.get(user.country, 0) + 1
    return counts
```

`scan_batches()` takes the same arguments, but it yields lists of models (one list per request).
It is useful if you want to process the documents in groups, e.g. with bulk operations:

```python
from esorm import ESBulk


async def anonymize_users():
    async for users in User.scan_batches(batch_size=500):
        # Here `users` is a list of maximum 500 models
        async with ESBulk() as bulk:
            for user in users:
                user.name = 'anonymous'
                await bulk.save(user)
```

Modifying the documents during the iteration is safe: the iteration doesn't see the changes, so every
document is processed exactly once. Optimistic concurrency control works as usual, so if a document is
modified by someone else after it was fetched, saving it raises a conflict error.

### Parameters

The parameters are similar to the ones of `search()`:

| Parameter | Default | Description |
|---|---|---|
| `query` | `None` | ElasticSearch query dict, if not set, all documents are returned |
| `batch_size` | `1000` | The number of documents fetched in one request |
| `sort` | `None` | Field name or sort term list like in `search()`, if not set, `ESConfig.default_sort` is used, or the index order |
| `keep_alive` | `'1m'` | How long the point in time is kept alive between two requests |
| `routing` | `None` | Shard routing value |
| `index` | `None` | Index name (or pattern), if not set, the index of the model is used |
| `_source` | `None` | Fields to return, see [Retreive Selected Fields Only](#retreive-selected-fields-only) |
| `**kwargs` | | Other search API parameters (e.g. `request_timeout`) |

### How it works

`scan()` uses a [point in time](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-open-point-in-time)
(PIT) with [`search_after`](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/paginate-search-results#search-after),
which is the recommended way of deep pagination since ElasticSearch 7.10:

1. It opens a point in time on the index. It is a lightweight snapshot of the index, so all the following
   requests see the same data, regardless of changes made in the meantime.
2. It fetches the results in batches. Instead of a page number, every request continues after the sort
   values of the last document of the previous batch (`search_after`), so every shard only needs to return
   `batch_size` documents, no matter how deep the iteration is.
3. When there are no more results, or the iteration is stopped, it closes the point in time.

Lazy properties are computed for every batch, like for the results of `search()`.

### Sorting

Without `sort` (and `ESConfig.default_sort`), the results come in index order, which is the most efficient.
If the order matters, set it like in `search()`. The `Sort` decorator works with `scan()` too
(`Pagination` is ignored, because there are no pages):

```python
from esorm import Sort


async def users_by_age():
    async for user in User.scan(sort=[{'age': {'order': 'desc'}}]):
        ...

    sort = Sort(sort='name')
    async for user in sort(User).scan():
        ...
```

`search_after` needs a unique sort order, otherwise documents with the same sort values at the border of
two batches would be lost. ESORM adds a unique tiebreaker to the sort automatically (`_shard_doc`, or `_id`
on OpenSearch 2.x, which doesn't support `_shard_doc`), so it is safe to sort by any field, e.g. by `age`,
which is the same for many users.

### Stopping early

The point in time is closed automatically when the iteration ends. But if you stop the iteration early
(by `break`, `return` or an exception), Python closes the async iterator only when it is garbage
collected, so the point in time would remain open until then (or until `keep_alive` expires). To close it
immediately, use `contextlib.aclosing()`:

```python
from contextlib import aclosing


async def find_first_admin():
    async with aclosing(User.scan()) as users:
        async for user in users:
            if user.name == 'admin':
                return user  # The point in time is closed here
```

### Keep alive and resources

`keep_alive` is not the time limit of the whole iteration: it is how long the point in time is kept alive
*between two requests*, and every request extends it. The default `'1m'` is enough if processing a batch
takes less than a minute. If it takes longer (e.g. you call an external service for every document),
increase it, or decrease `batch_size`:

```python
async for users in User.scan_batches(batch_size=100, keep_alive='5m'):
    await send_newsletter(users)  # This may be slow
```

An open point in time prevents ElasticSearch from deleting old index segments, which costs disk space and
memory. So don't make `keep_alive` longer than necessary, and stop the iteration properly.

### Performance tips

- Don't set `sort` if you don't need a specific order, the index order is the fastest.
- Use `_source` to fetch only the fields you need, it can reduce the transferred data a lot.
- The default `batch_size` (1000) is good in most cases. Bigger batches mean fewer requests, but more
  memory, and every batch must be processed within `keep_alive`.
- The total number of hits is not counted (`track_total_hits=False`), because it would make every request
  slower. If you need it, use `count()` with the same query.

### OpenSearch

`scan()` works the same way on OpenSearch (2.x and 3.x), ESORM converts the point in time API calls
(`create_pit`, `delete_pit`) automatically.

<a id="watchers"></a>
## Watchers

You can add watches to automatically perform an action when certain conditions are met. The conditions 
are generally based on data you’ve loaded into the watch, also known as the Watch Payload. This 
payload can be loaded from different sources - from Elasticsearch, an external HTTP service, or even 
a combination of the two.

More info: https://www.elastic.co/guide/en/elasticsearch/reference/current/how-watcher-works.html

<small>
The watcher feature is not free in Elasticsearch, you need to have a license for it. Or if you are 
an experienced developer, you can compile Elasticsearch from source and disable the license check.
You can do it for your own use, because the source code is available, though it is not free.
(hint: you need to recompile the `x-pack` plugin only, and you can replace it in the compiled version)
</small>

The following example shows how to create a watcher which deletes all draft documents older than 1 hour:
```python
from esorm.watcher import DeleteWatcher
from esorm import query

TIMEOUT = 60 * 60  # 1 hour


class PurgeDraft(DeleteWatcher):
    """
    Purge draft data after TIMEOUT
    """
    trigger = {
        "schedule": {
            "interval": f"30s"
        }
    }

    _index = f"draft"
    _query: query.ESQuery = {
        "bool": {
            "must": [
                # Search for all documents with id starting with "_"
                {
                    "wildcard": {
                        "id": {
                            "value": "_*"
                        }
                    }
                },
                # Filter documents which are older than TIMEOUT + 30s
                {
                    "range": {
                        "created_at": {
                            "lt": f"now-{TIMEOUT + 30}s",  # 30 sec buffer
                        }
                    }
                }
            ]
        }
    }
```

For more info check the [reference](https://esorm.readthedocs.io/en/latest/esorm.html#module-esorm.watcher),
or ElasticSearch documentation, or the source code.


<a id="fastapi-integration"></a>
## FastAPI integration

Because ESORM is based on pydantic, it can be easily integrated with FastAPI:

```python
from typing import List, Optional
from esorm import ESModelTimestamp
from fastapi import FastAPI


class User(ESModelTimestamp):
    """ The User model """
    first_name: str
    last_name: str
    

app = FastAPI()


@app.post("/users")
async def create_user(first_name: str, last_name: str) -> User:
    """ Create a new user """
    user = User(first_name=first_name, last_name=last_name)
    await user.save()
    return user


@app.get("/users")
async def users(first_name: Optional[str] = None, last_name: Optional[str] = None) -> List[User]:
    """ Search users """
    fields = {'first_name': first_name, 'last_name': last_name}
    return await User.search_by_fields({k: v for k, v in fields.items() if v is not None})
```

<a id="fastapi-pagination"></a>
### FastAPI pagination

You can add pagination and sort parameters as a dependency in endpoint arguments:
The pagination dependency set the `X-Total-Hits` header in the response to the total number of hits.
So in your frontend, you can get the total number of hits from this header.

```python 
from typing import List
from esorm import ESModelTimestamp, Pagination, Sort
from fastapi import FastAPI, Depends

from esorm.fastapi import make_dep_sort, make_dep_pagination


class User(ESModelTimestamp):
    """ The User model """
    first_name: str
    last_name: str
    

app = FastAPI()

@app.get("/all_users")
async def all_users(
    # This will create a _page, and a _page_size query parameter for the endpoint 
    pagination: Pagination = Depends(make_dep_pagination(default_page=1, default_page_size=10)),
    # This will create a _sort enum query parameter for the endpoint, so it is selectable in swagger UI
    sort: Sort = Depends(make_dep_sort( 
        first_name_last_name_asc=  # This is the name of the 1st sort option
        # Definition of the sort options
        [  
            {'first_name': {"order": "asc"}},
            {'last_name': {"order": "asc"}},
        ],
        last_name_first_name_asc=  # This is the name of the 2nd sort option
        # Definition of the sort options
        [              
            {'last_name': {"order": "asc"}},
            {'first_name': {"order": "asc"}},
        ]
    )),
) -> List[User]:
    """ Get all users """
    return await sort(pagination(User)).all()
```

<a id="fastapi-streaming-export"></a>
### FastAPI streaming export

`scan()` can be used to stream any number of documents from an endpoint, without loading all of them
into memory. E.g. as [NDJSON](https://github.com/ndjson/ndjson-spec) (one JSON document per line):

```python
from contextlib import aclosing
from typing import AsyncIterator
from esorm import ESModelTimestamp
from fastapi import FastAPI
from fastapi.responses import StreamingResponse


class User(ESModelTimestamp):
    """ The User model """
    first_name: str
    last_name: str


app = FastAPI()


@app.get("/users/export")
async def export_users() -> StreamingResponse:
    """ Export all users as NDJSON """

    async def generate() -> AsyncIterator[str]:
        # If the client disconnects, the point in time is closed immediately
        async with aclosing(User.scan(sort='last_name')) as users:
            async for user in users:
                yield user.model_dump_json() + '\n'

    return StreamingResponse(generate(), media_type='application/x-ndjson')
```