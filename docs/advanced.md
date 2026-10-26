<a id="advanced-usage"></a>
# 🔬 Advanced usage


<a id="optimistic-concurrency-control"></a>
## Optimistic concurrency control

If more processes update the same document at the same time, one of them could overwrite the changes of the
other. ESORM prevents this automatically with optimistic concurrency control.

When you load a document, ESORM stores its `_seq_no` and `_primary_term`. When you save or delete it, ESORM sends
them with the request, and ElasticSearch rejects the operation if the document was changed since it was loaded.
More information in the [ES documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/optimistic-concurrency-control).

This is how to handle the conflicts:

```python
from elasticsearch import ConflictError
from esorm import ESModel


class User(ESModel):
    first_name: str
    last_name: str


async def test_race_condition():
    # Load the same user twice
    user_load1 = await User.get(1)
    user_load2 = await User.get(1)
    # Both have the same _seq_no and _primary_term

    try:
        # Update the 1st loaded user
        user_load1.first_name = "John"
        await user_load1.save()
        # Update the 2nd loaded user
        user_load2.first_name = "Jane"
        await user_load2.save()  # This raises a ConflictError, the document was changed since it was loaded
    except ConflictError:
        print("Conflict error")

    # Load the same user twice again
    user_load1 = await User.get(1)
    user_load2 = await User.get(1)

    # Delete works the same way
    try:
        user_load1.first_name = "John"
        await user_load1.save()
        await user_load2.delete()  # This raises a ConflictError
    except ConflictError:
        print("Conflict error")
```

In bulk operations the conflicts are reported by `BulkError`:

```python
from esorm import ESModel, ESBulk, error


class User(ESModel):
    first_name: str
    last_name: str


async def test_bulk():
    # Load the same user twice
    user_load1 = await User.get(1)
    user_load2 = await User.get(1)
    try:
        async with ESBulk(refresh="wait_for") as bulk:
            user_load1.first_name = "John"
            await bulk.save(user_load1)
            user_load2.first_name = "Jane"
            await bulk.save(user_load2)  # This one fails, the 1st one changed the document
    except error.BulkError as e:
        # The failed operations, with their models
        print("Bulk error:", e.failed_operations)
```

### Retry on conflict

The `retry_on_conflict` decorator runs an async function again if it raises a conflict error:

```python
import asyncio
from esorm import ESModel, retry_on_conflict


class User(ESModel):
    first_name: str
    last_name: str
    logins: int = 0


async def test_retry_on_conflict(user: User):
    @retry_on_conflict(3)  # Retry at most 3 times
    async def login(user_id):
        _user = await User.get(user_id)
        _user.logins += 1
        await _user.save()

    # These run at the same time, but they don't raise ConflictError,
    # because the conflicting ones load the user again and retry
    await asyncio.gather(
        login(user.__id__),
        login(user.__id__),
        login(user.__id__),
    )
```

Without a number it retries until the function succeeds. You can also raise `esorm.error.ConflictError` yourself to
retry the function.

#### Model methods

If the decorated function is a method of a model, the model is reloaded from ElasticSearch before the next try
(`reload_on_conflict=True` is the default):

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
    # This doesn't raise a ConflictError, because the conflicting models are reloaded, and the method is retried
    await asyncio.gather(
        user_1.login(),
        user_2.login(),
        user_3.login(),
    )
```

<a id="lazy-properties"></a>
## Lazy properties

Pydantic is synchronous, so a computed field can't be async, and it can't query other documents. Lazy properties
solve this: they are async functions, which ESORM runs after the models are loaded.

```python
from typing import List
from esorm import ESModel, lazy_property
from pydantic import computed_field


class User(ESModel):
    first_name: str
    last_name: str

    # A classic Pydantic computed field, it can only be synchronous
    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    # A lazy property, it can be async
    @lazy_property
    async def same_first_name(self) -> List["User"]:
        return await User.search_by_fields({'first_name': self.first_name})
```

How they work:

- `@lazy_property` turns the async function into a read-only property, and registers the function in the model.
- After `get()`, `reload()`, `search()`, `scan()` and the other loading methods, ESORM runs the functions of the
  loaded models (and of their sub-models), and stores the results.
- The property returns the stored result, e.g. when the model is serialized to JSON. Before the function runs, it
  returns `None`. For models you create yourself, run them with `await model.calc_lazy_properties()`.

The function must have a return annotation.

The lazy properties of more models are computed concurrently, at most 16 at the same time by default. You can
change this with `set_max_lazy_property_concurrency`:

```python
from esorm.model import set_max_lazy_property_concurrency

set_max_lazy_property_concurrency(10)  # At most 10 at the same time
```

The example above is recursive: `same_first_name` loads `User` models, which have a `same_first_name` property
too, and so on. To prevent an infinite loop, ESORM limits the depth of the recursion, which is 1 by default. Beyond
this depth the lazy properties are not computed, they return `None`, and a warning is logged.

You can change the depth with `lazy_property_max_recursion_depth` in `ESConfig`:

```python
from esorm import ESModel


class User(ESModel):
    class ESConfig:
        lazy_property_max_recursion_depth = 2  # Set the recursion depth to 2

    first_name: str
    last_name: str
    # ...
```

<a id="shard-routing"></a>
## Shard routing

ElasticSearch stores every document in one shard of the index. By default, the shard is chosen by the id of the
document. With a routing value you can choose it yourself: documents with the same routing value are stored in the
same shard. If you search with a routing value, ElasticSearch searches only that shard, instead of all of them.
This is useful e.g. if most queries are for the documents of a single customer or region.

More info in the [ES documentation](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/mapping-routing-field).

In ESORM the routing value of a document is returned by its `__routing__` property:

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

`save()`, `delete()`, `reload()` and the bulk operations use `__routing__` automatically. For `get()`, the search
methods and `scan()` give the routing value in the `routing` parameter, e.g. `await User.get(user_id,
routing='europe_routing')`. Without it, `get()` may look for the document in the wrong shard.

<a id="retrieve-selected-fields-only"></a>
## Retrieve selected fields only

To transfer less data, you can tell ElasticSearch which fields to return, with the `_source` parameter of `get()`,
the search methods and `scan()`. The other fields get their default values, so this works only if the skipped
fields have defaults:

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
    assert doc is not None
    assert doc.f_str == 'b'
    assert doc.f_int == 0

    doc = await Model.search_one_by_fields(dict(_id=doc_id), _source=['f_int'])
    assert doc is not None
    assert doc.f_str == 'a'
    assert doc.f_int == 1

    doc = await Model.get(doc_id, _source=['f_str'])
    assert doc.f_str == 'b'
    assert doc.f_int == 0

    doc = await Model.get(doc_id, _source=['f_int'])
    assert doc.f_str == 'a'
    assert doc.f_int == 1
```

Don't save a model loaded this way, because the skipped fields would be overwritten with the defaults.

<a id="iterating-through-large-result-sets"></a>
## Iterating through large result sets

`search()` with pagination is for showing results to users: it returns a page of results, and it can't go
deeper than 10,000 results by default (`index.max_result_window`), because the deeper the page, the more documents
every shard needs to sort and return. When you need to process *all* results of a query, e.g. to export, migrate
or reindex data, or to run batch jobs, use `scan()` or `scan_batches()`. They can iterate through any number of
documents (even millions) efficiently.

| | `search()` + `Pagination` | `scan()` / `scan_batches()` |
|---|---|---|
| Use case | Pages for users (UI, API) | Processing all results (export, batch jobs) |
| Number of results | Max. 10,000 by default | Unlimited |
| Jump to any page | Yes | No, forward only |
| Consistent results | No, changes between pages are visible | Yes, a snapshot of the index |
| Returns | A list of models | An async iterator of models (or lists of models) |

### Basic usage

`scan()` returns an async iterator, which yields the models one by one, while it fetches them in batches in the
background:

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

Without a query, it returns all documents:

```python
async def count_by_country():
    counts = {}
    async for user in User.scan():
        counts[user.country] = counts.get(user.country, 0) + 1
    return counts
```

`scan_batches()` has the same parameters, but it yields lists of models (one list per request). Use it if you
process the documents in groups, e.g. with bulk operations:

```python
from esorm import ESBulk


async def anonymize_users():
    async for users in User.scan_batches(batch_size=500):
        # `users` is a list of at most 500 models
        async with ESBulk() as bulk:
            for user in users:
                user.name = 'anonymous'
                await bulk.save(user)
```

You can safely modify the documents during the iteration: the iteration doesn't see the changes, so every
document is processed exactly once. Optimistic concurrency control works as usual, so if someone else modifies a
document after it was fetched, saving it raises a conflict error.

### Parameters

The parameters are similar to the ones of `search()`:

| Parameter | Default | Description |
|---|---|---|
| `query` | `None` | ElasticSearch query dict or builder query. If not set, all documents are returned |
| `batch_size` | `1000` | The number of documents fetched in one request |
| `sort` | `None` | Field name or sort term list, as in `search()`. If not set, `ESConfig.default_sort` is used, or the index order |
| `keep_alive` | `'1m'` | How long the point in time is kept alive between two requests |
| `routing` | `None` | Shard routing value |
| `index` | `None` | Index name (or pattern). If not set, the index of the model is used |
| `_source` | `None` | Fields to return, see [Retrieve selected fields only](#retrieve-selected-fields-only) |
| `**kwargs` | | Other search API parameters (e.g. `request_timeout`) |

### How it works

`scan()` uses a [point in time](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-open-point-in-time)
(PIT) with [`search_after`](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/paginate-search-results#search-after),
which is the recommended way of deep pagination since ElasticSearch 7.10:

1. It opens a point in time on the index. It is a lightweight snapshot of the index, so all the following
   requests see the same data, regardless of the changes made in the meantime.
2. It fetches the results in batches. Instead of a page number, every request continues after the sort
   values of the last document of the previous batch (`search_after`), so every shard needs to return only
   `batch_size` documents, no matter how deep the iteration is.
3. When there are no more results, or the iteration is stopped, it closes the point in time.

Lazy properties are computed for every batch, as for the results of `search()`.

### Sorting

Without `sort` (and `ESConfig.default_sort`), the results come in index order, which is the most efficient.
If the order matters, set it as in `search()`. The `Sort` decorator works with `scan()` too
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

`search_after` needs a unique sort order, otherwise the documents with the same sort values at the border of
two batches would be lost. ESORM adds a unique tiebreaker to the sort automatically (`_shard_doc`, or `_id`
on OpenSearch 2.x, which doesn't support `_shard_doc`), so you can sort by any field, e.g. by `age`, which is the
same for many users.

### Stopping early

The point in time is closed automatically when the iteration ends. But if you stop the iteration early
(by `break`, `return` or an exception), Python closes the async iterator only when it is garbage
collected, so the point in time stays open until then (or until `keep_alive` expires). To close it
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

ElasticSearch [Watcher](https://www.elastic.co/docs/explore-analyze/alerting/watcher) runs actions on a
schedule, when a condition is met. The condition usually checks data loaded from ElasticSearch or from an HTTP
service (the watch payload), and the action can e.g. send an email, call a webhook or index a document.

Watcher is not part of the free (Basic) license of ElasticSearch, it needs a paid subscription or a trial license.
OpenSearch doesn't support it.

In ESORM a watch is a subclass of `esorm.watcher.Watcher`, with the parts of the
[watch definition](https://www.elastic.co/docs/explore-analyze/alerting/watcher/how-watcher-works) as class
attributes: `trigger`, `input`, `condition`, `actions` and `metadata`. `DeleteWatcher` is a ready-made watch, which
deletes the documents matching a query. This example deletes the drafts older than 1 hour:

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
            "interval": "30s"
        }
    }

    _index = "esorm_draft"  # The full index name, with the prefix
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

`DeleteWatcher` calls the delete by query API of ElasticSearch with a webhook, by default on
`http://127.0.0.1:9200`, as seen from the ElasticSearch node. You can change this with the `_scheme`, `_host` and
`_port` attributes.

The watches are created by `setup_watchers()`, after `connect()`. It creates (or updates) every defined watch, the
id of a watch is the name of its class:

```python
from esorm.watcher import setup_watchers


async def prepare_es():
    import watchers  # Import your watchers
    await setup_watchers(watchers)
```

For more info see the [API reference](https://esorm.readthedocs.io/en/latest/esorm.html#module-esorm.watcher),
the ElasticSearch documentation, or the source code.


<a id="fastapi-integration"></a>
## FastAPI integration

ESORM models are Pydantic models, so FastAPI can use them as request and response models:

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

`esorm.fastapi` creates FastAPI dependencies for pagination and sorting:

- `make_dep_pagination()` adds `_page` and `_page_size` query parameters to the endpoint, and sets the
  `X-Total-Hits` response header to the total number of hits, so the frontend can show the number of pages.
  The page size is limited to 10,000 by default, you can change it with `set_max_page_size()`.
- `make_dep_sort()` adds a `_sort` query parameter, whose values are the names of the given sort options. It is an
  enum, so the options can be selected in the Swagger UI.

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
    # This creates the _page and _page_size query parameters of the endpoint
    pagination: Pagination = Depends(make_dep_pagination(default_page=1, default_page_size=10)),
    # This creates the _sort query parameter of the endpoint, its values are the names of the options
    sort: Sort = Depends(make_dep_sort(
        first_name_last_name_asc=  # The name of the 1st sort option
        # The definition of the sort option
        [
            {'first_name': {"order": "asc"}},
            {'last_name': {"order": "asc"}},
        ],
        last_name_first_name_asc=  # The name of the 2nd sort option
        # The definition of the sort option
        [
            {'last_name': {"order": "asc"}},
            {'first_name': {"order": "asc"}},
        ]
    )),
) -> List[User]:
    """ Get all users """
    return await sort(pagination(User)).all()
```

If the frontend is on another domain, add `X-Total-Hits` to the `expose_headers` of the CORS middleware, otherwise
the browser hides it from the frontend code.

<a id="fastapi-streaming-export"></a>
### FastAPI streaming export

With `scan()` an endpoint can stream any number of documents, without loading all of them into memory, e.g. as
[NDJSON](https://github.com/ndjson/ndjson-spec) (one JSON document per line):

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
