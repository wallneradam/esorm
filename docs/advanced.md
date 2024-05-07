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
    import asyncio
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
        return await self.search_by_fields(first_name=self.first_name)
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
    return await User.search_by_fields(region=region, routing=f"{region}_routing")  
```

<a id="watchers"></a>
## Watchers

You can add watches to automatically perform an action when certain conditions are met. The conditions 
are generally based on data you’ve loaded into the watch, also known as the Watch Payload. This 
payload can be loaded from different sources - from Elasticsearch, an external HTTP service, or even 
a combination of the two.

More info: https://www.elastic.co/guide/en/elasticsearch/reference/current/how-watcher-works.html

<small>
The watcher feature is not free in Elasticsearch, you need to have a license for it. Or if your are 
an experienced developer, you can compile Elasticsearch from source and disable the license check.
You can do it for your own use, because the source code is available, though it is not free.
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
    return await User.search_by_fields(first_name=first_name, last_name=last_name)
```

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