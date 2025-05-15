<img src="https://raw.githubusercontent.com/wallneradam/esorm/main/docs/_static/img/esorm.svg" width="110" height="110" align="left" style="margin-right: 1em; margin-bottom: 0.5em" alt="Logo"/>

# ESORM - Python ElasticSearch ORM based on Pydantic

ESORM is an ElasticSearch Object Relational Mapper or Object Document Mapper (ODM) if you like,
 for Python based on Pydantic. It is a high-level library for managing ElasticSearch documents
 in Python. It is fully async and uses annotations and type hints for type checking and IDE autocompletion.

## ☰ Table of Contents

- [💾 Installation](#installation)
- [🚀 Features](#features)
    - [Supported ElasticSearch versions](#supported-elasticsearch-versions)
    - [Supported Python versions](#supported-python-versions)
- [📖 Usage](#usage)
    - [Define a model](#define-a-model)
        - [Python basic types](#python-basic-types)
        - [ESORM field types](#esorm-field-types)
        - [Field Definition](#field-definition)
        - [geo_point](#geo_point)
        - [dense_vector](#dense_vector)
        - [Nested documents](#nested-documents)
        - [List primitive fields](#list-primitive-fields)
        - [ESBaseModel](#esbasemodel)
        - [Id field](#id-field)
        - [Model Settings](#model-settings)
        - [Describe fields](#describe-fields)
        - [ESModelTimestamp](#esmodeltimestamp)
    - [Connecting to ElasticSearch](#connecting-to-elasticsearch)
        - [Client](#client)
    - [Create index templates](#create-index-templates)
    - [Create indices and mappings](#create-indices-and-mappings)
    - [Model instances](#model-instances)
    - [CRUD: Create](#crud-create)
    - [CRUD: Read](#crud-read)
    - [CRUD: Update](#crud-update)
    - [CRUD: Delete](#crud-delete)
    - [Bulk operations](#bulk-operations)
    - [Search](#search)
        - [General search](#general-search)
        - [Search with field value terms (dictionary search)](#search-with-field-value-terms-dictionary-search)
        - [Vector Search](#vector-search)
    - [Aggregations](#aggregations)
    - [Pagination and sorting](#pagination-and-sorting)
- [🔬 Advanced usage](docs/advanced.md#advanced-usage)
    - [Optimistic concurrency control](docs/advanced.md#optimistic-concurrency-control)
    - [Lazy properties](docs/advanced.md#lazy-properties)
    - [Shard routing](docs/advanced.md#shard-routing)
    - [Retreive Selected Fields Only](docs/advanced.md#retreive-selected-fields-only)
    - [Watchers](docs/advanced.md#watchers)
    - [FastAPI integration](docs/advanced.md#fastapi-integration)
- [🖥 IDE Support](#ide-support)
- [🧪 Testing](#testing)
- [🛡 License](#license)
- [📃 Citation](#citation)

<a id="installation"></a>
## 💾 Installation


```bash
pip install pyesorm
```

<a id="features"></a>
## 🚀 Features

- Pydantic model representation of ElasticSearch documents
- Automatic mapping and index creation
- CRUD operations
- Full async support (no sync version at all)
- Mapping to and from ElasticSearch types
- Support for nested documents
- Automatic optimistic concurrency control
- Custom id field
- Context for bulk operations
- Supported IDE autocompletion and type checking (PyCharm tested)
- Everything in the source code is documented and annotated
- `TypedDict`s for ElasticSearch queries and aggregations
- Docstring support for fields
- Shard routing support
- Lazy properties
- Support >= Python 3.8 (tested with 3.8 through 3.13)
- Support for ElasticSearch 9.x, 8.x and 7.x
- Watcher support (You may need ElasticSearch subscription license for this)
- Pagination and sorting
- FastAPI integration

Not all ElasticSearch features are supported yet, pull requests are welcome.

<a id="supported-elasticsearch-versions"></a>
### Supported ElasticSearch versions

It is tested with ElasticSearch 7.x, 8.x and 9.x.

<a id="supported-python-versions"></a>
### Supported Python versions

Tested with Python 3.8 through 3.13.

<a id="usage"></a>
## 📖 Usage

<a id="define-a-model"></a>
### Define a model

You can use all [Pydantic](https://pydantic-docs.helpmanual.io/usage/models/) model features, because `ESModel` is a subclass of `pydantic.BaseModel`.
(Actually it is a subclass of `ESBaseModel`, see more [below...](#esbasemodel))

`ESModel` extends pydantic `BaseModel` with ElasticSearch specific features. It serializes and deserializes
documents to and from ElasticSearch types and handle ElasticSearch operations in the background.

<a id="python-basic-types"></a>
#### Python basic types

```python
from esorm import ESModel


class User(ESModel):
    name: str
    age: int
```

This is how the python types are converted to ES types:

| Python type         | ES type   | Comment                     |
|---------------------|-----------|-----------------------------|
| `str`               | `text`    |                             |
| `int`               | `long`    |                             |
| `float`             | `double`  |                             |
| `bool`              | `boolean` |                             |
| `datetime.datetime` | `date`    |                             |
| `datetime.date`     | `date`    |                             |
| `datetime.time`     | `date`    | Stored as 1970-01-01 + time |
| `typing.Literal`    | `keyword` |                             |
| `UUID`              | `keyword` |                             |
| `Path`              | `keyword` |                             |
| `IntEnum`           | `integer` |                             |
| `Enum`              | `keyword` | also StrEnum                |

Some special pydanctic types are also supported:

| Pydantic type   | ES type   | Comment |
|-----------------|-----------|---------|
| `URL`           | `keyword` |         |
| `IPvAddressAny` | `ip`      |         |


<a id="esorm-field-types"></a>
#### ESORM field types

You can specify ElasticSearch special fields using `esorm.fields` module.

```python
from esorm import ESModel
from esorm.fields import keyword, text, byte, geo_point, dense_vector


class User(ESModel):
    name: text
    email: keyword
    age: byte
    location: geo_point
    embedding: dense_vector
    ...
```

The supported fields are:

| Field name                  | ES type         |
|-----------------------------|-----------------|
| `keyword`                   | `keyword`       |
| `text`                      | `text`          |
| `binary`                    | `binary`        |
| `byte`                      | `byte`          |
| `short`                     | `short`         |
| `integer` or `int32`        | `integer`       |
| `long` or `int64`           | `long`          |
| `unsigned_long` or `uint64` | `unsigned_long` |
| `float16` or `half_float`   | `half_float`    |
| `float32`                   | `float`         |
| `double`                    | `double`        |
| `boolean`                   | `boolean`       |
| `geo_point`                 | `geo_point`     |
| `dense_vector`              | `dense_vector`  |

The `binary` field accepts **base64** encoded strings. However, if you provide `bytes` to it, they
will be automatically converted to a **base64** string during serialization. When you retrieve the
field, it will always be a **base64** encoded string. You can easily convert it back to bytes using
the `bytes()` method: `binary_field.bytes()`.

You can also use `Annotated` types to specify the ES type, like Pydantic `PositiveInt` and
`NegativeInt` and similar.

<a id="field-definition"></a>
#### Field Definition

ESORM fields can be defined with specialized field definition functions for more control:

```python
from esorm.fields import Field, TextField, NumericField, DenseVectorField

class Product(ESModel):
    id: str
    name: str = TextField(..., min_length=3, max_length=100)
    price: float = NumericField(..., gt=0)
    is_available: bool = Field(True)
    location: geo_point
    embedding: dense_vector = DenseVectorField(..., dims=384, similarity="cosine")
```

<a id="geo_point"></a>
#### geo_point

You can use geo_point field type for location data:

```python
from esorm import ESModel
from esorm.fields import geo_point


class Place(ESModel):
    name: str
    location: geo_point


def create_place():
    place = Place(name='Budapest', location=geo_point(lat=47.4979, long=19.0402))
    place.save()
```

<a id="dense_vector"></a>
#### dense_vector

The `dense_vector` field type enables vector similarity search capabilities in Elasticsearch (available in ES 8.x):

```python
from esorm import ESModel
from esorm.fields import dense_vector, DenseVectorField

class Document(ESModel):
    id: str
    content: str
    embedding: dense_vector = DenseVectorField(
        ...,  # required field
        dims=384,  # dimension of the vector
        similarity="cosine"  # similarity metric: 'cosine', 'dot_product', or 'l2'
    )
```

<a id="nested-documents"></a>
#### Nested documents

```python
from esorm import ESModel
from esorm.fields import keyword, text, byte


class User(ESModel):
    name: text
    email: keyword
    age: byte = 18


class Post(ESModel):
    title: text
    content: text
    writer: User  # User is a nested document
```

<a id="list-primitive-fields"></a>
#### List primitive fields

You can use list of primitive fields:

```python
from typing import List
from esorm import ESModel


class User(ESModel):
    emails: List[str]
    favorite_ids: List[int]
    ...
```

<a id="esbasemodel"></a>
#### ESBaseModel

`ESBaseModel` is the base of `ESModel`.

##### Use it for abstract models

```python
from esorm import ESModel, ESBaseModel
from esorm.fields import keyword, text, byte


# This way `User` model won't be in the index
class BaseUser(ESBaseModel):  # <---------------
    # This config will be inherited by User
    class ESConfig:
        id_field = 'email'

    name: text
    email: keyword


# This will be in the index because it is a subclass of ESModel
class UserExtended(BaseUser, ESModel):
    age: byte = 18


async def create_user():
    user = UserExtended(
        name='John Doe',
        email="john@example.com",
        age=25
    )
    await user.save()
```

##### Use it for nested documents

It is useful to use it for nested documents, because by using it will not be included in the
ElasticSearch index.

```python
from esorm import ESModel, ESBaseModel
from esorm.fields import keyword, text, byte


# This way `User` model won't be in the index
class User(ESBaseModel):  # <---------------
    name: text
    email: keyword
    age: byte = 18


class Post(ESModel):
    title: text
    content: text
    writer: User  # User is a nested document

```

<a id="id-field"></a>
#### Id field

You can specify id field in [model settings](#model-settings):

```python
from esorm import ESModel
from esorm.fields import keyword, text, byte


class User(ESModel):
    class ESConfig:
        id_field = 'email'

    name: text
    email: keyword
    age: byte = 18
```

This way the field specified in `id_field` will be removed from the document and used as the document `_id` in the
index.

If you specify a field named `id` in your model, it will be used as the document `_id` in the index
(it will automatically override the `id_field` setting):

```python
from esorm import ESModel


class User(ESModel):
    id: int  # This will be used as the document _id in the index
    name: str
```

You can also create an `__id__` property in your model to return a custom id:

```python
from esorm import ESModel
from esorm.fields import keyword, text, byte


class User(ESModel):
    name: text
    email: keyword
    age: byte = 18

    @property
    def __id__(self) -> str:
        return self.email
```

NOTE: annotation of `__id__` method is important, and it must be declared as a property.

<a id="model-settings"></a>
#### Model Settings

You can specify model settings using `ESConfig` child class.

```python
from typing import Optional, List, Dict, Any
from esorm import ESModel


class User(ESModel):
    class ESConfig:
        """ ESModel Config """
        # The index name
        index: Optional[str] = None
        # The name of the 'id' field
        id_field: Optional[str] = None
        # Default sort
        default_sort: Optional[List[Dict[str, Dict[str, str]]]] = None
        # ElasticSearch index settings (https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules.html)
        settings: Optional[Dict[str, Any]] = None
        # Maximum recursion depth of lazy properties
        lazy_property_max_recursion_depth: int = 1
```

<a id="esmodeltimestamp"></a>
#### ESModelTimestamp

You can use `ESModelTimestamp` class to add `created_at` and `updated_at` fields to your model:

```python
from esorm import ESModelTimestamp


class User(ESModelTimestamp):
    name: str
    age: int
```

These fields will be automatically updated to the actual `datetime` when you create or update a document.
The `created_at` field will be set only when you create a document. The `updated_at` field will be set
when you create or update a document.

<a id="describe-fields"></a>
#### Describe fields

You can use the usual `Pydantic` field description, but you can also use docstrings like this:

```python
from esorm import ESModel
from esorm.fields import TextField


class User(ESModel):
    name: str = 'John Doe'
    """ The name of the user """
    age: int = 18
    """ The age of the user """

    # This is the usual Pydantic way, but I think docstrings are more intuitive and readable
    address: str = TextField(description="The address of the user")
```

The documentation is usseful if you create an API and you want to generate documentation from the model.
It can be used in [FastAPI](https://fastapi.tiangolo.com/) for example.

<a id="aliases"></a>
### Aliases

You can specify aliases for fields:

```python
from esorm import ESModel
from esorm.fields import keyword, Field


class User(ESModel):
    full_name: keyword = Field(alias='fullName')  # In ES `fullName` will be the field name
```

This is good for renaming fields in the model without changing the ElasticSearch field name.

<a id="connecting-to-elasticsearch"></a>
### Connecting to ElasticSearch

You can connect with a simple connection string:

```python
from esorm import connect


async def es_init():
    await connect('localhost:9200')
```

Also you can connect to multiple hosts if you have a cluster:

```python
from esorm import connect


async def es_init():
    await connect(['localhost:9200', 'localhost:9201'])
```

You can wait for node or cluster to be ready (recommended):

```python
from esorm import connect


async def es_init():
    await connect('localhost:9200', wait=True)
```

This will ping the node in 2 seconds intervals until it is ready. It can be a long time.

You can pass any arguments that `AsyncElasticsearch` supports:

```python
from esorm import connect


async def es_init():
    await connect('localhost:9200', wait=True, sniff_on_start=True, sniff_on_connection_fail=True)
```

<a id="client"></a>
#### Client

The `connect` function is a wrapper for the `AsyncElasticsearch` constructor. It creates and stores
a global instance of a proxy to an `AsyncElasticsearch` instance. The model operations will use this
instance to communicate with ElasticSearch. You can retrieve the proxy client instance and you can
use the same way as `AsyncElasticsearch` instance:

```python
from esorm import es


async def es_init():
    await es.ping()
```

<a id="create-index-templates"></a>
### Create index templates

You can create index templates easily:

```python
from esorm import model as esorm_model


# Create index template
async def prepare_es():
    await esorm_model.create_index_template('default_template',
                                            prefix_name='esorm_',
                                            shards=3,
                                            auto_expand_replicas='1-5')
```

Here this will be applied all `esorm_` prefixed (default) indices.

All indices created by ESORM have a prefix, which you can modify globally if you want:

```python
from esorm.model import set_default_index_prefix

set_default_index_prefix('custom_prefix_')
```

The default prefix is `esorm_`.

<a id="create-indices-and-mappings"></a>
### Create indices and mappings

You can create indices and mappings automatically from your models:

```python
from esorm import setup_mappings


# Create indices and mappings
async def prepare_es():
    import models  # Import your models
    # Here models argument is not needed, but you can pass it to prevent unused import warning
    await setup_mappings(models)
```

First you must create (import) all model classes. Model classes will be registered into a global registry.
Then you can call `setup_mappings` function to create indices and mappings for all registered models.

**IMPORTANT:** This method will ignore mapping errors if you already have an index with the same name. It can update the
indices
by new fields, but cannot modify or delete fields! For that you need to reindex your ES database. It is an ElasticSearch
limitation.

<a id="model-instances"></a>
### Model instances

When you get a model instance from elasticsearch by `search` or `get` methods, you will get the following private
attributes filled automatically:

| Attribute       | Description                         |
|-----------------|-------------------------------------|
| `_id`           | The ES id of the document           |
| `_routing`      | The routing value of the document   |
| `_version`      | Version of the document             |
| `_primary_term` | The primary term of the document    |
| `_seq_no`       | The sequence number of the document |

<a id="crud-create"></a>
### CRUD: Create

```python
from esorm import ESModel


# Here the model have automatically generated id
class User(ESModel):
    name: str
    age: int


async def create_user():
    # Create a new user
    user = User(name='John Doe', age=25)
    # Save the user to ElasticSearch
    new_user_id = await user.save()
    print(new_user_id)
```

<a id="crud-read"></a>
### CRUD: Read

```python
from esorm import ESModel


# Here the model have automatically generated id
class User(ESModel):
    name: str
    age: int


async def get_user(user_id: str):
    user = await User.get(user_id)
    print(user.name)
```

<a id="crud-update"></a>
### CRUD: Update

On update race conditions are checked automatically (with the help of _primary_term and _seq_no fields).
This way an optimistic locking mechanism is implemented.

```python
from esorm import ESModel


# Here the model have automatically generated id
class User(ESModel):
    name: str
    age: int


async def update_user(user_id: str):
    user = await User.get(user_id)
    user.name = 'Jane Doe'
    await user.save()
```

<a id="crud-delete"></a>
### CRUD: Delete

```python
from esorm import ESModel


# Here the model have automatically generated id
class User(ESModel):
    name: str
    age: int


async def delete_user(user_id: str):
    user = await User.get(user_id)
    await user.delete()
```

<a id="bulk-operations"></a>
### Bulk operations

Bulk operations could be much faster than single operations, if you have lot of documents to
create, update or delete.

You can use context for bulk operations:

```python
from typing import List
from esorm import ESModel, ESBulk


# Here the model have automatically generated id
class User(ESModel):
    name: str
    age: int


async def bulk_create_users():
    async with ESBulk() as bulk:
        # Creating or modifiying models
        for i in range(10):
            user = User(name=f'User {i}', age=i)
            await bulk.save(user)


async def bulk_delete_users(users: List[User]):
    async with ESBulk(wait_for=True) as bulk:  # Here we wait for the bulk operation to finish
        # Deleting models
        for user in users:
            await bulk.delete(user)
```

The `wait_for` argument is optional. If it is `True`, the context will wait for the bulk operation to finish.

<a id="search"></a>
### Search

<a id="general-search"></a>
#### General search

You can search for documents using `search` method, where an ES query can be specified as a dictionary.
You can use `res_dict=True` argument to get the result as a dictionary instead of a list. The key will be the
`id` of the document: `await User.search(query, res_dict=True)`.

If you only need one result, you can use `search_one` method.

```python
from esorm import ESModel


# Here the model have automatically generated id
class User(ESModel):
    name: str
    age: int


async def search_users():
    # Search for users at least 18 years old
    users = await User.search(
        query={
            'bool': {
                'must': [{
                    'range': {
                        'age': {
                            'gte': 18
                        }
                    }
                }]
            }
        }
    )
    for user in users:
        print(user.name)


async def search_one_user():
    # Search a user named John Doe
    user = await User.search_one(
        query={
            'bool': {
                'must': [{
                    'match': {
                        'name': {
                            'query': 'John Doe'
                        }
                    }
                }]
            }
        }
    )
    print(user.name)
```

Queries are type checked, because they are annotated as `TypedDict`s. You can use IDE autocompletion and type checking.

<a id="search-with-field-value-terms-dictionary-search"></a>
#### Search with field value terms (dictionary search)

You can search for documents using `search_by_fields` method, where you can specify a field and a value.
It also has a `res_dict` argument and `search_one_by_fields` variant.

```python
from esorm import ESModel


# Here the model have automatically generated id
class User(ESModel):
    name: str
    age: int


async def search_users():
    # Search users age is 18
    users = await User.search_by_fields({'age': 18})
    for user in users:
        print(user.name)
```

<a id="vector-search"></a>
#### Vector Search

ESORM supports Elasticsearch's vector search capabilities with the `dense_vector` type, enabling semantic search and similarity operations.

### Defining Vector Fields

To define a vector field, use the `DenseVectorField` function:

```python
from esorm import ESModel
from esorm.fields import dense_vector, DenseVectorField

class Document(ESModel):
    id: str
    content: str
    embedding: dense_vector = DenseVectorField(
        ...,  # required field
        dims=384,  # dimension of the vector
        similarity="cosine"  # similarity metric: 'cosine', 'dot_product', or 'l2'
    )
```

### Vector Search with kNN

To perform vector search using k-nearest neighbors (kNN):

```python
# Vector search using kNN
results = await Document.search({
    "knn": {
        "field": "embedding",
        "query_vector": [0.1, 0.2, ...],  # your query vector
        "k": 10,  # number of neighbors to return
        "num_candidates": 100  # number of candidates to consider
    }
})
```

### Hybrid Search

You can combine vector search with text search for hybrid search:

```python
# Hybrid search - combining text match with vector similarity
results = await Document.search({
    "bool": {
        "must": [
            {
                "match": {
                    "content": {"query": "search query"}
                }
            }
        ],
        "should": [
            {
                "knn": {
                    "field": "embedding",
                    "query_vector": [0.1, 0.2, ...],
                    "k": 10
                }
            }
        ]
    }
})
```

<a id="aggregations"></a>
### Aggregations

You can use `aggregate` method to get aggregations.
You can specify an ES aggregation query as a dictionary. It also accepts normal ES queries,
to be able to fiter which documents you want to aggregate.
Both the aggs parameter and the query parameter are type checked, because they are annotated as `TypedDict`s.
You can use IDE autocompletion and type checking.

```python
from esorm import ESModel

# Here the model have automatically generated id
class User(ESModel):
    name: str
    age: int
    country: str

async def aggregate_avg():
    # Get average age of users
    aggs_def = {
        'avg_age': {
            'avg': {
                'field': 'age'
            }
        }
    }
    aggs = await User.aggregate(aggs_def)
    print(aggs['avg_age']['value'])

async def aggregate_avg_by_country(country = 'Hungary'):
    # Get average age of users by country
    aggs_def = {
        'avg_age': {
            'avg': {
                'field': 'age'
            }
        }
    }
    query = {
        'bool': {
            'must': [{
                'match': {
                    'country': {
                        'query': country
                    }
                }
            }]
        }
    }
    aggs = await User.aggregate(aggs_def, query)
    print(aggs['avg_age']['value'])


async def aggregate_terms():
    # Get number of users by country
    aggs_def = {
        'countries': {
            'terms': {
                'field': 'country'
            }
        }
    }
    aggs = await User.aggregate(aggs_def)
    for bucket in aggs['countries']['buckets']:
        print(bucket['key'], bucket['doc_count'])
```

<a id="pagination-and-sorting"></a>
### Pagination and sorting

You can use `Pagination` and `Sort` classes to decorate your models. They simply wrap your models
and add pagination and sorting functionality to them.

#### Pagination

You can add a callback parameter to the `Pagination` class which will be invoked after the search with
the total number of documents found.

```python
from esorm.model import ESModel, Pagination


class User(ESModel):
    id: int  # This will be used as the document _id in the index
    name: str
    age: int


def get_users(page = 1, page_size = 10):

    def pagination_callback(total: int):
        # You may set a header value or something else here
        print(f'Total users: {total}')

    # 1st create the decorator itself
    pagination = Pagination(page=page, page_size=page_size)

    # Then decorate your model
    res = pagination(User).search_by_fields(age=18)

    # Here the result has maximum 10 items
    return res
```

#### Sorting

It is similar to pagination:

```python
from esorm.model import ESModel, Sort


class User(ESModel):
    id: int  # This will be used as the document _id in the index
    name: str
    age: int


def get_users():
    # 1st create the decorator itself
    sort = Sort(sort=[
        {'age': {'order': 'desc'}},
        {'name': {'order': 'asc'}}
    ])

    # Then decorate your model
    res = sort(User).search_by_fields(age=18)

    # Here the result is sorted by age ascending
    return res

def get_user_sorted_by_name():
    # You can also use this simplified syntax
    sort = Sort(sort='name')

    # Then decorate your model
    res = sort(User).all()

    # Here the result is sorted by age descending
    return res
```

<a id="ide-support"></a>
## 🖥 IDE Support

### PyCharm
This project is developed and tested with [PyCharm](https://www.jetbrains.com/pycharm/) IDE. It has full support for type hints and
annotations. You can use the IDE autocompletion and type checking features.
Recommended Jetbrains Plugins:
- [Pydantic](https://plugins.jetbrains.com/plugin/12861-pydantic): This plugin provides support for Pydantic models and type hints, helps a lot with autocompletion and type checking.

### VScode / Cursor

In VSCode you can use PyLance. Unfortunately Pylance use static type checking, unlike PyCharm's heuristic type checker,
which is not too good with ESORM's Union types.

<a id="testing"></a>
## 🧪 Testing

For testing you can use the `test.sh` in the root directory. It is a script to running
tests on multiple python interpreters in virtual environments. At the top of the file you can specify
which python interpreters you want to test. The ES versions are specified in `tests/docker-compose.yml` file.

If you already have a virtual environment, simply use `pytest` to run the tests.

<a id="license"></a>
## 🛡 License

This project is licensed under the terms of the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/) (
MPL 2.0) license.

<a id="citation"></a>
## 📃 Citation

If you use this project in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{esorm,
  author = {Adam Wallner},
  title = {ESORM: ElasticSearch Object Relational Mapper},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/wallneradam/esorm}},
}
```
