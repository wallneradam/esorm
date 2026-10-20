<img src="https://raw.githubusercontent.com/wallneradam/esorm/main/docs/_static/img/esorm.svg" width="110" height="110" align="left" style="margin-right: 1em; margin-bottom: 0.5em" alt="Logo"/>

# ESORM - Python ElasticSearch and OpenSearch ORM based on Pydantic

[![Tests](https://github.com/wallneradam/esorm/actions/workflows/tests.yml/badge.svg)](https://github.com/wallneradam/esorm/actions/workflows/tests.yml)

ESORM is an ElasticSearch Object Relational Mapper or Object Document Mapper (ODM) if you like,
 for Python based on Pydantic. It is a high-level library for managing ElasticSearch documents
 in Python. It is fully async and uses annotations and type hints for type checking and IDE autocompletion.
 It works with OpenSearch too.

## ☰ Table of Contents

- [💾 Installation](#installation)
- [🚀 Features](#features)
    - [Supported ElasticSearch versions](#supported-elasticsearch-versions)
    - [Supported OpenSearch versions](#supported-opensearch-versions)
    - [Supported Python versions](#supported-python-versions)
- [📖 Usage](#usage)
    - [Define a model](#define-a-model)
        - [Python basic types](#python-basic-types)
        - [ESORM field types](#esorm-field-types)
        - [Field Definition](#field-definition)
        - [Subfields](#subfields)
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
        - [Connecting to OpenSearch](#connecting-to-opensearch)
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
        - [Count](#count)
        - [Vector Search](#vector-search)
    - [Aggregations](#aggregations)
    - [Pagination and sorting](#pagination-and-sorting)
    - [Iterating through all results (scan)](#iterating-through-all-results)
- [🔬 Advanced usage](docs/advanced.md#advanced-usage)
    - [Optimistic concurrency control](docs/advanced.md#optimistic-concurrency-control)
    - [Lazy properties](docs/advanced.md#lazy-properties)
    - [Shard routing](docs/advanced.md#shard-routing)
    - [Retreive Selected Fields Only](docs/advanced.md#retreive-selected-fields-only)
    - [Iterating through large result sets](docs/advanced.md#iterating-through-large-result-sets)
    - [Watchers](docs/advanced.md#watchers)
    - [FastAPI integration](docs/advanced.md#fastapi-integration)
        - [FastAPI pagination](docs/advanced.md#fastapi-pagination)
        - [FastAPI streaming export](docs/advanced.md#fastapi-streaming-export)
- [🖥 IDE Support and type checking](#ide-support)
    - [Type checking of queries and aggregations](#type-checking)
- [🧪 Testing](#testing)
- [🛡 License](#license)
- [📃 Citation](#citation)

<a id="installation"></a>
## 💾 Installation


```bash
pip install pyesorm
```

For OpenSearch support install it with the `opensearch` extra:

```bash
pip install pyesorm[opensearch]
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
- Support >= Python 3.10 (tested with 3.10 through 3.14)
- Support for ElasticSearch 9.x and 8.x
- Support for OpenSearch 3.x and 2.x
- Watcher support (You may need ElasticSearch subscription license for this)
- Pagination and sorting
- Iterating through any number of results efficiently (point in time + `search_after`)
- FastAPI integration

Not all ElasticSearch features are supported yet, pull requests are welcome.

<a id="supported-elasticsearch-versions"></a>
### Supported ElasticSearch versions

It is tested with ElasticSearch 8.x and 9.x. ElasticSearch 7.x is supported by ESORM 0.7.x.

ESORM uses the official `elasticsearch` 9.x Python client. The client asks the server for API compatibility
with its own major version, which ElasticSearch 8.x rejects, so ESORM detects the server version
and lowers the requested compatibility version to it automatically.

<a id="supported-opensearch-versions"></a>
### Supported OpenSearch versions

It is tested with OpenSearch 2.19 and 3.x.

OpenSearch is supported through the official `opensearch-py` client, which is installed with the `opensearch`
extra (`pip install pyesorm[opensearch]`). If it is installed, `connect` detects the type of the server
automatically, see [Connecting to OpenSearch](#connecting-to-opensearch).

ESORM wraps the `opensearch-py` client, so the same code works with both servers:

- The most common API methods (`index`, `create`, `update`, `bulk`, `mget`, `search`, `msearch`, `count`,
  `delete_by_query`, `update_by_query`, `indices.create`, `indices.put_mapping`, `indices.put_settings`,
  `indices.put_index_template`) accept the keyword arguments of the `elasticsearch` client (e.g. `query`, `sort`,
  `document`, `operations`, `mappings`), other methods should be called as in `opensearch-py`.
- `dense_vector` fields are mapped to `knn_vector` fields, and ElasticSearch `knn` queries are converted to
  OpenSearch `knn` queries.
- The exceptions of the `elasticsearch` client are raised (e.g. `elasticsearch.NotFoundError`,
  `elasticsearch.ConflictError`), so you can handle errors the same way.

Watchers are not supported with OpenSearch.

<a id="supported-python-versions"></a>
### Supported Python versions

Tested with Python 3.10 through 3.14.

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
| `str`               | `keyword` |                             |
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

<a id="subfields"></a>
#### Subfields

Elasticsearch allows you to define multiple ways to index the same field using subfields. 
This is particularly useful for text fields where you might want both full-text search and exact matching capabilities.

##### Basic Usage with `keyword` Shortcut

The most common use case is adding a `keyword` subfield to a `text` field for exact matching:

```python
from esorm import ESModel
from esorm.fields import TextField

class Product(ESModel):
    # Text field with automatic keyword subfield
    title: str = TextField(..., keyword=True)
    # Equivalent to: {'fields': {'keyword': {'type': 'keyword'}}}
```

With this setup, you can:
- Search full-text on `title`
- Do exact matches, aggregations, or sorting on `title.keyword`

##### Custom Subfields

You can define custom subfields for any field type using the `fields` parameter:

```python
from esorm import ESModel  
from esorm.fields import TextField, NumericField, Field

class Product(ESModel):
    # Text field with multiple analyzers
    description: str = TextField(..., fields={
        'english': {'type': 'text', 'analyzer': 'english'},
        'spanish': {'type': 'text', 'analyzer': 'spanish'}, 
        'suggest': {'type': 'completion'}
    })
    
    # Numeric field with keyword subfield for aggregations
    price: float = NumericField(..., fields={
        'raw': {'type': 'keyword'}
    })
    
    # Regular field with normalized keyword
    category: str = Field(..., fields={
        'lowercase': {'type': 'keyword', 'normalizer': 'lowercase'}
    })
```

##### Combining `keyword` Shortcut with Custom Fields

You can use both the `keyword` shortcut and custom fields together. The keyword subfield will be automatically added if not already defined:

```python
class Product(ESModel):
    # Both keyword and custom fields
    title: str = TextField(..., keyword=True, fields={
        'suggest': {'type': 'completion'},
        'ngram': {'type': 'text', 'analyzer': 'ngram'}
    })
    # Results in: {
    #     'keyword': {'type': 'keyword'},  # Added automatically
    #     'suggest': {'type': 'completion'},
    #     'ngram': {'type': 'text', 'analyzer': 'ngram'}
    # }
    
    # Custom keyword configuration takes precedence
    name: str = TextField(..., keyword=True, fields={
        'keyword': {'type': 'keyword', 'ignore_above': 512}  # Custom config preserved
    })
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


async def create_place():
    place = Place(name='Budapest', location=geo_point(lat=47.4979, lon=19.0402))
    await place.save()
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

ElasticSearch 9.2+ excludes vectors from `_source` by default, so they would be missing from the loaded models.
To keep the models complete, ESORM creates indices with dense vector fields with
`index.mapping.exclude_source_vectors: false`, unless you set `exclude_source_vectors` explicitly
in `ESConfig.settings`.

With OpenSearch, `dense_vector` fields are mapped to `knn_vector` fields (`dims` → `dimension`,
`similarity` → `space_type`) and `index.knn: true` is set on the index. Here `dims` is required, because
OpenSearch can't detect it from the documents. OpenSearch 3.x+ derives the vectors in `_source` from the index
(normalized for cosine similarity), so ESORM creates the indices with `index.knn.derived_source.enabled: false`,
unless you set it explicitly in `ESConfig.settings`.

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

<a id="connecting-to-opensearch"></a>
#### Connecting to OpenSearch

If the `opensearch-py` package is installed (`pip install pyesorm[opensearch]`), `connect` detects
automatically whether the server is ElasticSearch or OpenSearch. In this case use the arguments of
`AsyncElasticsearch`, the most common ones (`basic_auth`, `request_timeout`) are converted for OpenSearch:

```python
from esorm import connect, is_opensearch


async def es_init():
    await connect('http://localhost:9200', wait=True, basic_auth=('user', 'password'))
    print(is_opensearch())
```

If you need OpenSearch specific arguments (e.g. AWS authentication), specify the backend explicitly,
then all arguments are passed to `AsyncOpenSearch` as they are:

```python
from opensearchpy import AWSV4SignerAsyncAuth, AsyncHttpConnection
from esorm import connect


async def es_init(credentials):
    await connect('https://my-domain.eu-central-1.es.amazonaws.com', backend='opensearch',
                  http_auth=AWSV4SignerAsyncAuth(credentials, 'eu-central-1', 'es'),
                  connection_class=AsyncHttpConnection)
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

#### Refresh Control

You can control when changes become visible to search operations:

```python
# Default: No refresh - fastest, changes visible after refresh_interval
await user.save()

# Immediate refresh - changes immediately visible to search
await user.save(refresh=True)

# Wait for next refresh cycle - waits for the scheduled refresh_interval
await user.save(wait_for=True)

# Alternative: ES API compatible syntax (same as wait_for=True)
await user.save(refresh="wait_for")
```

**Performance Considerations:**
- **Default** (`refresh=False`, `wait_for=False`): Best performance. Changes become visible after the index's `refresh_interval` (default 1 second).
- **`refresh=True`**: Forces an immediate refresh. Use for testing or when immediate visibility is critical. Can impact performance under heavy load.
- **`wait_for=True`**: Waits for the next scheduled refresh. Good balance between consistency and performance.
- **`refresh="wait_for"`**: Same as `wait_for=True`, provided for ElasticSearch API compatibility.

**Note:** If both `refresh` and `wait_for` are specified, `refresh` takes priority.

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
    async with ESBulk(wait_for=True) as bulk:  # Wait for scheduled refresh
        # Deleting models
        for user in users:
            await bulk.delete(user)


async def bulk_create_users_immediate():
    async with ESBulk(refresh=True) as bulk:  # Force immediate refresh
        for i in range(10):
            user = User(name=f'User {i}', age=i)
            await bulk.save(user)
```

**Refresh Control for Bulk Operations:**
- **`wait_for=True`**: Waits for the next scheduled refresh after bulk operations complete. Good balance between consistency and performance.
- **`refresh=True`**: Forces an immediate refresh after bulk operations. Use for testing or when immediate visibility is critical.
- **`refresh="wait_for"`**: Same as `wait_for=True`, ES API compatible syntax.
- **Default** (no parameters): Best performance, changes visible after the index's `refresh_interval`.

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
    if user is not None:  # It is None if no document matches
        print(user.name)
```

Queries are type checked, because they are annotated as `TypedDict`s. You can use IDE autocompletion and type checking.
If you store a query in a variable, annotate it with `ESQuery`, see [Type checking](#type-checking).

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

<a id="count"></a>
#### Count

You can count the number of documents in the index with the `count` method:

```python
from esorm import ESModel


class User(ESModel):
    name: str
    age: int


async def count_users():
    count = await User.count()
    print(count)


async def count_users_by_age():
    count = await User.count(query={'term': {'age': 18}})
    print(count)
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
You can use IDE autocompletion and type checking. If you store them in variables, annotate them with `ESAggs` and
`ESQuery` (see [Type checking](#type-checking)). Sub-aggregations can be added with the `aggs` key, their results
are accessible by their names.

```python
from esorm import ESModel
from esorm.aggs import ESAggs
from esorm.query import ESQuery

# Here the model have automatically generated id
class User(ESModel):
    name: str
    age: int
    country: str

async def aggregate_avg():
    # Get average age of users
    aggs_def: ESAggs = {
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
    aggs_def: ESAggs = {
        'avg_age': {
            'avg': {
                'field': 'age'
            }
        }
    }
    query: ESQuery = {
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
    aggs = await User.aggregate(aggs_def, query=query)
    print(aggs['avg_age']['value'])


async def aggregate_terms():
    # Get number of users by country
    aggs_def: ESAggs = {
        'countries': {
            'terms': {
                'field': 'country'
            }
        }
    }
    aggs = await User.aggregate(aggs_def)
    for bucket in aggs['countries']['buckets']:
        print(bucket['key'], bucket['doc_count'])


async def aggregate_avg_age_by_country():
    # Get the average age of users by country, using a sub-aggregation
    aggs = await User.aggregate({
        'countries': {
            'terms': {'field': 'country'},
            'aggs': {
                'avg_age': {'avg': {'field': 'age'}}
            }
        }
    })
    for bucket in aggs['countries']['buckets']:
        print(bucket['key'], bucket['avg_age']['value'])
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


async def get_users(page = 1, page_size = 10):

    async def pagination_callback(total: int):
        # You may set a header value or something else here
        print(f'Total users: {total}')

    # 1st create the decorator itself
    pagination = Pagination(page=page, page_size=page_size, callback=pagination_callback)

    # Then decorate your model
    res = await pagination(User).search_by_fields({'age': 18})

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


async def get_users():
    # 1st create the decorator itself
    sort = Sort(sort=[
        {'age': {'order': 'desc'}},
        {'name': {'order': 'asc'}}
    ])

    # Then decorate your model
    res = await sort(User).search_by_fields({'age': 18})

    # Here the result is sorted by age descending, then by name ascending
    return res

async def get_user_sorted_by_name():
    # You can also use this simplified syntax
    sort = Sort(sort='name')

    # Then decorate your model
    res = await sort(User).all()

    # Here the result is sorted by name ascending
    return res
```

<a id="iterating-through-all-results"></a>
### Iterating through all results (scan)

Pagination is for showing results to users, it can't go deeper than 10,000 results by default
(`index.max_result_window`). To process all results of a query (e.g. export, reindex, batch jobs),
use `scan()`, which can iterate through any number of documents efficiently:

```python
from esorm import ESModel


class User(ESModel):
    name: str
    age: int


async def process_adults():
    async for user in User.scan({'range': {'age': {'gte': 18}}}):
        print(user.name)
```

To process the documents in batches (e.g. with bulk operations), use `scan_batches()`:

```python
async def process_in_batches():
    async for users in User.scan_batches(batch_size=500):
        # Here `users` is a list of maximum 500 models
        print(len(users))
```

It uses a point in time (PIT) with `search_after`, so the results are consistent: changes made during
the iteration are not visible. Sorting, the `Sort` decorator, `_source` and shard routing work like
in `search()`.

If the iteration may be stopped early (e.g. by `break`), wrap it in `contextlib.aclosing()` to close
the point in time immediately. More details in
[Iterating through large result sets](docs/advanced.md#iterating-through-large-result-sets).

<a id="ide-support"></a>
## 🖥 IDE Support and type checking

ESORM has type hints everywhere (it is a [PEP 561](https://peps.python.org/pep-0561/) typed package), so IDEs can
autocomplete and check the models, the queries, the aggregations and the results. The type hints are tested with
[pyright](https://github.com/microsoft/pyright) (used by Pylance in VS Code / Cursor) and
[mypy](https://mypy-lang.org/) in the CI (`tests/typecheck`): correct code must not have any warnings, and typical
mistakes must be reported.

<a id="type-checking"></a>
### Type checking of queries and aggregations

Queries and aggregations are `TypedDict`s, so the keys are autocompleted, and typos, wrong value types and
missing required keys are reported:

```python
from esorm import ESModel


class User(ESModel):
    name: str
    age: int
    country: str


async def search():
    # Error: "rnage" is an undefined item in "ESQuery"
    await User.search({'bool': {'filter': [{'rnage': {'age': {'gte': 18}}}]}})
    # Error: "lesser" is an undefined item in "ESRange"
    await User.search({'range': {'age': {'gte': 18, 'lesser': 65}}})
    # Error: "query" is required in "ESNested"
    await User.search({'nested': {'path': 'tags'}})
    # Error: "adn" is not assignable to the operator literals
    await User.search({'match': {'name': {'query': 'John Doe', 'operator': 'adn'}}})
```

Every query clause has the same type: `ESQuery`. It is used for the whole query, and also for the clauses of
compound queries (`bool`, `nested`, `function_score`, ...). An aggregation is an `ESAgg`, the aggregations
(and the sub-aggregations) are `ESAggs`.

If you pass a query directly to a method, it is checked automatically. If you store it in a variable,
annotate the variable, because type checkers infer an unannotated dict literal as a plain `dict`, and report that
it is not an `ESQuery`:

```python
from typing import List
from esorm.query import ESQuery
from esorm.aggs import ESAggs


async def search_users(name: str, min_age: int):
    must: List[ESQuery] = []
    if name:
        must.append({'match': {'name': name}})
    if min_age:
        must.append({'range': {'age': {'gte': min_age}}})
    query: ESQuery = {'bool': {'must': must}}
    # Or create it with the TypedDict class: query = ESQuery(bool={'must': must})
    return await User.search(query)


async def avg_age():
    aggs_def: ESAggs = {'avg_age': {'avg': {'field': 'age'}}}
    aggs = await User.aggregate(aggs_def)
    return aggs['avg_age']['value']
```

The results are typed too: `search()` returns a list of models (a dict of id - model pairs with `res_dict=True`),
`search_one()` returns a model or `None`, and in the result of `aggregate()` the known keys (`value`, `buckets`,
`key`, `doc_count`, ...) have their types, while any other key is the result of a sub-aggregation:

```python
async def avg_age_by_country():
    aggs = await User.aggregate({
        'countries': {
            'terms': {'field': 'country'},
            'aggs': {'avg_age': {'avg': {'field': 'age'}}}
        }
    })
    for bucket in aggs['countries']['buckets']:
        print(bucket['key'], bucket['doc_count'], bucket['avg_age']['value'])
```

Some notes:

- The clauses of a `bool` query must be lists (`'must': [{...}]`). ES accepts a single clause too, but PyCharm
  can't check a union of a dict and a list, so the type hints allow only lists. The same applies to the `filters`
  aggregation: only named filters (a dict) are allowed.
- The `value` of the `avg`, `min` and `max` aggregations is `None` if there are no documents, so it is
  `Optional[float]`.
- Buckets are typed as a list. With `keyed: True` (or with named filters in a `filters` aggregation) ES returns them
  as a dict, use `typing.cast()` there.
- ES allows only one query type in a query clause (`{'match': ..., 'term': ...}` is invalid), this is not checked,
  use `bool` to combine them.
- Rarely used queries and aggregations (e.g. span queries, `intervals`, `scripted_metric`) accept any dict.
- The query and aggregation types can be used in pydantic models too, e.g. in a FastAPI request body.

### PyCharm

This project is developed with [PyCharm](https://www.jetbrains.com/pycharm/) IDE. You can use the IDE
autocompletion and type checking features. The type hints are designed to work with PyCharm's type checker too
(tested with PyCharm 2026.2): correct queries and aggregations are not reported, and the typical mistakes are.
Its limitations:

- It checks the query clauses only to the first level of nesting, e.g. a clause in a `bool` query which is in a
  clause of another `bool` query is not checked. Pyright and mypy check every level.
- If a model method returning models (`search()`, `search_one()`, `scan()`, ...) gets a dict literal as query,
  PyCharm can't infer the type of the result (it is a PyCharm bug: a generic type can't be inferred if a `TypedDict`
  literal argument doesn't have all the keys). So there is no autocompletion for the results. Annotate the query
  or the result to have it:

  ```python
  query: ESQuery = {'term': {'country': 'HU'}}
  users = await User.search(query)  # List[User]
  # or
  users: List[User] = await User.search({'term': {'country': 'HU'}})
  ```

Recommended Jetbrains Plugins:
- [Pydantic](https://plugins.jetbrains.com/plugin/12861-pydantic): This plugin provides support for Pydantic models and type hints, helps a lot with autocompletion and type checking.

### VS Code / Cursor

In VS Code and Cursor you can use Pylance (or the pyright extension). Set the type checking mode to at least
`basic` (`"python.analysis.typeCheckingMode": "standard"` is recommended), otherwise Pylance doesn't report type
errors.

<a id="testing"></a>
## 🧪 Testing

Tests run automatically on GitHub Actions for every push and pull request, on all supported
Python versions (3.10 - 3.14) against ElasticSearch 8.x, 9.x and OpenSearch 2.x, 3.x.

For local testing you can use the `tests.sh` in the root directory. It is a script to running
tests on multiple python interpreters in virtual environments. At the top of the file you can specify
which python interpreters you want to test. The ES and OpenSearch versions are specified in
`tests/docker-compose.yml` file.

The development dependencies are in the `dev` dependency group of `pyproject.toml`, you can install them with
`uv sync` or `pip install -e . --group dev` (pip 25.1+).

If you already have a virtual environment, simply use `pytest` to run the tests. Docker is needed, the tests
start the ES and OpenSearch containers automatically. By default all versions are tested, you can limit them with
the `ESORM_TEST_SERVICES` environment variable, e.g. `ESORM_TEST_SERVICES=es9x,os3x pytest`
(available services: `es8x`, `es9x`, `os2x`, `os3x`).

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
