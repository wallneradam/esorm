<img src="docs/_static/img/esorm.svg" width="110" height="110" align="left" style="margin-right: 1em; margin-bottom: 0.5em" alt="Logo"/>

# ESORM - Python ElasticSearch and OpenSearch ORM based on Pydantic

[![Tests](https://github.com/wallneradam/esorm/actions/workflows/tests.yml/badge.svg)](https://github.com/wallneradam/esorm/actions/workflows/tests.yml)

ESORM is an ElasticSearch Object Relational Mapper (or Object Document Mapper, if you like) for Python, based on
Pydantic. You define your documents as Pydantic models, and ESORM creates the indices and mappings, and saves, loads
and searches the documents. It is fully async, and it is fully type hinted, so IDEs can autocomplete and check
the models, the queries and the results. It works with OpenSearch too.

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
        - [Field definition](#field-definition)
        - [Subfields](#subfields)
        - [geo_point](#geo_point)
        - [dense_vector](#dense_vector)
        - [Nested documents](#nested-documents)
        - [List primitive fields](#list-primitive-fields)
        - [ESBaseModel](#esbasemodel)
        - [Id field](#id-field)
        - [Model settings](#model-settings)
        - [ESModelTimestamp](#esmodeltimestamp)
        - [Describe fields](#describe-fields)
        - [Aliases](#aliases)
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
    - [Search](#searching)
        - [General search](#general-search)
        - [Query builder](#query-builder-intro)
        - [Search with field value terms (dictionary search)](#search-with-field-value-terms-dictionary-search)
        - [Count](#count)
        - [Vector search](#vector-search)
    - [Aggregations](#aggregations)
    - [Pagination and sorting](#pagination-and-sorting)
    - [Iterating through all results (scan)](#iterating-through-all-results)
- [🏗 Query builder](docs/query_builder.md#query-builder)
    - [Two ways to build a query: parameters and `with` blocks](docs/query_builder.md#two-ways-to-build-a-query)
    - [How `with` blocks work](docs/query_builder.md#how-with-blocks-work)
    - [Combining queries with operators](docs/query_builder.md#combining-queries-with-operators)
    - [Building queries dynamically](docs/query_builder.md#building-queries-dynamically)
    - [Query clauses](docs/query_builder.md#query-clauses)
- [🔬 Advanced usage](docs/advanced.md#advanced-usage)
    - [Optimistic concurrency control](docs/advanced.md#optimistic-concurrency-control)
    - [Lazy properties](docs/advanced.md#lazy-properties)
    - [Shard routing](docs/advanced.md#shard-routing)
    - [Retrieve selected fields only](docs/advanced.md#retrieve-selected-fields-only)
    - [Iterating through large result sets](docs/advanced.md#iterating-through-large-result-sets)
    - [Watchers](docs/advanced.md#watchers)
    - [FastAPI integration](docs/advanced.md#fastapi-integration)
        - [FastAPI pagination](docs/advanced.md#fastapi-pagination)
        - [FastAPI streaming export](docs/advanced.md#fastapi-streaming-export)
- [🖥 IDE support and type checking](#ide-support)
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

- Pydantic models for ElasticSearch documents
- Automatic index and mapping creation from the models
- CRUD operations
- Async only (there is no sync API)
- Conversion between Python and ElasticSearch types
- Nested documents
- Automatic optimistic concurrency control
- Custom id field
- Bulk operations in an `async with` block
- IDE autocompletion and type checking (tested with PyCharm, pyright and mypy)
- Docstrings and type hints everywhere in the source code
- `TypedDict`s for ElasticSearch queries and aggregations
- A query builder, if you don't like nested dicts
- Field descriptions from docstrings
- Shard routing
- Lazy properties (async computed fields)
- Python 3.10 - 3.14
- ElasticSearch 8.x and 9.x
- OpenSearch 2.x and 3.x
- Watchers (ElasticSearch needs a paid subscription for them)
- Pagination and sorting
- Iterating through any number of results efficiently (point in time + `search_after`)
- FastAPI integration

Not all ElasticSearch features are supported yet. Pull requests are welcome.

<a id="supported-elasticsearch-versions"></a>
### Supported ElasticSearch versions

ESORM is tested with ElasticSearch 8.x and 9.x. If you need ElasticSearch 7.x, use ESORM 0.7.x.

ESORM uses the official `elasticsearch` 9.x Python client. This client asks the server for API compatibility with
its own major version, which ElasticSearch 8.x rejects. ESORM detects the version of the server, and if it is older,
it asks for the compatibility of that version instead.

<a id="supported-opensearch-versions"></a>
### Supported OpenSearch versions

ESORM is tested with OpenSearch 2.19 and 3.x.

OpenSearch is supported through the official `opensearch-py` client, which is installed with the `opensearch` extra
(`pip install pyesorm[opensearch]`). If it is installed, `connect` detects the type of the server automatically,
see [Connecting to OpenSearch](#connecting-to-opensearch).

ESORM wraps the `opensearch-py` client, so the same code works with both servers:

- The most common API methods (`index`, `create`, `update`, `bulk`, `mget`, `search`, `msearch`, `count`,
  `delete_by_query`, `update_by_query`, `indices.create`, `indices.put_mapping`, `indices.put_settings`,
  `indices.put_index_template`) accept the keyword arguments of the `elasticsearch` client (e.g. `query`, `sort`,
  `document`, `operations`, `mappings`). Call the other methods as you would in `opensearch-py`.
- `dense_vector` fields are mapped to `knn_vector` fields, and ElasticSearch `knn` queries are converted to
  OpenSearch `knn` queries.
- ESORM raises the exceptions of the `elasticsearch` client (e.g. `elasticsearch.NotFoundError`,
  `elasticsearch.ConflictError`), so you can handle errors the same way for both servers.

Watchers are not supported with OpenSearch.

<a id="supported-python-versions"></a>
### Supported Python versions

Python 3.10 through 3.14 are tested.

<a id="usage"></a>
## 📖 Usage

<a id="define-a-model"></a>
### Define a model

`ESModel` is a subclass of `pydantic.BaseModel` (through `ESBaseModel`, see [below](#esbasemodel)), so you can use
every feature of [Pydantic](https://docs.pydantic.dev/latest/concepts/models/) models. ESORM adds the ElasticSearch
specific parts: it converts the documents to and from ElasticSearch types, and runs the ElasticSearch operations for
you.

Every model has its own index. By default, the name of the index is the name of the class in snake_case with the
`esorm_` prefix, e.g. the index of `User` is `esorm_user`, and the index of `BlogPost` is `esorm_blog_post`. See
[Model settings](#model-settings) if you need another name.

<a id="python-basic-types"></a>
#### Python basic types

```python
from esorm import ESModel


class User(ESModel):
    name: str
    age: int
```

The Python types are mapped to these ElasticSearch types:

| Python type         | ES type   | Comment                         |
|---------------------|-----------|---------------------------------|
| `str`               | `keyword` |                                 |
| `int`               | `long`    |                                 |
| `float`             | `double`  |                                 |
| `bool`              | `boolean` |                                 |
| `datetime.datetime` | `date`    |                                 |
| `datetime.date`     | `date`    |                                 |
| `datetime.time`     | `date`    | Stored as 1970-01-01 + the time |
| `typing.Literal`    | `keyword` |                                 |
| `UUID`              | `keyword` |                                 |
| `Path`              | `keyword` |                                 |
| `IntEnum`           | `integer` |                                 |
| `Enum`              | `keyword` | `StrEnum` too                   |

Note that `str` is mapped to `keyword`, which is for exact values. For full text search use the `text` type of
`esorm.fields` (see below).

Use UTC datetimes: the timezone is removed from the value without conversion, and the value is stored as UTC.

Some Pydantic types are supported too:

| Pydantic type                         | ES type   |
|---------------------------------------|-----------|
| `AnyUrl` and its subclasses (`HttpUrl`, ...) | `keyword` |
| `IPvAnyAddress`                       | `ip`      |

Fields can be `Optional`, e.g. `age: Optional[int] = None`. The mapping is the same as without `Optional`.

<a id="esorm-field-types"></a>
#### ESORM field types

The `esorm.fields` module has types for the other ElasticSearch field types:

```python
from esorm import ESModel
from esorm.fields import keyword, text, byte, geo_point, dense_vector


class User(ESModel):
    name: text
    email: keyword
    age: byte
    location: geo_point
    embedding: dense_vector
    # ...
```

The supported types are:

| Field type                  | ES type         |
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

In Python these are ordinary values: a `text` field is a `str`, a `byte` field is an `int`, and so on.

A `binary` field stores **base64** encoded strings. You can give it `bytes` too, they are converted to base64. When
you read the field, it is always the base64 string, but its `bytes` property returns the decoded data:
`user.avatar.bytes`.

The ES type of `Annotated` types is the type of their base type, e.g. Pydantic `PositiveInt` and `NegativeInt` are
`long` fields.

<a id="field-definition"></a>
#### Field definition

The type annotation of a field decides its ES type. If you need more control, use the field functions of
`esorm.fields` as the default value of the field. They accept the arguments of `pydantic.Field` (validation,
`alias`, `description`, ...) and some ElasticSearch mapping options:

```python
from esorm import ESModel
from esorm.fields import Field, TextField, NumericField, DenseVectorField, text, geo_point, dense_vector


class Product(ESModel):
    id: str
    name: text = TextField(..., min_length=3, max_length=100)
    price: float = NumericField(..., gt=0)
    is_available: bool = Field(True)
    internal_note: str = Field('', index=False)  # Stored, but not searchable
    location: geo_point
    embedding: dense_vector = DenseVectorField(..., dims=384, similarity="cosine")
```

The first argument is the default value, `...` means that the field is required.

| Function           | For                 | ES mapping options                    |
|--------------------|---------------------|---------------------------------------|
| `Field`            | any field           | `index`, `fields`                     |
| `TextField`        | `text` and `keyword` | `index`, `fields`, `keyword`         |
| `NumericField`     | numeric fields      | `index`, `fields`                     |
| `DenseVectorField` | `dense_vector`      | `index`, `dims`, `similarity`         |

`index=False` stores the field, but it can't be searched. You can also give `analyzer` and `search_analyzer` to the
field functions, they are added to the mapping as they are.

<a id="subfields"></a>
#### Subfields

ElasticSearch can index the same field in more ways, using subfields (multi-fields). The most common case is a
`text` field for full text search, with a `keyword` subfield for exact matching, sorting and aggregations.

##### The `keyword` shortcut

```python
from esorm import ESModel
from esorm.fields import TextField, text


class Product(ESModel):
    # A text field with a keyword subfield
    title: text = TextField(..., keyword=True)
    # The same as: TextField(..., fields={'keyword': {'type': 'keyword'}})
```

With this mapping you can:

- search `title` with full text queries (e.g. `match`)
- use `title.keyword` for exact matches, aggregations and sorting

##### Custom subfields

The `fields` parameter adds any subfields, to any type of field:

```python
from esorm import ESModel
from esorm.fields import TextField, NumericField, Field, text


class Product(ESModel):
    # A text field with more analyzers
    description: text = TextField(..., fields={
        'english': {'type': 'text', 'analyzer': 'english'},
        'spanish': {'type': 'text', 'analyzer': 'spanish'},
        'suggest': {'type': 'completion'}
    })

    # A numeric field with a keyword subfield
    price: float = NumericField(..., fields={
        'raw': {'type': 'keyword'}
    })

    # A keyword field with a lowercase subfield
    category: str = Field(..., fields={
        'lowercase': {'type': 'keyword', 'normalizer': 'lowercase'}
    })
```

##### The `keyword` shortcut together with custom subfields

If you give both, the `keyword` subfield is added to the custom subfields, unless they already have one:

```python
from esorm import ESModel
from esorm.fields import TextField, text


class Product(ESModel):
    title: text = TextField(..., keyword=True, fields={
        'suggest': {'type': 'completion'},
        'ngram': {'type': 'text', 'analyzer': 'ngram'}
    })
    # The subfields: {
    #     'keyword': {'type': 'keyword'},  # Added by keyword=True
    #     'suggest': {'type': 'completion'},
    #     'ngram': {'type': 'text', 'analyzer': 'ngram'}
    # }

    # Your own keyword subfield is kept as it is
    name: text = TextField(..., keyword=True, fields={
        'keyword': {'type': 'keyword', 'ignore_above': 512}
    })
```

<a id="geo_point"></a>
#### geo_point

Use the `geo_point` type for locations:

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

The `dense_vector` type stores vectors (embeddings) for vector similarity search, see
[Vector search](#vector-search):

```python
from esorm import ESModel
from esorm.fields import dense_vector, DenseVectorField


class Document(ESModel):
    id: str
    content: str
    embedding: dense_vector = DenseVectorField(
        ...,  # Required field
        dims=384,  # The dimension of the vectors
        # The similarity metric: 'cosine' (default), 'dot_product', 'l2_norm' or 'max_inner_product'
        similarity="cosine"
    )
```

ElasticSearch 9.2+ excludes vectors from `_source` by default, so they would be missing from the loaded models.
To keep the models complete, ESORM creates the indices which have dense vector fields with
`index.mapping.exclude_source_vectors: false`, unless you set `exclude_source_vectors` in `ESConfig.settings`.

With OpenSearch, `dense_vector` fields are mapped to `knn_vector` fields (`dims` → `dimension`,
`similarity` → `space_type`), and `index.knn: true` is set on the index. `dims` is required here, because
OpenSearch can't detect it from the documents. OpenSearch 3.x+ derives the vectors in `_source` from the index
(normalized for cosine similarity), so ESORM creates the indices with `index.knn.derived_source.enabled: false`,
unless you set it in `ESConfig.settings`.

<a id="nested-documents"></a>
#### Nested documents

A field can be another model. Use `ESBaseModel` for these sub-models, because an `ESModel` would get its own index
too (see [ESBaseModel](#esbasemodel)):

```python
from typing import List
from esorm import ESModel, ESBaseModel
from esorm.fields import keyword, text, byte


class User(ESBaseModel):
    name: text
    email: keyword
    age: byte = 18


class Comment(ESBaseModel):
    author: keyword
    text: text


class Post(ESModel):
    title: text
    content: text
    writer: User  # An object field
    comments: List[Comment] = []  # A nested field
```

A single sub-model is an `object` field, its fields are stored as if they were fields of the main document, so you
can query them directly, e.g. `writer.name`.

A list of sub-models is a `nested` field. ElasticSearch stores every item as a separate (hidden) document, so a
query can match the fields of the same item. You need a `nested` query for these fields, see
[Joining queries](docs/query_builder.md#builder-joining-queries) in the query builder chapter.

<a id="list-primitive-fields"></a>
#### List primitive fields

Fields can be lists of primitive types:

```python
from typing import List
from esorm import ESModel


class User(ESModel):
    emails: List[str]
    favorite_ids: List[int]
    # ...
```

ElasticSearch has no list type, every field can have more values. So the mapping is the same as for a single
value, e.g. `List[str]` is a `keyword` field.

<a id="esbasemodel"></a>
#### ESBaseModel

`ESBaseModel` is the base class of `ESModel`. It has the same features, but it doesn't have an index. Use it for
sub-models (see [Nested documents](#nested-documents)), and for base classes of your models.

##### Base classes

```python
from esorm import ESModel, ESBaseModel
from esorm.fields import keyword, text, byte


# BaseUser doesn't have an index
class BaseUser(ESBaseModel):
    # The subclasses inherit this config
    class ESConfig:
        id_field = 'email'

    name: text
    email: keyword


# UserExtended has an index, because it is a subclass of ESModel
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

The `ESConfig` is inherited from the first base class, and the `index` setting is never inherited, every model
has its own index.

<a id="id-field"></a>
#### Id field

By default, ElasticSearch generates the id of the document when it is saved. If you want to set it yourself,
specify the id field in the [model settings](#model-settings):

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

The id field is not stored in the document, its value is the document `_id`.

A field named `id` is always the id field, you don't need to set `id_field` for it:

```python
from esorm import ESModel


class User(ESModel):
    id: int  # This is the document _id
    name: str
```

You can also compute the id in an `__id__` property:

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

`__id__` must be a property. In this case `email` is stored in the document too.

The `_id` of a document is always a string in ElasticSearch, even if the id field is an `int`.

<a id="model-settings"></a>
#### Model settings

You can configure a model in its `ESConfig` inner class. These are the settings and their defaults:

```python
from typing import Optional, List, Dict, Any
from esorm import ESModel


class User(ESModel):
    class ESConfig:
        """ ESModel Config """
        # The name of the index, the default is the class name in snake_case with the index prefix
        index: Optional[str] = None
        # The name of the id field
        id_field: Optional[str] = None
        # The default sort of the searches, e.g. [{'name': {'order': 'asc'}}]
        default_sort: Optional[List[Dict[str, Dict[str, str]]]] = None
        # ElasticSearch index settings, they are used when the index is created
        # (https://www.elastic.co/docs/reference/elasticsearch/index-settings/)
        settings: Optional[Dict[str, Any]] = None
        # Don't create the index and the mappings of this model in setup_mappings()
        skip_init: bool = False
        # Maximum recursion depth of lazy properties
        lazy_property_max_recursion_depth: int = 1
```

<a id="esmodeltimestamp"></a>
#### ESModelTimestamp

`ESModelTimestamp` adds `created_at` and `modified_at` fields to your model:

```python
from esorm import ESModelTimestamp


class User(ESModelTimestamp):
    name: str
    age: int
```

They are set automatically when the document is saved (also in bulk operations): `created_at` when it is saved
for the first time, `modified_at` every time.

<a id="describe-fields"></a>
#### Describe fields

You can describe the fields the usual Pydantic way, but docstrings work too:

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

The descriptions are in the JSON schema of the model, so they are useful if you generate API documentation from
your models, e.g. with [FastAPI](https://fastapi.tiangolo.com/).

<a id="aliases"></a>
#### Aliases

A field can have another name in ElasticSearch:

```python
from esorm import ESModel
from esorm.fields import keyword, Field


class User(ESModel):
    full_name: keyword = Field(alias='fullName')  # The field is `fullName` in ES
```

This way you can rename a field in the model, without changing the mapping and the stored documents.

<a id="connecting-to-elasticsearch"></a>
### Connecting to ElasticSearch

Connect with a simple connection string:

```python
from esorm import connect


async def es_init():
    await connect('localhost:9200')
```

Or connect to more hosts of a cluster:

```python
from esorm import connect


async def es_init():
    await connect(['localhost:9200', 'localhost:9201'])
```

You can wait for the node or the cluster to be ready (recommended):

```python
from esorm import connect


async def es_init():
    await connect('localhost:9200', wait=True)
```

It pings the server every 2 seconds until it answers. There is no time limit, so it waits as long as it takes.

`connect` passes the other arguments to `AsyncElasticsearch`:

```python
from esorm import connect


async def es_init():
    await connect('localhost:9200', wait=True, sniff_on_start=True, sniff_on_node_failure=True)
```

<a id="connecting-to-opensearch"></a>
#### Connecting to OpenSearch

If the `opensearch-py` package is installed (`pip install pyesorm[opensearch]`), `connect` detects automatically
whether the server is ElasticSearch or OpenSearch. Use the arguments of `AsyncElasticsearch` here too, the most
common ones (`basic_auth`, `request_timeout`) are converted for OpenSearch:

```python
from esorm import connect, is_opensearch


async def es_init():
    await connect('http://localhost:9200', wait=True, basic_auth=('user', 'password'))
    print(is_opensearch())
```

If you need OpenSearch specific arguments (e.g. AWS authentication), specify the backend explicitly. Then all
arguments are passed to `AsyncOpenSearch` as they are:

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

`connect` creates an `AsyncElasticsearch` client and stores it globally. The models use this client.
`esorm.es` is a proxy to this client, use it as an `AsyncElasticsearch` instance if you need the ElasticSearch API
directly:

```python
from esorm import es


async def es_init():
    await es.ping()
```

<a id="create-index-templates"></a>
### Create index templates

You can create index templates:

```python
from esorm.model import create_index_template


async def prepare_es():
    await create_index_template('default_template',
                                prefix_name='esorm_',
                                shards=3,
                                auto_expand_replicas='1-5')
```

This template is applied to every index whose name starts with `esorm_`, which is the default prefix of the
indices. The arguments after `replicas` are added to the index settings.

You can change the prefix of the indices globally, before the models are defined:

```python
from esorm.model import set_default_index_prefix

set_default_index_prefix('custom_prefix_')
```

<a id="create-indices-and-mappings"></a>
### Create indices and mappings

ESORM creates the indices and mappings from your models:

```python
from esorm import setup_mappings


async def prepare_es():
    import models  # Import your models
    # The models argument is not needed, but passing it prevents the unused import warning
    await setup_mappings(models)
```

The model classes are registered in a global registry when they are defined, so first import every model. Then
`setup_mappings` creates the index and the mapping of every registered model.

If an index already exists, `setup_mappings` adds the new fields to its mapping, but it can't change or delete the
existing fields. This is an ElasticSearch limitation, for such changes you need to reindex the data. If the new
mapping is not compatible with the existing one, a warning is logged, and the other models are processed.
`ESConfig.settings` is used only when the index is created.

With `setup_mappings(debug=True)` the mappings are logged.

<a id="model-instances"></a>
### Model instances

The models returned by `get`, `search` and the other methods have these private attributes:

| Attribute       | Description                         |
|-----------------|-------------------------------------|
| `_id`           | The ES id of the document           |
| `_routing`      | The routing value of the document   |
| `_version`      | The version of the document         |
| `_primary_term` | The primary term of the document    |
| `_seq_no`       | The sequence number of the document |

<a id="crud-create"></a>
### CRUD: Create

```python
from esorm import ESModel


# This model has no id field, so ElasticSearch generates the ids
class User(ESModel):
    name: str
    age: int


async def create_user():
    user = User(name='John Doe', age=25)
    # Save the user to ElasticSearch
    new_user_id = await user.save()
    print(new_user_id)
```

`save()` returns the id of the document. If a document with the same id already exists, it is replaced.

#### Refresh control

ElasticSearch makes the changes visible to searches when the index is refreshed. `save()`, `delete()` and
`ESBulk` have a `refresh` parameter to control this:

```python
# Default: no refresh, the change is visible after the next periodic refresh (refresh_interval)
await user.save()

# Refresh immediately, the change is visible when save() returns
await user.save(refresh=True)

# Wait for the next periodic refresh, the change is visible when save() returns
await user.save(refresh="wait_for")
```

- **No refresh** (default): the fastest. The change is visible after the `refresh_interval` of the index
  (1 second by default).
- **`refresh=True`**: forces a refresh. It is useful in tests, but it can slow down the cluster under heavy load.
- **`refresh="wait_for"`**: doesn't force a refresh, it waits for the next one. It is a good balance between
  visibility and performance.

`wait_for=True` is the old (deprecated) form of `refresh="wait_for"`. If both are given, `refresh` wins.

<a id="crud-read"></a>
### CRUD: Read

```python
from esorm import ESModel, NotFoundError


class User(ESModel):
    name: str
    age: int


async def get_user(user_id: str):
    try:
        user = await User.get(user_id)
    except NotFoundError:
        print('No such user')
        return
    print(user.name)
```

`get()` raises `esorm.NotFoundError` if the document doesn't exist. `await user.reload()` loads the current
version of a model from ElasticSearch.

<a id="crud-update"></a>
### CRUD: Update

Load the document, change it, and save it:

```python
from esorm import ESModel


class User(ESModel):
    name: str
    age: int


async def update_user(user_id: str):
    user = await User.get(user_id)
    user.name = 'Jane Doe'
    await user.save()
```

If the document was changed by someone else after you loaded it, `save()` raises a conflict error instead of
overwriting the other change. ESORM checks this with the `_primary_term` and `_seq_no` of the document (optimistic
concurrency control), see [Optimistic concurrency control](docs/advanced.md#optimistic-concurrency-control).

<a id="crud-delete"></a>
### CRUD: Delete

```python
from esorm import ESModel


class User(ESModel):
    name: str
    age: int


async def delete_user(user_id: str):
    user = await User.get(user_id)
    await user.delete()
```

`delete()` raises `esorm.NotFoundError` if the document doesn't exist, and it has the same `refresh` parameter as
`save()`.

<a id="bulk-operations"></a>
### Bulk operations

If you create, update or delete many documents, bulk operations are much faster than single operations, because
they send all the changes in one request:

```python
from typing import List
from esorm import ESModel, ESBulk


class User(ESModel):
    name: str
    age: int


async def bulk_create_users():
    async with ESBulk() as bulk:
        # Create or modify models
        for i in range(10):
            user = User(name=f'User {i}', age=i)
            await bulk.save(user)


async def bulk_delete_users(users: List[User]):
    async with ESBulk(refresh="wait_for") as bulk:  # Wait for the next refresh
        for user in users:
            await bulk.delete(user)


async def bulk_create_users_immediate():
    async with ESBulk(refresh=True) as bulk:  # Refresh immediately
        for i in range(10):
            user = User(name=f'User {i}', age=i)
            await bulk.save(user)
```

The operations are collected in the `async with` block, and they are sent when the block ends. If there is an
exception in the block, nothing is sent. `ESBulk` has the same `refresh` parameter as `save()`, see
[Refresh control](#crud-create).

If some operations fail, `esorm.error.BulkError` is raised after the request. Its `failed_operations` attribute
has the details and the model of every failed operation, the other operations are done.

<a id="searching"></a>
### Search

<a id="general-search"></a>
#### General search

The `search` method searches with an ES query, given as a dict. It returns a list of models. With `res_dict=True`
you get a dict instead, where the keys are the ids of the documents: `await User.search(query, res_dict=True)`.

If you need only one result, use `search_one`, it returns a model or `None`.

```python
from esorm import ESModel


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
    # Search for a user named John Doe
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

`search` has `page_size`, `page` and `sort` parameters too, see [Pagination and sorting](#pagination-and-sorting).
Other keyword arguments are passed to the search API of ElasticSearch.

The queries are type checked, because they are `TypedDict`s, so you get IDE autocompletion and type checking. If
you store a query in a variable, annotate it with `ESQuery`, see [Type checking](#type-checking).

<a id="query-builder-intro"></a>
#### Query builder

Instead of nested dicts, you can build queries from classes too. The query builder produces exactly the same query
dict, so the two can be mixed. Compound queries can be built with parameters, or with `with` blocks: the query
clauses created inside a block are added to it:

```python
from esorm.builder import Bool, Match, Range, Term, Exists

with Bool() as q:
    Match('name', 'John')  # directly in the Bool block: must
    with q.filter:
        Range('age', gte=18)
        Term('active', True)
    with q.must_not:
        Exists('deleted_at')

# The same with parameters:
# q = Bool(must=[Match('name', 'John')], filter=[Range('age', gte=18), Term('active', True)],
#          must_not=[Exists('deleted_at')])
```

The search methods accept the builder queries directly (`users = await User.search(q)`), and `q.to_query()`
returns the query dict:

```json
{
  "bool": {
    "must": [{"match": {"name": "John"}}],
    "filter": [{"range": {"age": {"gte": 18}}}, {"term": {"active": true}}],
    "must_not": [{"exists": {"field": "deleted_at"}}]
  }
}
```

Queries can be combined with operators too: `(Term('tag', 'a') | Term('tag', 'b')) & ~Exists('deleted_at')`.
The [Query builder](docs/query_builder.md#query-builder) chapter has all the details.

<a id="search-with-field-value-terms-dictionary-search"></a>
#### Search with field value terms (dictionary search)

`search_by_fields` searches for documents whose fields have the given values. Every field must match: it is a
`bool` query of `match` queries with `operator: and`, so on `text` fields all the words must be in the field. It
has the `res_dict` parameter too, and a `search_one_by_fields` variant.

```python
from esorm import ESModel


class User(ESModel):
    name: str
    age: int


async def search_users():
    # Search for the users who are 18
    users = await User.search_by_fields({'age': 18})
    for user in users:
        print(user.name)
```

`User.all()` returns all documents (the first 10 without `page_size`, like every search).

<a id="count"></a>
#### Count

The `count` method counts the documents of the index, or the documents which match a query:

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
#### Vector search

With [dense_vector](#dense_vector) fields you can search for similar vectors (semantic search). The vectors are
created by an embedding model, which is not part of ESORM.

##### kNN search

The `knn` query finds the `k` nearest neighbors of the query vector:

```python
results = await Document.search({
    "knn": {
        "field": "embedding",
        "query_vector": [0.1, 0.2, ...],  # The query vector
        "k": 10,  # The number of neighbors to return
        "num_candidates": 100  # The number of candidates to consider on each shard
    }
})
```

##### Hybrid search

You can combine vector search with full text search:

```python
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

Here the documents must match the text query, and the ones with similar vectors get a higher score. The query
builder has a `Knn` class too, see [Vector queries](docs/query_builder.md#builder-vector-queries).

<a id="aggregations"></a>
### Aggregations

The `aggregate` method runs aggregations, given as a dict. Its `query` parameter selects which documents are
aggregated. Both are `TypedDict`s, so they are type checked, and IDEs can autocomplete them. If you store them in
variables, annotate them with `ESAggs` and `ESQuery` (see [Type checking](#type-checking)). Sub-aggregations can be
added with the `aggs` key, their results are in the buckets by their names.

```python
from esorm import ESModel
from esorm.aggs import ESAggs
from esorm.query import ESQuery


class User(ESModel):
    name: str
    age: int
    country: str


async def aggregate_avg():
    # The average age of the users
    aggs_def: ESAggs = {
        'avg_age': {
            'avg': {
                'field': 'age'
            }
        }
    }
    aggs = await User.aggregate(aggs_def)
    print(aggs['avg_age']['value'])


async def aggregate_avg_by_country(country='Hungary'):
    # The average age of the users of a country
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
    # The number of users by country
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
    # The average age of the users by country, with a sub-aggregation
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

The search methods have `page`, `page_size` and `sort` parameters:

```python
users = await User.search_by_fields({'age': 18}, page=2, page_size=20, sort='name')
```

Without `page_size` ElasticSearch returns the first 10 results.

The `Pagination` and `Sort` classes do the same, but they wrap the model, so you can set them in one place, and
use the wrapped model as the original one. This is useful in APIs (see
[FastAPI pagination](docs/advanced.md#fastapi-pagination)).

#### Pagination

`Pagination` can call a callback after the search with the total number of matching documents:

```python
from esorm import ESModel, Pagination


class User(ESModel):
    id: int  # This is the document _id
    name: str
    age: int


async def get_users(page=1, page_size=10):

    async def pagination_callback(total: int):
        # E.g. set a response header here
        print(f'Total users: {total}')

    # Create the pagination
    pagination = Pagination(page=page, page_size=page_size, callback=pagination_callback)

    # Then wrap your model with it
    res = await pagination(User).search_by_fields({'age': 18})

    # The result has at most 10 items
    return res
```

The callback must be an async function.

#### Sorting

`Sort` works the same way:

```python
from esorm import ESModel, Sort


class User(ESModel):
    id: int  # This is the document _id
    name: str
    age: int


async def get_users():
    # Create the sort
    sort = Sort(sort=[
        {'age': {'order': 'desc'}},
        {'name': {'order': 'asc'}}
    ])

    # Then wrap your model with it
    res = await sort(User).search_by_fields({'age': 18})

    # The result is sorted by age descending, then by name ascending
    return res


async def get_user_sorted_by_name():
    # A field name sorts in ascending order
    sort = Sort(sort='name')

    res = await sort(User).all()

    # The result is sorted by name ascending
    return res
```

`Pagination` and `Sort` can be used together: `sort(pagination(User))`. If no sort is given, the `default_sort` of
the [model settings](#model-settings) is used.

<a id="iterating-through-all-results"></a>
### Iterating through all results (scan)

Pagination is for showing results to users, and it can't go deeper than 10,000 results by default
(`index.max_result_window`). To process all results of a query (e.g. export, reindex, batch jobs), use `scan()`,
which can iterate through any number of documents efficiently:

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
        # `users` is a list of at most 500 models
        print(len(users))
```

It uses a point in time (PIT) with `search_after`, so the results are consistent: the changes made during the
iteration are not visible. Sorting, the `Sort` decorator, `_source` and shard routing work as in `search()`.

If the iteration may stop early (e.g. by `break`), wrap it in `contextlib.aclosing()` to close the point in time
immediately. More details in [Iterating through large result sets](docs/advanced.md#iterating-through-large-result-sets).

<a id="ide-support"></a>
## 🖥 IDE support and type checking

ESORM has type hints everywhere (it is a [PEP 561](https://peps.python.org/pep-0561/) typed package), so IDEs can
autocomplete and check the models, the queries, the aggregations and the results. The type hints are tested with
[pyright](https://github.com/microsoft/pyright) (used by Pylance in VS Code and Cursor) and
[mypy](https://mypy-lang.org/) in the CI (`tests/typecheck`): correct code must not produce any warnings, and the
typical mistakes must be reported.

<a id="type-checking"></a>
### Type checking of queries and aggregations

Queries and aggregations are `TypedDict`s, so their keys are autocompleted, and typos, wrong value types and
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

Every query clause has the same type: `ESQuery`. It is the type of the whole query, and also of the clauses of
compound queries (`bool`, `nested`, `function_score`, ...). An aggregation is an `ESAgg`, and the aggregations
(and the sub-aggregations) are `ESAggs`.

A query given directly to a method is checked automatically. If you store it in a variable, annotate the
variable, because type checkers infer an unannotated dict literal as a plain `dict`, and they report that it is not
an `ESQuery`:

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
`key`, `doc_count`, ...) have their own types, while every other key is the result of a sub-aggregation:

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
- The `value` of the `avg`, `min` and `max` aggregations is `None` if there are no documents, so its type is
  `Optional[float]`.
- Buckets are typed as a list. With `keyed: True` (or with named filters in a `filters` aggregation) ES returns
  them as a dict, use `typing.cast()` there.
- ES allows only one query type in a query clause (`{'match': ..., 'term': ...}` is invalid). This is not checked,
  use `bool` to combine them.
- The rarely used queries and aggregations (e.g. span queries, `intervals`, `scripted_metric`) accept any dict.
- The query and aggregation types can be used in Pydantic models too, e.g. in a FastAPI request body.

The classes of the [query builder](docs/query_builder.md#query-builder) are ordinary classes with typed parameters,
so every type checker checks them, without the limitations of the `TypedDict` literals. See
[Type checking](docs/query_builder.md#builder-type-checking) in the query builder chapter.

### PyCharm

ESORM is developed in [PyCharm](https://www.jetbrains.com/pycharm/), and the type hints are designed to work with
PyCharm's type checker too (tested with PyCharm 2026.2): correct queries and aggregations are not reported, and the
typical mistakes are. Its limitations:

- It checks the clauses of `bool` queries to two levels of nesting (a `bool` query in a clause of a `bool` query),
  and the sub-aggregations to two levels. The clauses of the other compound queries (e.g. `nested`, `has_child`)
  are not checked. Pyright and mypy check every level.
- If the query of `search()` is a dict literal, PyCharm can't tell whether the result is a list or a dict
  (`res_dict=True`), so it accepts the methods of both (this is a PyCharm bug). The models in the result are typed.

The [Pydantic plugin](https://plugins.jetbrains.com/plugin/12861-pydantic) of PyCharm helps a lot with the
autocompletion and type checking of Pydantic models.

### VS Code / Cursor

In VS Code and Cursor use Pylance (or the pyright extension). Set the type checking mode to at least `basic`,
otherwise Pylance doesn't report type errors. `"python.analysis.typeCheckingMode": "standard"` is recommended.

<a id="testing"></a>
## 🧪 Testing

The tests run on GitHub Actions for every push and pull request, on all supported Python versions (3.10 - 3.14),
against ElasticSearch 8.x, 9.x and OpenSearch 2.x, 3.x.

For local testing there is a `tests.sh` script in the root directory. It runs the tests with more Python
interpreters, in virtual environments. You can set the interpreters at the top of the file. The ES and OpenSearch
versions are in `tests/docker-compose.yml`.

The development dependencies are in the `dev` dependency group of `pyproject.toml`, install them with `uv sync` or
`pip install -e . --group dev` (pip 25.1+).

If you already have a virtual environment, simply run `pytest`. The tests need Docker, they start the ES and
OpenSearch containers automatically. By default all versions are tested, you can choose with the
`ESORM_TEST_SERVICES` environment variable, e.g. `ESORM_TEST_SERVICES=es9x,os3x pytest`
(available services: `es8x`, `es9x`, `os2x`, `os3x`).

The type hints are tested with `python tests/typecheck/check.py` (pyright and mypy, no ES needed). The type checkers
use stubs for the query and aggregation types (`esorm/query.pyi`, `esorm/aggs.pyi`). They are generated by
`python tools/generate_stubs.py`, run it after you change the types in `esorm/query.py` or `esorm/aggs.py`.

<a id="license"></a>
## 🛡 License

This project is licensed under the terms of the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/)
(MPL 2.0).

<a id="citation"></a>
## 📃 Citation

If you use this project in your research, please cite it with the following BibTeX entry:

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
