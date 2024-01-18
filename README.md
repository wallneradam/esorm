<!--suppress HtmlDeprecatedAttribute -->
<img src="https://raw.githubusercontent.com/wallneradam/esorm/main/docs/_static/img/esorm.svg" width="110" height="110" align="left" style="margin-right: 1em;" alt="Logo"/>

# ESORM - Python ElasticSearch ORM based on Pydantic

<small>Some ideas come from [Pydastic](https://github.com/RamiAwar/pydastic) library, which is similar,
but not as advanced (yet).</small>

## ☰ Table of Contents

- [💾 Installation](#installation)
- [🚀 Features](#features)
    - [Supported ElasticSearch versions](#supported-elasticsearch-versions)
    - [Supported Python versions](#supported-python-versions)
- [📖 Usage](#usage)
    - [Define a model](#define-a-model)
        - [Python basic types](#python-basic-types)
        - [ESORM field types](#esorm-field-types)
        - [Nested documents](#nested-documents)
        - [Id field](#id-field)
        - [Model Settings](#model-settings)
        - [Describe fields](#describe-fields)
        - [ESModelTimestamp](#esmodeltimestamp)
    - [Connecting to ElasticSearch](#connecting-to-elasticsearch)
        - [Client](#client)
    - [Create index templates](#create-index-templates)
    - [Create indices and mappings](#create-indices-and-mappings)
    - [CRUD: Create](#crud-create)
    - [CRUD: Read](#crud-read)
    - [CRUD: Update](#crud-update)
    - [CRUD: Delete](#crud-delete)
    - [Bulk operations](#bulk-operations)
    - [Search](#search)
        - [General search](#general-search)
        - [Search with field value terms (dictioanry search)](#search-with-field-value-terms-dictionary-search)
    - [Aggregations](#aggregations)
- [🔬 Advanced usage](#advanced-usage)
    - [Pagination and sorting](#pagination-and-sorting)
    - [Lazy properties](#lazy-properties)
    - [Shard routing](#shard-routing)
    - [Watchers](#watchers)
    - [FastAPI integration](#fastapi-integration)
    - [Logging](#logging)
- [🧪 Testing](#testing)
- [🛡 License](#license)
- [📃 Citation](#citation)

<a id="installation"></a>
## 💾 Installation


```bash
pip install pyesorm
```

<a id="features"></a>
## 🚀 Features

- Pydantic model representation of ElasticSearch documents
- Automatic mapping and index creation
- CRUD operations
- Full async support (no sync version at all)
- Mapping to and from ElasticSearch types
- Support for nested documents
- Custom id field
- Context for bulk operations
- Supported IDE autocompletion and type checking (PyCharm tested)
- Everything in the source code is documented and annotated
- TypeDicts for ElasticSearch queries and aggregations
- Docstring support for fields
- Shard routing support
- Lazy properties
- Support >= Python 3.8 (tested with 3.8 through 3.12)
- Support for ElasticSearch 8.x and 7.x
- Watcher support (You may need ElasticSearch subscription license for this)
- Pagination and sorting
- FastAPI integration

Not all ElasticSearch features are supported yet, pull requests are welcome.

<a id="supported-elasticsearch-versions"></a>
### Supported ElasticSearch versions

It is tested with ElasticSearch 7.x and 8.x.

<a id="supported-python-versions"></a>
### Supported Python versions

Tested with Python 3.8 through 3.12.

<a id="usage"></a>
## 📖 Usage

<a id="define-a-model"></a>
### Define a model

You can use all [Pydantic](https://pydantic-docs.helpmanual.io/usage/models/) model features, because
`ESModel` is a subclass of `pydantic.BaseModel`.

<a id="python-basic-types"></a>
#### Python basic types

```python
from esorm import ESModel


class User(ESModel):
    name: str
    age: int
```

This is how the python types are converted to ES types:

| Python type         | ES type   |
|---------------------|-----------|
| `str`               | `text`    |
| `int`               | `long`    |
| `float`             | `double`  |
| `bool`              | `boolean` |
| `datetime.datetime` | `date`    |
| `datetime.date`     | `date`    |
| `datetime.time`     | `date`    |

<a id="esorm-field-types"></a>
#### ESORM field types

You can specify ElasticSearch special fields using `esorm.fields` module.

```python
from esorm import ESModel
from esorm.fields import keyword, text, byte, geo_point


class User(ESModel):
    name: text
    email: keyword
    age: byte
    location: geo_point
    ...
```

The supported fields are:

| Field name                | ES type      |
|---------------------------|--------------|
| `keyword`                 | `keyword`    |
| `text`                    | `text`       |
| `binary`                  | `binary`     |
| `byte`                    | `byte`       |
| `short`                   | `short`      |
| `integer` or `int32`      | `integer`    |
| `long` or `int64`         | `long`       |
| `float16` or `half_float` | `half_float` |
| `float32`                 | `float`      |
| `double`                  | `double`     |
| `boolean`                 | `boolean`    |
| `geo_point`               | `geo_point`  |

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
    async with ESBulk() as bulk:
        # Deleting models
        for user in users:
            await bulk.delete(user)
```

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

Queries are type checked, because they are annotated as TypeDicts. You can use IDE autocompletion and type checking.

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

<a id="aggregations"></a>
### Aggregations

You can use `aggregate` method to get aggregations. 
You can specify an ES aggregation query as a dictionary. It also accepts normal ES queries,
to be able to fiter which documents you want to aggregate. 
Both the aggs parameter and the query parameter are type checked, because they are annotated as TypeDicts.
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

<a id="advanced-usage"></a>
## 🔬 Advanced usage

TODO...
These features may not documented yet, but working.

<a id="pagination-and-sorting"></a>
### Pagination and sorting

<a id="lazy-properties"></a>
### Lazy properties

<a id="shard-routing"></a>
### Shard routing

<a id="watchers"></a>
### Watchers

<a id="fastapi-integration"></a>
### FastAPI integration

<a id="logging"></a>
### Logging

<a id="testing"></a>
## 🧪 Testing

For testing you can use the `test.sh` in the root directory. It is a script to running
tests on multiple python interpreters in virtual environments. At the top of the file you can specify
which python interpreters you want to test. The ES versions are specified in `tests/docker-compose.yml` file.

If you already have a virtual environment, simply use `pytest` to run the tests.

<a id="license"></a>
## 🛡 License

This project is licensed under the terms of the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/) (
MPL 2.0) license.

<a id="citation"></a>
## 📃 Citation

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
