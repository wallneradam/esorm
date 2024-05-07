from typing import List, Optional

import sys
import os
import asyncio
import subprocess
import importlib

import pytest
import logging

logger = logging.getLogger(__name__)
disable_loggers = ['elastic_transport.node_pool', 'elastic_transport.transport']


#
# Hooks
#

def pytest_configure():
    for logger_name in disable_loggers:
        logging.getLogger(logger_name).disabled = True


@pytest.fixture(scope="class")
def event_loop():
    """
    Use asyncio event loop in class scope
    """
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


#
# Fixtures
#

# noinspection PyProtectedMember
@pytest.fixture(scope="class")
async def esorm():
    """
    Simply import esorm module
    This way it can be removed too
    """
    esorm = importlib.import_module('esorm')
    # It should be empty for every class
    assert len(esorm.model._ESModelMeta.__models__) == 0
    yield esorm
    esorm.model._ESModelMeta.__models__.clear()
    for module_name in list(sys.modules):
        if module_name.startswith('esorm'):
            del sys.modules[module_name]


@pytest.fixture(scope="class")  # Test both ES 8.x and 7.x
def docker_es(service):
    compose_file = os.path.join(os.path.dirname(__file__), 'docker-compose.yml')
    res = subprocess.run(['docker', 'compose', '-f', compose_file, 'up', '-d', service], check=True)
    logger.info(f"Started service: {service}")
    assert res.returncode == 0
    yield
    subprocess.run(['docker', 'compose', '-f', compose_file, 'down', service, '-v'], check=True)
    logger.info(f"Stopped service: {service}")


# noinspection PyUnresolvedReferences
@pytest.fixture(scope="class")
async def es(docker_es, esorm):
    """
    ElasticSearch fixture for version 8.x
    """
    es = await esorm.connect(hosts=["http://localhost:9200"], wait=True)
    assert es is not None
    yield es
    await es.close()


# noinspection PyUnresolvedReferences
@pytest.fixture(scope="class")
def model_python(esorm):
    """
    Model to test python types
    """

    from datetime import datetime, date, time

    class PythonFieldModel(esorm.ESModel):
        f_str: str
        f_int: int
        f_float: float
        f_bool: bool
        f_datetime: datetime
        f_date: date
        f_time: time

    yield PythonFieldModel

    del PythonFieldModel


# noinspection PyUnresolvedReferences
@pytest.fixture(scope="class")
def model_es(esorm):
    """
    Model to test ES types
    """

    class ESFieldModel(esorm.ESModel):
        f_keyword: esorm.fields.keyword
        f_text: esorm.fields.text
        f_binary: esorm.fields.binary
        f_byte: esorm.fields.byte
        f_short: esorm.fields.short
        f_int32: esorm.fields.int32
        f_long: esorm.fields.long
        f_float16: esorm.fields.float16
        f_float32: esorm.fields.float32
        f_double: esorm.fields.double
        f_geo_point: esorm.fields.geo_point

    yield ESFieldModel

    del ESFieldModel


@pytest.fixture(scope="class")
def model_es_optional(esorm):
    """
    Model to test optional ES types
    """

    class ESOptionalFieldModel(esorm.ESModel):
        f_keyword: Optional[esorm.fields.keyword] = None
        f_text: Optional[esorm.fields.text] = None
        f_binary: Optional[esorm.fields.binary] = esorm.Field(None, index=False)
        f_byte: Optional[esorm.fields.byte] = None
        f_short: Optional[esorm.fields.short] = None
        f_int32: Optional[esorm.fields.int32] = None
        f_long: Optional[esorm.fields.long] = None
        f_float16: Optional[esorm.fields.float16] = None
        f_float32: Optional[esorm.fields.float32] = None
        f_double: Optional[esorm.fields.double] = None
        f_geo_point: Optional[esorm.fields.geo_point] = None

    yield ESOptionalFieldModel

    del ESOptionalFieldModel


# noinspection PyUnresolvedReferences
@pytest.fixture(scope="class")
def model_timestamp(esorm):
    """
    Model to test timestamp
    """

    class TimestampModel(esorm.ESModelTimestamp):
        f_str: str
        f_int: int = 1

    yield TimestampModel

    del TimestampModel


# noinspection PyUnresolvedReferences
@pytest.fixture(scope="class")
def model_config(esorm):
    """
    Model to test config
    """

    class ConfigModel(esorm.ESModel):
        class ESConfig:
            index = 'custom_index'
            id_field = 'custom_id'
            default_sort = [{'f_str': {'order': 'desc'}}]
            settings = dict(
                number_of_shards=6,
                number_of_replicas=1,
                refresh_interval="5s",
            )

        custom_id: str
        """ The id of the document """

        f_str: str
        """ Field to test """

        @property
        def __routing__(self) -> str:
            return self.custom_id + '_routing'

    yield ConfigModel

    del ConfigModel


# noinspection PyUnresolvedReferences
@pytest.fixture(scope="class")
def model_with_id(esorm):
    """
    Model to test id field
    """

    class IdModel(esorm.ESModel):
        id: str
        f_str: str

    yield IdModel

    del IdModel


# noinspection PyUnresolvedReferences
@pytest.fixture(scope="class")
def model_with_int_id(esorm):
    """
    Model to test int id field
    """

    class IntIdModel(esorm.ESModel):
        class ESConfig:
            id_field = 'custom_id'

        custom_id: int
        f_str: str

    yield IntIdModel

    del IntIdModel


# noinspection PyUnresolvedReferences
@pytest.fixture(scope="class")
def model_with_prop_id(esorm):
    """
    Model to test property id field
    """

    class PropIdModel(esorm.ESModel):
        @property
        def __id__(self) -> int:
            return self.custom_id + 1000

        custom_id: int
        f_str: str

    yield PropIdModel

    del PropIdModel


# noinspection PyUnresolvedReferences
@pytest.fixture(scope="class")
def model_nested(esorm, model_timestamp):
    """
    Model to test nested fields
    """

    class NestedFieldModel(esorm.ESModel):
        f_nested: model_timestamp
        f_float: float = 0.5

    yield NestedFieldModel

    del NestedFieldModel


# noinspection PyUnresolvedReferences
@pytest.fixture(scope="class")
def model_lazy_prop(esorm):
    """
    Model to test lazy properties
    """

    class LazyPropModel(esorm.ESModel):
        class ESConfig:
            lazy_property_max_recursion_depth = 1

        f_str: str

        # noinspection PyUnresolvedReferences
        @esorm.lazy_property
        async def same_f_strs(self) -> List['LazyPropModel']:
            return await self.search_by_fields({'f_str': self.f_str})

    yield LazyPropModel

    del LazyPropModel


# noinspection PyUnresolvedReferences
@pytest.fixture(scope="class")
async def model_nested_binary(esorm):
    """
    Model to test nested binary fields
    """

    class BinaryModel(esorm.ESBaseModel):
        f_binary: Optional[esorm.fields.binary] = esorm.Field(None, index=False)

    class NestedBinaryModel(esorm.ESModel):
        f_nested: BinaryModel

    await esorm.setup_mappings()

    yield BinaryModel, NestedBinaryModel
    del BinaryModel
    del NestedBinaryModel


@pytest.fixture(scope="class")
async def model_nested_base_model(esorm):
    """
    Model to test nested base model
    """
    from asyncio import sleep

    class NestedBaseModel(esorm.ESBaseModel):
        f_str: str
        f_int: int

        @esorm.lazy_property
        async def lazy_prop(self) -> str:
            await sleep(0.1)
            return f'lazy(f_str={self.f_str}, f_int={self.f_int})'

    class NestedBaseModelModel(esorm.ESModel):
        f_nested: NestedBaseModel

    await esorm.setup_mappings()

    yield NestedBaseModel, NestedBaseModelModel
    del NestedBaseModel
    del NestedBaseModelModel
