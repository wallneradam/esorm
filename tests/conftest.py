import sys
import os
import asyncio
import pytest
import subprocess
import importlib

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
    return asyncio.get_event_loop()


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
    esorm_model = importlib.import_module('esorm.model')
    # It should be empty for every class
    assert len(esorm_model._ESModelMeta.__models__) == 0
    yield esorm
    for module_name in list(sys.modules):
        if module_name.startswith('esorm'):
            del sys.modules[module_name]


@pytest.fixture(scope="class")  # Test both ES 8.x and 7.x
async def docker_es(service):
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
    assert es


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

    from datetime import datetime, date, time

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
                number_of_shards=3,
                number_of_replicas=1,
                refresh_interval="5s",
            )

        custom_id: str
        """ The id of the document """

        f_str: str
        """ Field to test """

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
def model_nested(esorm, model_timestamp):
    """
    Model to test nested fields
    """

    class NestedFieldModel(esorm.ESModel):
        f_nested: model_timestamp
        f_float: float = 0.5

    yield NestedFieldModel

    del NestedFieldModel
