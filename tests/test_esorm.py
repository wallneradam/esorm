import asyncio
from typing import TYPE_CHECKING
import os
import sys
import pytest

# ES and OpenSearch services (from docker-compose.yml) to test against, can be limited by
# e.g. `ESORM_TEST_SERVICES=es8x,es9x`
services = os.environ.get('ESORM_TEST_SERVICES', 'es8x,es9x,os2x,os3x').split(',')


@pytest.mark.parametrize('service', services, scope='class')
class TestBaseTests:
    """
    Base tests
    """

    doc_id: str = None

    async def test_connect(self, es):
        """
        Test connection
        """
        assert es is not None

    async def test_create_basemodel(self, es, esorm):
        """
        Test create base model
        """

        class BaseModel(esorm.ESBaseModel):
            f_str: str
            f_int: int

        class BaseChild(BaseModel):
            f_float: float

        # Basemodels should not be registered as models they should not be created in the DB as indexes
        # TODO: This is not working in the 2nd run, this is only test issue
        # assert len(esorm.model._ESModelMeta.__models__) == 0
        assert not hasattr(BaseModel.ESConfig, 'index')
        assert not hasattr(BaseChild.ESConfig, 'index')

    async def test_create_model_with_python_fields(self, es, esorm, model_python):
        """
        Test model with python fields
        """
        assert model_python is not None
        assert model_python.ESConfig.index == 'esorm_python_field_model'

    async def test_create_model_with_es_fields(self, es, esorm, model_es):
        """
        Test model with ES fields
        """
        assert model_es is not None
        assert model_es.ESConfig.index == 'esorm_es_field_model'

    async def test_create_model_with_es_optional_fields(self, es, esorm, model_es_optional):
        """
        Test model with ES fields
        """
        assert model_es_optional is not None
        assert model_es_optional.ESConfig.index == 'esorm_es_optional_field_model'

    async def test_create_model_with_es_optional_fields_new_syntax(self, es, esorm, model_es_optional_new_syntax):
        """
        Test model with ES fields with new syntax
        """
        assert model_es_optional_new_syntax is not None
        assert model_es_optional_new_syntax.ESConfig.index == 'esorm_es_optional_field_new_syntax_model'

    async def test_create_timestamp_models(self, es, esorm, model_timestamp):
        """
        Test timestamp models
        """
        assert model_timestamp is not None
        assert model_timestamp.ESConfig.index == 'esorm_timestamp_model'
        assert 'created_at' in model_timestamp.model_fields
        assert 'modified_at' in model_timestamp.model_fields

    async def test_create_model_config(self, es, esorm, model_config):
        """
        Test model config
        """
        assert model_config is not None
        assert model_config.ESConfig.index == 'custom_index'

    async def test_model_template(self, es, esorm, model_index_template):
        """
        Test model config
        """
        assert model_index_template is not None
        assert model_index_template.ESConfig.index == 'custom_index_template'

    async def test_create_model_with_id(self, es, esorm, model_with_id):
        """
        Test model with config
        """
        assert model_with_id is not None
        assert model_with_id.ESConfig.index == 'esorm_id_model'
        assert model_with_id.ESConfig.id_field == 'id'

    async def test_create_model_with_int_id(self, es, esorm, model_with_int_id):
        """
        Test model with int id
        """
        assert model_with_int_id is not None
        assert model_with_int_id.ESConfig.index == 'esorm_int_id_model'
        assert model_with_int_id.ESConfig.id_field == 'custom_id'

    async def test_create_model_with_prop_id(self, es, esorm, model_with_prop_id):
        """
        Test model with property id
        """
        assert model_with_prop_id is not None
        assert model_with_prop_id.ESConfig.index == 'esorm_prop_id_model'
        assert model_with_prop_id.ESConfig.id_field is None

    async def test_create_nested_model(self, es, esorm, model_nested):
        """
        Test nested model
        """
        assert model_nested is not None
        assert model_nested.ESConfig.index == 'esorm_nested_field_model'

    async def test_create_mappings(self, es, esorm, model_python, model_es,
                                   model_es_optional, model_es_optional_new_syntax,
                                   model_timestamp, model_config, model_with_id, model_with_int_id,
                                   model_with_prop_id, model_nested, model_lazy_prop,
                                   model_index_template):
        """
        Test create mappings
        """
        # Create index template
        await esorm.model.create_index_template('custom_index_template', prefix_name='custom_index_',
                                                shards=5, auto_expand_replicas="1-2")
        await esorm.setup_mappings()

        # Check if index exists
        assert await es.indices.exists(index=model_python.ESConfig.index)

        # Check if mappings are correct for python fields
        mappings = await es.indices.get_mapping(index=model_python.ESConfig.index)
        assert mappings[model_python.ESConfig.index]['mappings']['properties']['f_str']['type'] == 'keyword'
        assert mappings[model_python.ESConfig.index]['mappings']['properties']['f_int']['type'] == 'long'
        assert mappings[model_python.ESConfig.index]['mappings']['properties']['f_float']['type'] == 'double'
        assert mappings[model_python.ESConfig.index]['mappings']['properties']['f_bool']['type'] == 'boolean'
        assert mappings[model_python.ESConfig.index]['mappings']['properties']['f_datetime']['type'] == 'date'
        assert mappings[model_python.ESConfig.index]['mappings']['properties']['f_date']['type'] == 'date'
        assert mappings[model_python.ESConfig.index]['mappings']['properties']['f_time']['type'] == 'date'
        assert mappings[model_python.ESConfig.index]['mappings']['properties']['f_literal']['type'] == 'keyword'
        assert mappings[model_python.ESConfig.index]['mappings']['properties']['f_uuid4']['type'] == 'keyword'
        assert mappings[model_python.ESConfig.index]['mappings']['properties']['f_file_path']['type'] == 'keyword'
        assert mappings[model_python.ESConfig.index]['mappings']['properties']['f_http_url']['type'] == 'keyword'

        # Check if mappings are correct for ES fields
        mappings = await es.indices.get_mapping(index=model_es.ESConfig.index)
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_keyword']['type'] == 'keyword'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_text']['type'] == 'text'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_binary']['type'] == 'binary'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_byte']['type'] == 'byte'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_short']['type'] == 'short'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_int32']['type'] == 'integer'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_long']['type'] == 'long'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_unsigned_long']['type'] == 'unsigned_long'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_float16']['type'] == 'half_float'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_float32']['type'] == 'float'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_double']['type'] == 'double'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_geo_point']['type'] == 'geo_point'

        # Check if mappings are correct for ES Optional fields
        mappings = await es.indices.get_mapping(index=model_es_optional.ESConfig.index)
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_keyword']['type'] == 'keyword'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_text']['type'] == 'text'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_binary']['type'] == 'binary'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_byte']['type'] == 'byte'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_short']['type'] == 'short'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_int32']['type'] == 'integer'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_long']['type'] == 'long'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_unsigned_long'][
                   'type'] == 'unsigned_long'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_float16']['type'] == 'half_float'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_float32']['type'] == 'float'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_double']['type'] == 'double'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_geo_point'][
                   'type'] == 'geo_point'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_positive_int']['type'] == 'long'

        # Test optional new syntax
        mappings = await es.indices.get_mapping(index=model_es_optional_new_syntax.ESConfig.index)
        assert mappings[model_es_optional_new_syntax.ESConfig.index]['mappings']['properties']['f_keyword'][
                   'type'] == 'keyword'
        assert mappings[model_es_optional_new_syntax.ESConfig.index]['mappings']['properties']['f_text'][
                   'type'] == 'text'
        assert mappings[model_es_optional_new_syntax.ESConfig.index]['mappings']['properties']['f_binary'][
                   'type'] == 'binary'
        assert mappings[model_es_optional_new_syntax.ESConfig.index]['mappings']['properties']['f_byte'][
                   'type'] == 'byte'
        assert mappings[model_es_optional_new_syntax.ESConfig.index]['mappings']['properties']['f_short'][
                   'type'] == 'short'
        assert mappings[model_es_optional_new_syntax.ESConfig.index]['mappings']['properties']['f_int32'][
                   'type'] == 'integer'
        assert mappings[model_es_optional_new_syntax.ESConfig.index]['mappings']['properties']['f_long'][
                   'type'] == 'long'
        assert mappings[model_es_optional_new_syntax.ESConfig.index]['mappings']['properties']['f_unsigned_long'][
                   'type'] == 'unsigned_long'
        assert mappings[model_es_optional_new_syntax.ESConfig.index]['mappings']['properties']['f_float16'][
                   'type'] == 'half_float'
        assert mappings[model_es_optional_new_syntax.ESConfig.index]['mappings']['properties']['f_float32'][
                   'type'] == 'float'
        assert mappings[model_es_optional_new_syntax.ESConfig.index]['mappings']['properties']['f_double'][
                   'type'] == 'double'
        assert mappings[model_es_optional_new_syntax.ESConfig.index]['mappings']['properties']['f_geo_point'][
                   'type'] == 'geo_point'

        # Check if mappings are correct for timestamp fields
        mappings = await es.indices.get_mapping(index=model_timestamp.ESConfig.index)
        assert mappings[model_timestamp.ESConfig.index]['mappings']['properties']['created_at']['type'] == 'date'
        assert mappings[model_timestamp.ESConfig.index]['mappings']['properties']['modified_at']['type'] == 'date'

        # Check if mappings are correct for model with config
        mappings = await es.indices.get_mapping(index=model_config.ESConfig.index)
        assert mappings[model_config.ESConfig.index]['mappings']['properties']['f_str']['type'] == 'keyword'
        # Id should not be in mappings
        assert 'custom_id' not in mappings[model_config.ESConfig.index]['mappings']['properties'], \
            "Id fields should not be in mappings"
        # Check index settings
        settings = await es.indices.get_settings(index=model_config.ESConfig.index)
        assert settings[model_config.ESConfig.index]['settings']['index']['number_of_shards'] == '6'
        assert settings[model_config.ESConfig.index]['settings']['index']['number_of_replicas'] == '1'
        assert settings[model_config.ESConfig.index]['settings']['index']['refresh_interval'] == '5s'

        # Check if mappings are correct for model with index template
        mappings = await es.indices.get_mapping(index=model_index_template.ESConfig.index)
        assert mappings[model_index_template.ESConfig.index]['mappings']['properties']['f_str']['type'] == 'keyword'
        # Check index settings
        settings = await es.indices.get_settings(index=model_index_template.ESConfig.index)
        assert settings[model_index_template.ESConfig.index]['settings']['index']['number_of_shards'] == '5'
        assert settings[model_index_template.ESConfig.index]['settings']['index']['auto_expand_replicas'] == '1-2'

        # Check if mappings are correct for model with id
        mappings = await es.indices.get_mapping(index=model_with_id.ESConfig.index)
        assert mappings[model_with_id.ESConfig.index]['mappings']['properties']['f_str']['type'] == 'keyword'
        assert 'id' not in mappings[model_with_id.ESConfig.index]['mappings']['properties'], \
            "Id fields should not be in mappings"

        # Check if mappings are correct for model with int id
        mappings = await es.indices.get_mapping(index=model_with_int_id.ESConfig.index)
        assert mappings[model_with_int_id.ESConfig.index]['mappings']['properties']['f_str']['type'] == 'keyword'
        assert 'custom_id' not in mappings[model_with_int_id.ESConfig.index]['mappings']['properties'], \
            "Id fields should not be in mappings"

        # Check if mappings are correct for model with prop id
        mappings = await es.indices.get_mapping(index=model_with_prop_id.ESConfig.index)
        assert mappings[model_with_prop_id.ESConfig.index]['mappings']['properties']['f_str']['type'] == 'keyword'
        # It is used in the id, but remains in the db
        assert mappings[model_with_prop_id.ESConfig.index]['mappings']['properties']['custom_id']['type'] == 'long'

        # Check if mappings are correct for nested model
        mappings = await es.indices.get_mapping(index=model_nested.ESConfig.index)
        assert 'properties' in mappings[model_nested.ESConfig.index]['mappings']['properties']['f_nested'], \
            "Nested field should have properties"
        assert 'created_at' in mappings[model_nested.ESConfig.index]['mappings']['properties']['f_nested'][
            'properties']
        assert 'modified_at' in mappings[model_nested.ESConfig.index]['mappings']['properties']['f_nested'][
            'properties']
        assert 'f_str' in mappings[model_nested.ESConfig.index]['mappings']['properties']['f_nested'][
            'properties']

    async def test_crud_create(self, es, esorm, model_nested, model_timestamp):
        """
        Test create
        """
        # It should have a ES generated document id
        doc_id = await model_nested(f_nested=model_timestamp(f_str="nested_test")).save()
        assert doc_id is not None and len(doc_id) == 20, "Incorrect document id"
        self.__class__.doc_id = doc_id  # Strange but it works

        # Create some other documents
        doc_id = await model_nested(f_nested=model_timestamp(f_str="nested_test2", f_int=2), f_float=1.0).save()
        assert doc_id is not None and len(doc_id) == 20, "Incorrect document id"
        doc_id = await model_nested(f_nested=model_timestamp(f_str="nested_test3", f_int=3), f_float=1.5).save()
        assert doc_id is not None and len(doc_id) == 20, "Incorrect document id"
        doc_id = await model_nested(f_nested=model_timestamp(f_str="nested_test4", f_int=4), f_float=2.0).save()
        assert doc_id is not None and len(doc_id) == 20, "Incorrect document id"
        doc_id = await model_nested(f_nested=model_timestamp(f_str="nested_test5", f_int=5), f_float=2.5).save()
        assert doc_id is not None and len(doc_id) == 20, "Incorrect document id"

    async def test_crud_get_by_id(self, es, esorm, model_nested, model_timestamp):
        """
        Test get by id
        """
        doc_id = self.__class__.doc_id
        # Get by id
        doc = await model_nested.get(doc_id)
        assert doc is not None
        assert doc.f_float == 0.5  # Default value of the field
        assert doc.f_nested is not None
        assert doc.f_nested.f_str == "nested_test"

        # Check private fields
        assert doc._version == 1
        assert doc._primary_term == 1
        assert doc._seq_no == 0

    async def test_crud_get_by_int_id(self, es, esorm, model_with_int_id):
        """
        Test get by int id
        """
        # Create a document
        doc_id = await model_with_int_id(custom_id=1, f_str="int_id_test").save()
        assert doc_id == "1"

        # Get by id
        doc = await model_with_int_id.get(1)
        assert doc is not None
        assert doc.custom_id == 1
        assert doc.f_str == "int_id_test"
        assert doc._id == "1"

    async def test_crud_get_by_prop_id(self, es, esorm, model_with_prop_id):
        """
        Test get by prop id
        """
        # Create a document
        doc_id = await model_with_prop_id(custom_id=1, f_str="prop_id_test").save()
        assert doc_id == "1001"

        # Get by id
        doc = await model_with_prop_id.get(1001)
        assert doc is not None
        assert doc.custom_id == 1
        assert doc.f_str == "prop_id_test"
        assert doc._id == "1001"

    async def test_crud_update(self, es, esorm, model_nested):
        """
        Test update
        """
        doc_id = self.__class__.doc_id
        # Update
        doc = await model_nested.get(doc_id)
        modified_at = doc.f_nested.modified_at
        created_at = doc.f_nested.created_at
        assert doc is not None
        # Nested created_at is set on creation
        assert created_at is not None
        doc.f_nested.f_str = "nested_test_updated"
        await doc.save()

        await asyncio.sleep(0.1)

        # Get by id
        doc = await model_nested.get(doc_id)
        assert doc is not None
        assert doc.f_nested.f_str == "nested_test_updated"
        # Test if modified_at is updated even in nested fields
        assert doc.f_nested.modified_at > modified_at
        # Test if created_at is not updated
        assert doc.f_nested.created_at == created_at

        # Check private fields
        assert doc._version == 2
        assert doc._primary_term == 1
        assert doc._seq_no == 5  # It is 5 because other documents have been created

    async def test_python_types(self, es, esorm, model_python):
        """
        Test pydantic types
        """
        from datetime import datetime, date, time
        from uuid import uuid4, UUID

        doc = model_python(
            f_str='test', f_int=123, f_float=0.123, f_bool=True,
            f_datetime=datetime.now(),
            f_date=date.today(), f_time=datetime.now().time(),
            f_literal='a',
            f_uuid4=uuid4(),
            f_file_path='/bin/sh',
            f_http_url='http://example.com'
        )
        doc_id = await doc.save()
        assert doc_id is not None

        # Get by id
        doc = await model_python.get(doc_id)
        assert doc is not None
        assert doc.f_str == 'test'
        assert doc.f_int == 123
        assert doc.f_float == 0.123
        assert doc.f_bool is True
        assert isinstance(doc.f_datetime, datetime)
        assert isinstance(doc.f_date, date)
        assert isinstance(doc.f_time, time)
        assert doc.f_literal == 'a'
        assert isinstance(doc.f_uuid4, UUID)
        assert str(doc.f_file_path) == '/bin/sh'
        assert str(doc.f_http_url) == 'http://example.com/'

    async def test_timezones(self, es, esorm):
        """
        Test saving and loading naive, aware UTC and aware non-UTC datetimes and times
        """
        from datetime import datetime, time, timezone, timedelta
        from esorm.builder import Range

        utc = timezone.utc
        cest = timezone(timedelta(hours=2))

        class TimezoneModel(esorm.ESModel):
            f_name: str
            f_datetime: datetime
            f_time: time

        class TimezoneTimestampModel(esorm.ESModelTimestamp):
            f_str: str

        await esorm.setup_mappings()

        # All of them are 2024-01-01 12:00 UTC (and 12:00 UTC time)
        values = {
            'naive': (datetime(2024, 1, 1, 12), time(12)),
            'utc': (datetime(2024, 1, 1, 12, tzinfo=utc), time(12, tzinfo=utc)),
            'cest': (datetime(2024, 1, 1, 14, tzinfo=cest), time(14, tzinfo=cest)),
        }
        ids = {}
        for name, (f_datetime, f_time) in values.items():
            ids[name] = await TimezoneModel(f_name=name, f_datetime=f_datetime, f_time=f_time).save(refresh=True)

        for name, doc_id in ids.items():
            # Stored as UTC
            res = await TimezoneModel.call('get', id=doc_id)
            assert res['_source']['f_datetime'] == '2024-01-01T12:00:00Z', name
            assert res['_source']['f_time'] == '1970-01-01T12:00:00Z', name

            # Loaded back as aware UTC datetime and naive UTC time
            doc = await TimezoneModel.get(doc_id)
            assert doc.f_datetime == datetime(2024, 1, 1, 12, tzinfo=utc), name
            assert doc.f_datetime.utcoffset() == timedelta(0), name
            assert doc.f_time == time(12), name
            assert doc.f_time.tzinfo is None, name

        # Queries find all of them by the same instant, whatever timezone it is given in
        for instant in (datetime(2024, 1, 1, 12), datetime(2024, 1, 1, 12, tzinfo=utc),
                        datetime(2024, 1, 1, 14, tzinfo=cest)):
            docs = await TimezoneModel.search(Range('f_datetime', gte=instant, lte=instant))
            assert sorted(doc.f_name for doc in docs) == ['cest', 'naive', 'utc'], instant

        # Timestamps of ESModelTimestamp are stored in the same format by save() and by bulk save
        doc_id = await TimezoneTimestampModel(f_str='save').save(refresh=True)
        async with esorm.ESBulk(refresh=True) as bulk:
            await bulk.save(TimezoneTimestampModel(f_str='bulk'))
        bulk_doc = await TimezoneTimestampModel.search_one_by_fields({'f_str': 'bulk'})
        assert bulk_doc is not None

        for doc_id in (doc_id, bulk_doc.__id__):
            res = await TimezoneTimestampModel.call('get', id=doc_id)
            source = res['_source']
            assert source['modified_at'].endswith('Z')
            assert '+' not in source['modified_at']
            assert source['created_at'] == source['modified_at']
            doc = await TimezoneTimestampModel.get(doc_id)
            assert doc.modified_at.utcoffset() == timedelta(0)
            # Loaded timestamps can be compared to the current UTC time
            assert doc.modified_at <= esorm.utils.utcnow()

    async def test_time_fields(self, es, esorm):
        """
        Test saving and loading optional, list and nested time fields
        """
        from datetime import time, timezone, timedelta
        from typing import Optional, List

        cest = timezone(timedelta(hours=2))

        class TimeSubModel(esorm.ESBaseModel):
            f_time: time
            f_optional_time: Optional[time] = None

        class TimeFieldsModel(esorm.ESModel):
            f_optional_time: Optional[time] = None
            f_union_time: time | None = None
            f_list_time: List[time] = []
            f_nested: Optional[TimeSubModel] = None
            f_nested_list: List[TimeSubModel] = []

        await esorm.setup_mappings()

        doc = TimeFieldsModel(
            f_optional_time=time(10, tzinfo=cest),
            f_union_time=time(11),
            f_list_time=[time(12), time(13)],
            f_nested=TimeSubModel(f_time=time(14), f_optional_time=time(15)),
            f_nested_list=[TimeSubModel(f_time=time(16)), TimeSubModel(f_time=time(17), f_optional_time=time(18))],
        )
        doc_id = await doc.save(refresh=True)

        res = await TimeFieldsModel.call('get', id=doc_id)
        assert res['_source']['f_optional_time'] == '1970-01-01T08:00:00Z'
        assert res['_source']['f_nested_list'][1]['f_optional_time'] == '1970-01-01T18:00:00Z'

        for loaded in (await TimeFieldsModel.get(doc_id),
                       await TimeFieldsModel.search_one_by_fields({'_id': doc_id})):
            assert loaded is not None
            assert loaded.f_optional_time == time(8)
            assert loaded.f_union_time == time(11)
            assert loaded.f_list_time == [time(12), time(13)]
            assert loaded.f_nested == doc.f_nested
            assert loaded.f_nested_list == doc.f_nested_list

        # None values stay None
        doc_id = await TimeFieldsModel(f_nested=TimeSubModel(f_time=time(12))).save(refresh=True)
        loaded = await TimeFieldsModel.get(doc_id)
        assert loaded.f_optional_time is None
        assert loaded.f_union_time is None
        assert loaded.f_nested.f_optional_time is None

        # reload() converts the time fields, and updates the fields not set explicitly too
        local = TimeFieldsModel(f_nested=TimeSubModel(f_time=time(12)))
        doc_id = await local.save(refresh=True)
        other = await TimeFieldsModel.get(doc_id)
        other.f_optional_time = time(9)
        other.f_nested.f_optional_time = time(10)
        await other.save(refresh=True)
        await local.reload()
        assert local.f_optional_time == time(9)
        assert isinstance(local.f_nested, TimeSubModel)
        assert local.f_nested.f_optional_time == time(10)

    async def test_race_condition(self, es, esorm, model_nested):
        """
        Test race condition
        Race conditions should raise a ConflictError
        """
        from elasticsearch import ConflictError

        ### Test update ###

        doc_id = self.__class__.doc_id
        # Get 2 instances of the same document
        doc1 = await model_nested.get(doc_id)
        doc1.f_float = 1.0
        doc2 = await model_nested.get(doc_id)
        doc1.f_float = 2.0

        # The 2nd save should not work
        with pytest.raises(ConflictError):
            await doc1.save()
            await doc2.save()

        ### Test delete ###

        # Get 2 instances of the same document
        doc1 = await model_nested.get(doc_id)
        doc1.f_float = 3.0
        doc2 = await model_nested.get(doc_id)

        # Update and delete at the same time
        with pytest.raises(ConflictError):
            await doc1.save(wait_for=True)
            await doc2.delete()

        doc = await model_nested.get(doc_id)
        assert doc is not None
        assert doc.f_float == 3.0

    async def test_retry_on_conflict(self, es, esorm, model_timestamp):
        """
        Test optimistic concurrency control with retry
        """
        import asyncio
        from elasticsearch import ConflictError

        doc = model_timestamp(f_str="occ_retry", f_int=10)
        await doc.save()

        @esorm.retry_on_conflict(3)
        async def update_doc(doc_id):
            _doc = await model_timestamp.get(doc_id)
            _doc.f_int += 1
            await _doc.save()

        await asyncio.gather(
            update_doc(doc._id),
            update_doc(doc._id),
            update_doc(doc._id),
        )

        await doc.reload()
        assert doc.f_int == 13

        doc = model_timestamp(f_str="occ_retry", f_int=10)
        await doc.save()

        # Test it without retry logic

        async def update_without_retry(doc_id):
            _doc = await model_timestamp.get(doc_id)
            _doc.f_int += 1
            await _doc.save()

        with pytest.raises(ConflictError):
            await asyncio.gather(
                update_without_retry(doc._id),
                update_without_retry(doc._id),
                update_without_retry(doc._id),
            )

    async def test_retry_on_conflict_reload(self, es, esorm, model_timestamp):
        import asyncio
        from elasticsearch import ConflictError

        class RetryTestModel(model_timestamp):
            @esorm.retry_on_conflict(6)
            async def update(self):
                self.f_int += 1
                await self.save()

            @esorm.retry_on_conflict(3, reload_on_conflict=False)
            async def update_without_retry(self):
                self.f_int += 1
                await self.save()

        await esorm.setup_mappings()

        doc = RetryTestModel(f_str="occ_retry", f_int=10)
        await doc.save()

        doc1 = await RetryTestModel.get(doc._id)
        doc2 = await RetryTestModel.get(doc._id)
        doc3 = await RetryTestModel.get(doc._id)
        doc4 = await RetryTestModel.get(doc._id)
        doc5 = await RetryTestModel.get(doc._id)
        doc6 = await RetryTestModel.get(doc._id)

        await asyncio.gather(
            doc1.update(),  # 11
            doc2.update(),  # 12
            doc3.update(),  # 13
            doc4.update(),  # 14
            doc5.update(),  # 15
            doc6.update(),  # 16
        )

        await doc.reload()
        assert doc.f_int == 16

        # Try without reload

        with pytest.raises(ConflictError):
            await asyncio.gather(
                doc1.update_without_retry(),
                doc2.update_without_retry(),
                doc3.update_without_retry(),
            )

    # noinspection PyBroadException
    async def test_crud_delete(self, es, esorm, model_nested):
        """
        Test delete
        """
        doc_id = self.__class__.doc_id
        # Delete
        doc = await model_nested.get(doc_id)
        assert doc is not None
        await doc.delete()

        # Get by id
        try:
            await model_nested.get(doc_id)
            assert False, "Document should not exist"
        except Exception as e:
            assert e.__class__.__name__ == 'NotFoundError', "Incorrect exception"
            self.__class__.doc_id = None  # noqa

    async def test_bulk_operations(self, es, esorm, model_nested, model_timestamp):
        """
        Test bulk operations
        """
        # Creating documents
        async with esorm.ESBulk(wait_for=True) as bulk:  # Here wait_for is important!
            for i in range(10):
                doc = model_nested(f_nested=model_timestamp(f_str=f"nested_test1{i}", f_int=10 + i), f_float=10.0 + i)
                await bulk.save(doc)

        # Deleting documents
        async with esorm.ESBulk(wait_for=True) as bulk:  # Here wait_for is important!
            for i in range(10):
                doc = await model_nested.search_one_by_fields({'f_nested.f_str': f"nested_test1{i}"})
                assert doc is not None
                await bulk.delete(doc)

    async def test_race_condition_bulk(self, es, esorm, model_timestamp):
        """
        Test race condition with bulk operations
        Race conditions should raise a ConflictError
        """

        class BulkTestModel(model_timestamp):
            ...

        await esorm.setup_mappings()

        doc = BulkTestModel(f_str="occ", f_int=10)
        await doc.save()

        # Get 2 instances of the same document
        doc1 = await BulkTestModel.get(doc._id)
        doc1.f_int = 1
        doc2 = await BulkTestModel.get(doc._id)
        doc2.f_int = 2

        # Save the 1st instance
        assert doc1._version == 1
        assert doc1._seq_no == 0
        await doc1.save()
        assert doc1._version == 2
        assert doc1._seq_no == 1

        # Save the 2nd in a bulk operation should raise a conflict
        assert doc2._version == 1
        assert doc2._seq_no == 0
        with pytest.raises(esorm.error.BulkError):
            try:
                async with esorm.ESBulk(wait_for=True) as bulk:
                    await bulk.save(doc2)
            except esorm.error.BulkError as e:
                assert len(e.failed_operations) == 1
                assert e.failed_operations[0]['type'] == 'version_conflict_engine_exception'
                raise e

        doc = await BulkTestModel.get(doc._id)
        assert doc.f_int == 1
        assert doc._version == 2
        assert doc._seq_no == 1

    async def test_bulk_race_condition_and_reload(self, es, esorm, model_nested, model_timestamp):
        """
        Test bulk operations with conflict
        """
        doc = model_nested(f_nested=model_timestamp(f_str=f"nested_test1"), f_float=10.0)
        await doc.save(refresh=True)

        with pytest.raises(esorm.error.BulkError):
            async with esorm.ESBulk(wait_for=True) as bulk:
                doc1 = await model_nested.search_one_by_fields({'f_nested.f_str': f"nested_test1"})
                assert doc1._version == 1
                doc2 = await model_nested.search_one_by_fields({'f_nested.f_str': f"nested_test1"})
                assert doc2._version == 1
                assert doc1 is not None
                assert doc2 is not None
                doc1.f_float = 11.0
                doc1.f_nested.f_str = "nested_test1_updated1"
                await bulk.save(doc1)
                # This should not work, because this is a race condition
                doc2.f_float = 11.1
                doc1.f_nested.f_str = "nested_test1_updated2"
                await bulk.save(doc2)

        # We test reloading here, it should now update the _primary_term and _seq_no
        await doc.reload()
        # So the deletion of document should be ok
        await doc.delete(refresh=True)

    async def test_search(self, es, esorm, model_nested):
        """
        Test search
        """
        if TYPE_CHECKING:
            from esorm.query import ESQuery

        # Search
        query: 'ESQuery' = {
            'bool': {
                'must': [{
                    'match': {
                        'f_nested.f_str': 'nested_test2'
                    }
                }]
            }
        }
        docs = await model_nested.search(query)
        assert len(docs) == 1
        assert docs[0].f_nested.f_str == 'nested_test2'

        # Should include private fields
        assert docs[0]._version == 1
        assert docs[0]._primary_term == 1
        assert docs[0]._seq_no > 0

        # Search all started with nested_test
        query: 'ESQuery' = {
            'bool': {
                'must': [{
                    'prefix': {
                        'f_nested.f_str': 'nested_test'
                    }
                }]
            }
        }
        docs = await model_nested.search(query)
        assert len(docs) == 4
        for doc in docs:
            assert doc.f_nested.f_str.startswith('nested_test')

    async def test_search_one(self, es, esorm, model_nested):
        """
        Test search one
        """
        if TYPE_CHECKING:
            from esorm.query import ESQuery

        # Search one
        query: 'ESQuery' = {
            'bool': {
                'must': [{
                    'match': {
                        'f_nested.f_str': 'nested_test2'
                    }
                }]
            }
        }
        doc = await model_nested.search_one(query)
        assert doc is not None
        assert doc.f_nested.f_str == 'nested_test2'

        # Search one with multiple results
        query: 'ESQuery' = {
            'bool': {
                'must': [{
                    'prefix': {
                        'f_nested.f_str': 'nested_test'
                    }
                }]
            }
        }
        doc = await model_nested.search_one(query)
        assert doc is not None

    async def test_search_by_fields(self, es, esorm, model_nested):
        """
        Test search by fields
        """
        # Search by fields
        docs = await model_nested.search_by_fields({
            'f_nested.f_str': 'nested_test2'
        })
        assert len(docs) == 1
        assert docs[0].f_nested.f_str == 'nested_test2'

        # Search by fields with multiple results
        doc = await model_nested.search_one_by_fields({
            'f_nested.f_str': 'nested_test3'
        })
        assert doc is not None
        assert doc.f_nested.f_str == 'nested_test3'

    async def test_basic_aggregations(self, es, esorm, model_nested):
        """
        Test basic aggregations
        """
        if TYPE_CHECKING:
            from esorm.query import ESQuery
            from esorm.aggs import ESAggsResponse, ESAggs

        # Test aggregations without query
        aggs_def: 'ESAggs' = {
            'f_int_sum': {
                'sum': {
                    'field': 'f_nested.f_int'
                }
            },
            'f_int_avg': {
                'avg': {
                    'field': 'f_nested.f_int'
                }
            },
            'f_int_min': {
                'min': {
                    'field': 'f_nested.f_int'
                }
            },
            'f_int_max': {
                'max': {
                    'field': 'f_nested.f_int'
                }
            },
        }
        aggs: 'ESAggsResponse' = await model_nested.aggregate(aggs_def)
        assert aggs['f_int_sum']['value'] == 14.0
        assert aggs['f_int_avg']['value'] == 3.5
        assert aggs['f_int_min']['value'] == 2.0
        assert aggs['f_int_max']['value'] == 5.0

        # Test aggregations with query
        query: 'ESQuery' = {
            'bool': {
                'must': [{
                    'match': {
                        'f_nested.f_str': 'nested_test2'
                    }
                }]
            }
        }
        aggs: 'ESAggsResponse' = await model_nested.aggregate(aggs_def, query=query)
        assert aggs['f_int_sum']['value'] == 2.0
        assert aggs['f_int_avg']['value'] == 2.0
        assert aggs['f_int_min']['value'] == 2.0
        assert aggs['f_int_max']['value'] == 2.0

        # Test terms aggregation
        aggs_def: 'ESAggs' = {
            'f_int_terms': {
                'terms': {
                    'field': 'f_nested.f_int'
                }
            }
        }
        aggs: 'ESAggsResponse' = await model_nested.aggregate(aggs_def)
        assert aggs['f_int_terms']['buckets'][0]['key'] == 2.0
        assert aggs['f_int_terms']['buckets'][0]['doc_count'] == 1
        assert aggs['f_int_terms']['buckets'][1]['key'] == 3.0
        assert aggs['f_int_terms']['buckets'][1]['doc_count'] == 1
        assert aggs['f_int_terms']['buckets'][2]['key'] == 4.0
        assert aggs['f_int_terms']['buckets'][2]['doc_count'] == 1
        assert aggs['f_int_terms']['buckets'][3]['key'] == 5.0
        assert aggs['f_int_terms']['buckets'][3]['doc_count'] == 1

    async def test_pagination(self, es, esorm, model_nested):
        """
        Test pagination
        """
        pagination = esorm.Pagination(page=1, page_size=2)
        res = await pagination(model_nested).all()
        assert len(res) == 2
        assert res[0].f_nested.f_str == 'nested_test2'
        assert res[1].f_nested.f_str == 'nested_test3'

        pagination = esorm.Pagination(page=2, page_size=2)
        res = await pagination(model_nested).all()
        assert len(res) == 2
        assert res[0].f_nested.f_str == 'nested_test4'
        assert res[1].f_nested.f_str == 'nested_test5'

    async def test_sort(self, es, esorm, model_nested):
        """
        Test sort
        """
        esorm.model.set_max_lazy_property_concurrency(2)

        sort = esorm.Sort(sort=[{'f_nested.f_int': {'order': 'asc'}}])
        res = await sort(model_nested).all()
        assert len(res) == 4
        assert res[0].f_nested.f_int == 2
        assert res[1].f_nested.f_int == 3
        assert res[2].f_nested.f_int == 4
        assert res[3].f_nested.f_int == 5

        sort = esorm.Sort(sort=[{'f_nested.f_int': {'order': 'desc'}}])
        res = await sort(model_nested).all()
        assert len(res) == 4
        assert res[0].f_nested.f_int == 5
        assert res[1].f_nested.f_int == 4
        assert res[2].f_nested.f_int == 3
        assert res[3].f_nested.f_int == 2

        sort = esorm.Sort(sort='f_nested.f_int')
        res = await sort(model_nested).all()
        assert len(res) == 4
        assert res[0].f_nested.f_int == 2
        assert res[1].f_nested.f_int == 3
        assert res[2].f_nested.f_int == 4
        assert res[3].f_nested.f_int == 5

    async def test_lazy_properties(self, es, esorm, model_lazy_prop):
        """
        Test lazy properties
        """
        esorm.model.set_max_lazy_property_concurrency(2)

        # Create 3 documents with the same content
        doc_id1 = await model_lazy_prop(f_str='test').save(refresh=True)
        assert doc_id1 is not None
        doc_id2 = await model_lazy_prop(f_str='test').save(refresh=True)
        assert doc_id2 is not None
        doc_id3 = await model_lazy_prop(f_str='test').save(refresh=True)
        assert doc_id3 is not None
        # Create 2 documents with different content
        doc_id = await model_lazy_prop(f_str='test2').save(refresh=True)
        assert doc_id is not None
        doc_id = await model_lazy_prop(f_str='test3').save(refresh=True)
        assert doc_id is not None

        # Test lazy properties
        sort = esorm.Sort(sort=[{'f_str': {'order': 'asc'}}])
        docs = await sort(model_lazy_prop).all()
        assert len(docs) == 5
        assert docs[0].f_str == 'test'
        same_f_strs = docs[0].same_f_strs
        assert len(same_f_strs) == 3
        assert same_f_strs[0].f_str == 'test'
        assert same_f_strs[0]._id == doc_id1
        assert same_f_strs[1].f_str == 'test'
        assert same_f_strs[1]._id == doc_id2
        assert same_f_strs[2].f_str == 'test'
        assert same_f_strs[2]._id == doc_id3
        assert docs[1].f_str == 'test'
        assert len(docs[1].same_f_strs) == 3
        assert docs[2].f_str == 'test'
        assert len(docs[2].same_f_strs) == 3
        assert docs[3].f_str == 'test2'
        assert len(docs[3].same_f_strs) == 1
        assert docs[4].f_str == 'test3'
        assert len(docs[4].same_f_strs) == 1

    async def test_shard_routing(self, es, esorm, model_config):
        """
        Test shard routing
        """
        doc = model_config(custom_id='test1', f_str='test1')
        doc_id = await doc.save(refresh=True)
        assert doc_id is not None
        query = {
            'bool': {
                'must': [{
                    'match': {
                        'f_str': 'test1'
                    }
                }]
            }
        }
        doc = await model_config.search_one(query)
        assert doc is not None
        assert doc._id == doc_id == "test1"
        assert doc._routing == "test1_routing"

        doc = model_config(custom_id='test2', f_str='test2')
        doc_id = await doc.save(refresh=True)
        assert doc_id is not None
        query = {
            'bool': {
                'must': [{
                    'match': {
                        'f_str': 'test2'
                    }
                }]
            }
        }
        docs = await model_config.search(query)
        assert len(docs) == 1
        doc = docs[0]
        assert doc._id == doc_id == "test2"
        assert doc._routing == "test2_routing"

        # Test different routing search
        docs = await model_config.search(query, routing="test1_routing")
        assert len(docs) == 0
        docs = await model_config.search(query, routing="test2_routing")
        assert len(docs) == 1

        # Override routing on save
        doc = model_config(custom_id='test3', f_str='test3')
        doc_id = await doc.save(refresh=True, routing="test3__routing__")
        assert doc_id is not None
        query = {
            'bool': {
                'must': [{
                    'match': {
                        'f_str': 'test3'
                    }
                }]
            }
        }
        doc = await model_config.search_one(query)
        assert doc is not None
        assert doc._id == doc_id == "test3"
        assert doc._routing == "test3__routing__"

    async def test_binary(self, es, esorm, model_es_optional):
        """
        Test binary fields
        """
        doc = model_es_optional(f_binary=b'\x01\x02\x03\x04')
        doc_id = await doc.save(refresh=True)

        assert doc_id is not None
        doc = await model_es_optional.get(doc_id)
        assert getattr(doc.f_binary, 'bytes') == b'\x01\x02\x03\x04'

        # Modify binary field
        doc.f_binary = b'\x05\x06\x07\x08'
        await doc.save(refresh=True)
        doc = await model_es_optional.get(doc_id)
        assert getattr(doc.f_binary, 'bytes') == b'\x05\x06\x07\x08'

    async def test_binary_nested(self, es, esorm, model_nested_binary):
        """
        Test nested binary fields
        """
        binary_model, nested_binary_model = model_nested_binary
        doc = nested_binary_model(f_nested=binary_model())
        doc.f_nested.f_binary = b'\x01\x02\x03\x04'
        doc_id = await doc.save()
        assert doc_id is not None

    async def test_nested_base_model_lazy_prop(self, es, esorm, model_nested_base_model):
        """
        Test nested base model
        """
        nested_base_model, nested_base_model_model = model_nested_base_model

        assert nested_base_model.ESConfig._lazy_properties['lazy_prop'] is not None

        doc = nested_base_model_model(f_nested=nested_base_model(f_str='test', f_int=1))
        doc_id = await doc.save()
        assert doc_id is not None

        doc = await nested_base_model_model.get(doc_id)
        assert doc is not None
        assert doc.f_nested.f_str == 'test'
        assert doc.f_nested.f_int == 1
        assert doc.f_nested.lazy_prop == 'lazy(f_str=test, f_int=1)'

    async def test_base_model_parent(self, es, esorm, model_base_model_parent):
        """
        Test base model parent
        """
        doc = model_base_model_parent(f_str='test', f_int=123, f_float=0.123)
        doc_id = await doc.save()
        assert doc_id is not None

        doc = await model_base_model_parent.get(doc_id)
        assert doc is not None
        assert doc.f_str == 'test'
        assert doc.f_int == 123
        assert doc.f_float == 0.123

        assert doc._id == 'test'

    async def test_source(self, es, esorm):
        """
        Test _source argument
        """

        class TestSourceModel(esorm.ESModel):
            f_int: int = 0
            f_str: str = 'a'

        await esorm.setup_mappings()

        doc = TestSourceModel(f_int=1, f_str='b')
        doc_id = await doc.save(refresh=True)
        assert doc_id is not None

        doc = await TestSourceModel.get(doc_id)
        assert doc.f_str == 'b'
        assert doc.f_int == 1

        doc = await TestSourceModel.search_one_by_fields(dict(_id=doc_id), _source=['f_str'])
        assert doc.f_str == 'b'
        assert doc.f_int == 0

        doc = await TestSourceModel.search_one_by_fields(dict(_id=doc_id), _source=['f_int'])
        assert doc.f_str == 'a'
        assert doc.f_int == 1

        doc = await TestSourceModel.get(doc_id, _source=['f_str'])
        assert doc.f_str == 'b'
        assert doc.f_int == 0

        doc = await TestSourceModel.get(doc_id, _source=['f_int'])
        assert doc.f_str == 'a'
        assert doc.f_int == 1

    async def test_primitive_list(self, es, esorm):
        """
        Test primitive list
        """
        from typing import List

        class PrimitiveListModel(esorm.ESModel):
            f_int_list: List[int] = []
            f_str_list: List[str] = []
            f_unsigned_long: List[esorm.fields.unsigned_long] = []  # type: ignore

        await esorm.setup_mappings()

        doc = PrimitiveListModel(
            f_int_list=[1, 2, 3], f_str_list=['a', 'b', 'c'],
            f_unsigned_long=[4, 5, 6]
        )
        doc_id = await doc.save(refresh=True)
        assert doc_id is not None

        doc = await PrimitiveListModel.get(doc_id)
        assert doc.f_int_list == [1, 2, 3]
        assert doc.f_str_list == ['a', 'b', 'c']
        assert doc.f_unsigned_long == [4, 5, 6]

    async def test_primitive_list_new_syntax(self, es, esorm):
        """
        Test primitive list new syntax
        """

        class PrimitiveListModel(esorm.ESModel):
            f_int_list: list[int] = []
            f_str_list: list[str] = []
            f_unsigned_long: list[esorm.fields.unsigned_long] = []  # type: ignore

        await esorm.setup_mappings()

        doc = PrimitiveListModel(f_int_list=[1, 2, 3], f_str_list=['a', 'b', 'c'])
        doc_id = await doc.save(refresh=True)
        assert doc_id is not None

        doc = await PrimitiveListModel.get(doc_id)
        assert doc.f_int_list == [1, 2, 3]
        assert doc.f_str_list == ['a', 'b', 'c']

    async def test_alias(self, es, esorm):
        """
        Test alias support
        """

        class AliasModel(esorm.ESModel):
            f_str: str = esorm.fields.Field(..., alias='f_str_alias')

        await esorm.setup_mappings()

        index = AliasModel.ESConfig.index
        properties = (await es.indices.get_mapping(index=index, request_timeout=90))[index]['mappings']['properties']
        assert 'f_str' not in properties
        assert 'f_str_alias' in properties

        doc = AliasModel(f_str='test')
        doc_id = await doc.save(refresh=True)
        assert doc_id is not None

        # Access by field name
        doc = await AliasModel.get(doc_id)
        assert doc.f_str == 'test'
        # Access by alias
        assert doc.dict(by_alias=True).get('f_str_alias') == 'test'

    async def test_ipvany(self, es, esorm):
        """
        Test IPvAnyAddress field
        """
        from pydantic.networks import IPvAnyAddress

        class IPvAnyModel(esorm.ESModel):
            f_ipv4: IPvAnyAddress
            f_ipv6: IPvAnyAddress

        await esorm.setup_mappings()

        doc = IPvAnyModel(f_ipv4='127.0.0.1', f_ipv6='::1')
        doc_id = await doc.save(refresh=True)
        assert doc_id is not None

        doc = await IPvAnyModel.get(doc_id)
        assert str(doc.f_ipv4) == '127.0.0.1'
        assert str(doc.f_ipv6) == '::1'

    async def test_enum(self, es, esorm):
        """
        Test Enum field
        """
        from enum import IntEnum, Enum

        class TestIntEnum(IntEnum):
            A = 1
            B = 2

        class TestStrEnum(str, Enum):
            A = 'a'
            B = 'b'

        class EnumModel(esorm.ESModel):
            f_int_enum: TestIntEnum
            f_str_enum: TestStrEnum

        await esorm.setup_mappings()

        doc = EnumModel(f_int_enum=TestIntEnum.A, f_str_enum=TestStrEnum.B)
        doc_id = await doc.save()
        assert doc_id is not None

        doc = await EnumModel.get(doc_id)
        assert doc.f_int_enum == TestIntEnum.A
        assert doc.f_str_enum == TestStrEnum.B

    async def test_es_types(self, es, model_es):
        """
        Test ES types
        """
        from esorm.fields import geo_point

        doc = model_es(
            f_keyword='test',
            f_text='test',
            f_binary=b'\x01\x02\x03\x04',
            f_byte=42,
            f_short=4242,
            f_int32=1337,
            f_long=-424242,
            f_unsigned_long=424242,
            f_float16=1.0,
            f_float32=1.0,
            f_double=1.0,
            f_geo_point=geo_point(lat=1.0, lon=2.0),
        )
        doc_id = await doc.save(refresh=True)
        assert doc_id is not None

        doc = await model_es.get(doc_id)
        assert doc.f_keyword == 'test'

    async def test_dense_vector(self, es, esorm):
        """
        Test Dense Vector feature
        """
        from esorm.fields import dense_vector

        class DenseVectorModel(esorm.ESModel):
            # OpenSearch needs the dimensions, ElasticSearch can detect it from the 1st document
            f_dense_vector: dense_vector = esorm.DenseVectorField(..., dims=3) if esorm.is_opensearch() else ...

        await esorm.setup_mappings()

        # Create a document with a dense vector
        doc = DenseVectorModel(f_dense_vector=[1.0, 2.0, 3.0])
        doc_id = await doc.save(refresh=True)
        assert doc_id is not None

        # Retrieve the document and check the dense vector
        doc = await DenseVectorModel.get(doc_id)
        assert doc.f_dense_vector == [1.0, 2.0, 3.0]

        # Perform a kNN search
        query = {
            "knn": {
                "field": "f_dense_vector",
                "query_vector": [1.0, 2.0, 3.0],
                "k": 1,
                "num_candidates": 10
            }
        }
        results = await DenseVectorModel.search(query)
        assert len(results) == 1
        assert results[0].f_dense_vector == [1.0, 2.0, 3.0]

    async def test_dense_vector_similarity(self, es, esorm):
        """
        Test the similarity metrics of the dense vector fields in the created mappings
        """
        from esorm.fields import dense_vector

        class DenseVectorSimilarityModel(esorm.ESModel):
            f_cosine: dense_vector = esorm.DenseVectorField(..., dims=3, similarity='cosine')
            f_dot_product: dense_vector = esorm.DenseVectorField(..., dims=3, similarity='dot_product')
            f_l2_norm: dense_vector = esorm.DenseVectorField(..., dims=3, similarity='l2_norm')
            f_max_inner_product: dense_vector = esorm.DenseVectorField(..., dims=3, similarity='max_inner_product')
            # The old 'l2' value is still accepted, it is converted to 'l2_norm'
            f_l2: dense_vector = esorm.DenseVectorField(..., dims=3, similarity='l2')  # type: ignore

        await esorm.setup_mappings()

        index = DenseVectorSimilarityModel.ESConfig.index
        properties = (await es.indices.get_mapping(index=index))[index]['mappings']['properties']
        if esorm.is_opensearch():
            expected = {'f_cosine': 'cosinesimil', 'f_dot_product': 'innerproduct', 'f_l2_norm': 'l2',
                        'f_max_inner_product': 'innerproduct', 'f_l2': 'l2'}
            for name, space_type in expected.items():
                prop = properties[name]
                assert prop['type'] == 'knn_vector'
                assert prop.get('space_type', prop.get('method', {}).get('space_type')) == space_type
        else:
            expected = {'f_cosine': 'cosine', 'f_dot_product': 'dot_product', 'f_l2_norm': 'l2_norm',
                        'f_max_inner_product': 'max_inner_product', 'f_l2': 'l2_norm'}
            for name, similarity in expected.items():
                assert properties[name]['type'] == 'dense_vector'
                assert properties[name]['similarity'] == similarity

    async def test_count(self, es, esorm):
        """
        Test count
        """

        class CountModel(esorm.ESModel):
            f_int: int

        await esorm.setup_mappings()

        # Create 10 documents
        for i in range(10):
            doc = CountModel(f_int=i)
            await doc.save(wait_for=True)

        # Count all documents
        count = await CountModel.count()
        assert count == 10

        # Count documents with a specific condition
        count = await CountModel.count(query={
            'bool': {
                'must': [{
                    "range": {
                        'f_int': {
                            'gte': 5
                        }
                    }
                }]
            }
        })
        assert count == 5

    async def test_query_builder(self, es, esorm):
        """
        Test the queries of the query builder
        """
        from typing import List, Optional
        from esorm.builder import Bool, Term, Terms, Range, Match, Exists, Nested, GeoDistance, Ids

        class BuilderComment(esorm.ESBaseModel):
            author: esorm.fields.keyword
            stars: int

        class BuilderModel(esorm.ESModel):
            id: int
            name: esorm.fields.text
            age: int
            tags: List[esorm.fields.keyword] = []
            email: Optional[esorm.fields.keyword] = None
            comments: List[BuilderComment] = []
            location: Optional[esorm.fields.geo_point] = None

        await esorm.setup_mappings()

        budapest = {'lat': 47.4979, 'lon': 19.0402}
        for doc in [
            BuilderModel(id=1, name='John Doe', age=30, tags=['python', 'es'], email='john@example.com',
                         comments=[BuilderComment(author='anna', stars=5)], location=budapest),
            BuilderModel(id=2, name='Jane Doe', age=17, tags=['python'],
                         comments=[BuilderComment(author='anna', stars=2), BuilderComment(author='bob', stars=5)]),
            BuilderModel(id=3, name='John Smith', age=45, tags=['docker'],
                         location={'lat': 46.2530, 'lon': 20.1414}),
        ]:
            await doc.save(wait_for=True)

        async def ids(query) -> List[int]:
            return sorted(doc.id for doc in await BuilderModel.search(query))

        # Parameters and with blocks give the same query and results
        q1 = Bool(must=[Match('name', 'John')], filter=[Range('age', gte=18)], must_not=[Exists('email')])
        with Bool() as q2:
            Match('name', 'John')
            with q2.filter:
                Range('age', gte=18)
            with q2.must_not:
                Exists('email')
        assert q1.to_query() == q2.to_query()
        assert await ids(q1) == [3]
        assert await ids(q2) == [3]
        # The query dict gives the same result
        assert sorted(doc.id for doc in await BuilderModel.search(q1.to_query())) == [3]

        # Operators
        assert await ids(Terms('tags', ['docker', 'es'])) == [1, 3]
        assert await ids(Term('tags', 'python') & ~Term('tags', 'es')) == [2]
        assert await ids(Range('age', lt=18) | Term('tags', 'docker')) == [2, 3]
        assert await ids(Ids(['1', '3'])) == [1, 3]

        # Nested documents: the same comment must match both conditions
        with Nested('comments') as nested:
            Term('comments.author', 'anna')
            Range('comments.stars', gte=4)
        assert await ids(nested) == [1]

        # Geo distance
        assert await ids(GeoDistance('location', budapest, '10km')) == [1]

        # The other methods accept builder queries too
        doc = await BuilderModel.search_one(Term('tags', 'docker'))
        assert doc is not None and doc.id == 3
        assert await BuilderModel.count(Term('tags', 'python')) == 2
        aggs = await BuilderModel.aggregate({'avg_age': {'avg': {'field': 'age'}}}, query=Match('name', 'Doe'))
        assert aggs['avg_age']['value'] == 23.5
        assert sorted([doc.id async for doc in BuilderModel.scan(Term('tags', 'python'), batch_size=1)]) == [1, 2]
        assert [len(batch) async for batch in BuilderModel.scan_batches(Terms('tags', ['python', 'docker']),
                                                                         batch_size=2)] == [2, 1]

        # Decorated models
        sorted_model = esorm.Sort(sort=[{'age': {'order': 'desc'}}])(BuilderModel)
        assert [doc.id for doc in await sorted_model.search(Range('age', gte=18))] == [3, 1]
        assert [doc.id async for doc in sorted_model.scan(Range('age', gte=18))] == [3, 1]
        paginated_model = esorm.Pagination(page=2, page_size=1)(BuilderModel)
        assert [doc.id for doc in await paginated_model.search(Range('age', gte=18), sort='age')] == [3]

    async def test_scan(self, es, esorm, monkeypatch):
        """
        Test iterating through all results with scan
        """
        from contextlib import aclosing

        class ScanModel(esorm.ESModel):
            class ESConfig:
                # More shards to test the tiebreaker of the sort
                settings = dict(number_of_shards=3)

            f_int: int
            f_group: int

        await esorm.setup_mappings()

        async with esorm.ESBulk(wait_for=True) as bulk:
            for i in range(50):
                await bulk.save(ScanModel(f_int=i, f_group=i % 3))

        # All documents
        docs = [doc async for doc in ScanModel.scan(batch_size=7)]
        assert sorted(doc.f_int for doc in docs) == list(range(50))
        assert len({doc._id for doc in docs}) == 50
        assert all(doc._version is not None and doc._seq_no is not None for doc in docs)

        # With query
        docs = [doc async for doc in ScanModel.scan({'range': {'f_int': {'gte': 20}}}, batch_size=7)]
        assert sorted(doc.f_int for doc in docs) == list(range(20, 50))

        # Sort by a field which is not unique, the tiebreaker must not lose documents
        docs = [doc async for doc in ScanModel.scan(sort='f_group', batch_size=7)]
        assert sorted(doc.f_int for doc in docs) == list(range(50))
        assert [doc.f_group for doc in docs] == sorted(doc.f_group for doc in docs)

        # Sort by a unique field
        docs = [doc async for doc in ScanModel.scan(sort=[{'f_int': {'order': 'desc'}}], batch_size=7)]
        assert [doc.f_int for doc in docs] == list(range(49, -1, -1))

        # Sort decorator
        sort = esorm.Sort(sort=[{'f_int': {'order': 'desc'}}])
        docs = [doc async for doc in sort(ScanModel).scan(batch_size=7)]
        assert [doc.f_int for doc in docs] == list(range(49, -1, -1))

        # Source filtering
        docs = [doc async for doc in ScanModel.scan(sort='f_int', batch_size=20, _source=['f_int', 'f_group'])]
        assert [doc.f_int for doc in docs] == list(range(50))

        # Batches
        batches = [batch async for batch in ScanModel.scan_batches(sort='f_int', batch_size=7)]
        assert [len(batch) for batch in batches] == [7] * 7 + [1]
        assert [doc.f_int for batch in batches for doc in batch] == list(range(50))

        # Empty result
        assert [doc async for doc in ScanModel.scan({'term': {'f_int': 1000}})] == []

        # Changes made during the iteration are not visible
        docs = []
        async for doc in ScanModel.scan(sort='f_int', batch_size=10):
            if doc.f_int == 0:
                await ScanModel(f_int=1000, f_group=0).save(refresh=True)
            docs.append(doc)
        assert [doc.f_int for doc in docs] == list(range(50))
        assert await ScanModel.count() == 51

        # The point in time is closed when the iteration is stopped early
        closed = []
        close_point_in_time = es.close_point_in_time

        async def close_point_in_time_spy(**kwargs):
            closed.append(kwargs['id'])
            return await close_point_in_time(**kwargs)

        monkeypatch.setattr(es, 'close_point_in_time', close_point_in_time_spy)
        async with aclosing(ScanModel.scan(batch_size=7)) as it:
            async for _ in it:
                break
        assert len(closed) == 1

    async def test_subfields(self, es, esorm):
        """
        Test subfields functionality
        """

        class SubfieldModel(esorm.ESModel):
            # Text field with keyword shortcut
            f_text_keyword: esorm.fields.text = esorm.fields.TextField(..., keyword=True)
            # Text field with custom subfields
            f_text_custom: esorm.fields.text = esorm.fields.TextField(..., fields={
                'raw': {'type': 'keyword'},
                'english': {'type': 'text', 'analyzer': 'english'}
            })
            # Text field with both keyword and custom fields
            f_text_combined: esorm.fields.text = esorm.fields.TextField(..., keyword=True, fields={
                'english': {'type': 'text', 'analyzer': 'english'}
            })
            # Numeric field with subfields
            f_numeric: float = esorm.fields.NumericField(..., es_type='double', fields={
                'raw': {'type': 'keyword'}
            })
            # Regular field with custom keyword subfield
            f_regular: str = esorm.fields.Field(..., fields={
                'keyword': {'type': 'keyword', 'ignore_above': 256}
            })

        await esorm.setup_mappings()

        # Check mappings
        mappings = await es.indices.get_mapping(index=SubfieldModel.ESConfig.index)
        properties = mappings[SubfieldModel.ESConfig.index]['mappings']['properties']

        # Test keyword shortcut
        assert properties['f_text_keyword']['type'] == 'text'
        assert 'fields' in properties['f_text_keyword']
        assert properties['f_text_keyword']['fields']['keyword']['type'] == 'keyword'

        # Test custom subfields
        assert properties['f_text_custom']['type'] == 'text'
        assert 'fields' in properties['f_text_custom']
        assert properties['f_text_custom']['fields']['raw']['type'] == 'keyword'
        assert properties['f_text_custom']['fields']['english']['type'] == 'text'
        assert properties['f_text_custom']['fields']['english']['analyzer'] == 'english'

        # Test combined (keyword + custom)
        assert properties['f_text_combined']['type'] == 'text'
        assert 'fields' in properties['f_text_combined']
        assert properties['f_text_combined']['fields']['keyword']['type'] == 'keyword'
        assert properties['f_text_combined']['fields']['english']['type'] == 'text'
        assert properties['f_text_combined']['fields']['english']['analyzer'] == 'english'

        # Test numeric subfields
        assert properties['f_numeric']['type'] == 'double'
        assert 'fields' in properties['f_numeric']
        assert properties['f_numeric']['fields']['raw']['type'] == 'keyword'

        # Test regular field with custom keyword
        assert properties['f_regular']['type'] == 'keyword'  # str maps to keyword by default
        assert 'fields' in properties['f_regular']
        assert properties['f_regular']['fields']['keyword']['type'] == 'keyword'
        assert properties['f_regular']['fields']['keyword']['ignore_above'] == 256

        # Create and save a document
        doc = SubfieldModel(
            f_text_keyword='Test with Keyword',
            f_text_custom='Custom Test',
            f_text_combined='Combined Test',
            f_numeric=42.5,
            f_regular='Regular Field'
        )
        doc_id = await doc.save(wait_for=True)
        assert doc_id is not None

        # Test searching with keyword subfield
        results = await SubfieldModel.search({
            'term': {'f_text_keyword.keyword': 'Test with Keyword'}
        })
        assert len(results) == 1
        assert results[0].f_text_keyword == 'Test with Keyword'

        # Test searching with custom subfield
        results = await SubfieldModel.search({
            'match': {'f_text_custom.english': 'custom testing'}
        })
        assert len(results) == 1
        assert results[0].f_text_custom == 'Custom Test'

        # Test numeric subfield search with keyword subfield
        results = await SubfieldModel.search({
            'term': {'f_numeric.raw': '42.5'}
        })
        assert len(results) == 1
        assert results[0].f_numeric == 42.5

    async def test_refresh_parameter(self, es, esorm):
        """
        Test refresh parameter functionality
        Tests refresh=True, refresh="wait_for", and backward compatibility with wait_for=True
        """

        class RefreshTestModel(esorm.ESModel):
            f_str: str
            f_int: int

        await esorm.setup_mappings()

        # Test 1: refresh=True (immediate refresh)
        doc1 = RefreshTestModel(f_str='test_refresh_true', f_int=1)
        doc_id1 = await doc1.save(refresh=True)
        assert doc_id1 is not None

        # Document should be immediately visible after save with refresh=True
        results = await RefreshTestModel.search_by_fields({'f_str': 'test_refresh_true'})
        assert len(results) == 1
        assert results[0].f_int == 1

        # Test 2: refresh="wait_for" (wait for next scheduled refresh)
        doc2 = RefreshTestModel(f_str='test_refresh_wait_for', f_int=2)
        doc_id2 = await doc2.save(refresh="wait_for")
        assert doc_id2 is not None

        # Document should be visible after save with refresh="wait_for"
        results = await RefreshTestModel.search_by_fields({'f_str': 'test_refresh_wait_for'})
        assert len(results) == 1
        assert results[0].f_int == 2

        # Test 3: wait_for=True (backward compatibility)
        doc3 = RefreshTestModel(f_str='test_wait_for_true', f_int=3)
        doc_id3 = await doc3.save(wait_for=True)
        assert doc_id3 is not None

        # Document should be visible after save with wait_for=True
        results = await RefreshTestModel.search_by_fields({'f_str': 'test_wait_for_true'})
        assert len(results) == 1
        assert results[0].f_int == 3

        # Test 4: refresh parameter has priority over wait_for
        doc4 = RefreshTestModel(f_str='test_priority', f_int=4)
        doc_id4 = await doc4.save(refresh=True, wait_for=True)
        assert doc_id4 is not None

        # Document should be visible immediately (refresh=True takes priority)
        results = await RefreshTestModel.search_by_fields({'f_str': 'test_priority'})
        assert len(results) == 1
        assert results[0].f_int == 4

        # Test 5: Test delete with refresh parameter
        doc5 = RefreshTestModel(f_str='test_delete_refresh', f_int=5)
        doc_id5 = await doc5.save(refresh=True)
        assert doc_id5 is not None

        # Verify document exists
        results = await RefreshTestModel.search_by_fields({'f_str': 'test_delete_refresh'})
        assert len(results) == 1

        # Delete with refresh=True
        await doc5.delete(refresh=True)

        # Document should be immediately gone
        results = await RefreshTestModel.search_by_fields({'f_str': 'test_delete_refresh'})
        assert len(results) == 0

        # Test 6: Test bulk operations with refresh parameter
        async with esorm.ESBulk(refresh=True) as bulk:
            for i in range(5):
                doc = RefreshTestModel(f_str='test_bulk_refresh', f_int=100 + i)
                await bulk.save(doc)

        # Documents should be immediately visible
        results = await RefreshTestModel.search_by_fields({'f_str': 'test_bulk_refresh'})
        assert len(results) == 5

        # Test 7: Test bulk operations with refresh="wait_for"
        async with esorm.ESBulk(refresh="wait_for") as bulk:
            for i in range(3):
                doc = RefreshTestModel(f_str='test_bulk_wait_for', f_int=200 + i)
                await bulk.save(doc)

        # Documents should be visible after refresh
        results = await RefreshTestModel.search_by_fields({'f_str': 'test_bulk_wait_for'})
        assert len(results) == 3

    async def test_client_api(self, es):
        """
        Test the client API with the keyword arguments of the elasticsearch client, which should work with both
        ElasticSearch and OpenSearch
        """
        from elasticsearch import NotFoundError, ConflictError, BadRequestError

        index = 'esorm_client_api'
        await es.indices.create(index=index, mappings={'properties': {
            'f_str': {'type': 'keyword'},
            'f_int': {'type': 'long'},
            'f_vector': {'type': 'dense_vector', 'dims': 2, 'similarity': 'l2_norm'},
        }}, settings={'number_of_shards': 1})
        await es.indices.put_mapping(index=index, properties={'f_new': {'type': 'keyword'}})
        await es.indices.put_settings(index=index, settings={'refresh_interval': '2s'})

        for i in range(5):
            await es.index(index=index, id=str(i), document={'f_str': f'doc{i}', 'f_int': i, 'f_vector': [i, i]})
        await es.update(index=index, id='0', doc={'f_new': 'updated'})
        await es.indices.refresh(index=index)

        res = await es.get(index=index, id='0')
        assert res['_source']['f_new'] == 'updated'

        res = await es.search(index=index, query={'range': {'f_int': {'gte': 1}}}, sort=[{'f_int': 'desc'}],
                              from_=1, size=2, source=['f_int'])
        assert res['hits']['total']['value'] == 4
        assert [hit['_source'] for hit in res['hits']['hits']] == [{'f_int': 3}, {'f_int': 2}]

        res = await es.search(index=index, knn={'field': 'f_vector', 'query_vector': [4.0, 4.0], 'k': 1,
                                                'num_candidates': 10}, size=1)
        assert res['hits']['hits'][0]['_id'] == '4'

        res = await es.count(index=index, query={'term': {'f_str': 'doc1'}})
        assert res['count'] == 1

        res = await es.delete_by_query(index=index, query={'term': {'f_str': 'doc1'}}, refresh=True)
        assert res['deleted'] == 1

        # The exceptions of the elasticsearch client should be raised
        with pytest.raises(NotFoundError):
            await es.get(index=index, id='1')
        with pytest.raises(ConflictError):
            await es.create(index=index, id='0', document={'f_str': 'conflict'})
        with pytest.raises(BadRequestError):
            await es.search(index=index, query={'invalid_query': {}})

        assert await es.ping()
        await es.indices.delete(index=index)
