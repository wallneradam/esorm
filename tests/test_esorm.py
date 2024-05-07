from typing import TYPE_CHECKING
import pytest


@pytest.mark.parametrize('service', ['es7x', 'es8x'], scope='class')
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
        assert model_python.ESConfig.index == 'esorm-python_field_model'

    async def test_create_model_with_es_fields(self, es, esorm, model_es):
        """
        Test model with ES fields
        """
        assert model_es is not None
        assert model_es.ESConfig.index == 'esorm-es_field_model'

    async def test_create_model_with_es_optional_fields(self, es, esorm, model_es_optional):
        """
        Test model with ES fields
        """
        assert model_es_optional is not None
        assert model_es_optional.ESConfig.index == 'esorm-es_optional_field_model'

    async def test_create_timestamp_models(self, es, esorm, model_timestamp):
        """
        Test timestamp models
        """
        assert model_timestamp is not None
        assert model_timestamp.ESConfig.index == 'esorm-timestamp_model'
        assert 'created_at' in model_timestamp.model_fields
        assert 'modified_at' in model_timestamp.model_fields

    async def test_create_model_config(self, es, esorm, model_config):
        """
        Test model config
        """
        assert model_config is not None
        assert model_config.ESConfig.index == 'custom_index'

    async def test_create_model_with_id(self, es, esorm, model_with_id):
        """
        Test model with config
        """
        assert model_with_id is not None
        assert model_with_id.ESConfig.index == 'esorm-id_model'
        assert model_with_id.ESConfig.id_field == 'id'

    async def test_create_model_with_int_id(self, es, esorm, model_with_int_id):
        """
        Test model with int id
        """
        assert model_with_int_id is not None
        assert model_with_int_id.ESConfig.index == 'esorm-int_id_model'
        assert model_with_int_id.ESConfig.id_field == 'custom_id'

    async def test_create_model_with_prop_id(self, es, esorm, model_with_prop_id):
        """
        Test model with property id
        """
        assert model_with_prop_id is not None
        assert model_with_prop_id.ESConfig.index == 'esorm-prop_id_model'
        assert model_with_prop_id.ESConfig.id_field is None

    async def test_create_nested_model(self, es, esorm, model_nested):
        """
        Test nested model
        """
        assert model_nested is not None
        assert model_nested.ESConfig.index == 'esorm-nested_field_model'

    async def test_create_mappings(self, es, esorm, model_python, model_es, model_es_optional,
                                   model_timestamp, model_config, model_with_id, model_with_int_id,
                                   model_with_prop_id, model_nested, model_lazy_prop):
        """
        Test create mappings
        """
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
        # Check if mappings are correct for ES fields
        mappings = await es.indices.get_mapping(index=model_es.ESConfig.index)
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_keyword']['type'] == 'keyword'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_text']['type'] == 'text'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_binary']['type'] == 'binary'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_byte']['type'] == 'byte'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_short']['type'] == 'short'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_int32']['type'] == 'integer'
        assert mappings[model_es.ESConfig.index]['mappings']['properties']['f_long']['type'] == 'long'
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
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_float16']['type'] == 'half_float'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_float32']['type'] == 'float'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_double']['type'] == 'double'
        assert mappings[model_es_optional.ESConfig.index]['mappings']['properties']['f_geo_point'][
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
        doc.f_nested.f_str = "nested_test_updated"
        await doc.save()

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
            self.__class__.doc_id = None

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

    async def test_bulk_race_condition_and_reload(self, es, esorm, model_nested, model_timestamp):
        """
        Test bulk operations with conflict
        """
        doc = model_nested(f_nested=model_timestamp(f_str=f"nested_test1"), f_float=10.0)
        await doc.save()

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
                doc1.f_nested.f_str = "nested_test1_updated1"
                await bulk.save(doc2)

        # We test reloading here, it should now update the _primary_term and _seq_no
        await doc.reload()
        # So the deletion of document should be ok
        await doc.delete()

    async def test_search(self, es, esorm, model_nested):
        """
        Test search
        """
        if TYPE_CHECKING:
            from esorm.query import ESQuery

        # Search
        query: 'ESQuery' = {
            'bool': {
                'must': {
                    'match': {
                        'f_nested.f_str': 'nested_test2'
                    }
                }
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
                'must': {
                    'prefix': {
                        'f_nested.f_str': 'nested_test'
                    }
                }
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
                'must': {
                    'match': {
                        'f_nested.f_str': 'nested_test2'
                    }
                }
            }
        }
        doc = await model_nested.search_one(query)
        assert doc is not None
        assert doc.f_nested.f_str == 'nested_test2'

        # Search one with multiple results
        query: 'ESQuery' = {
            'bool': {
                'must': {
                    'prefix': {
                        'f_nested.f_str': 'nested_test'
                    }
                }
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
                'must': {
                    'match': {
                        'f_nested.f_str': 'nested_test2'
                    }
                }
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
        doc_id1 = await model_lazy_prop(f_str='test').save()
        assert doc_id1 is not None
        doc_id2 = await model_lazy_prop(f_str='test').save()
        assert doc_id2 is not None
        doc_id3 = await model_lazy_prop(f_str='test').save()
        assert doc_id3 is not None
        # Create 2 documents with different content
        doc_id = await model_lazy_prop(f_str='test2').save()
        assert doc_id is not None
        doc_id = await model_lazy_prop(f_str='test3').save()
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
        doc_id = await doc.save()
        assert doc_id is not None
        query = {
            'bool': {
                'must': {
                    'match': {
                        'f_str': 'test1'
                    }
                }
            }
        }
        doc = await model_config.search_one(query)
        assert doc is not None
        assert doc._id == doc_id == "test1"
        assert doc._routing == "test1_routing"

        doc = model_config(custom_id='test2', f_str='test2')
        doc_id = await doc.save()
        assert doc_id is not None
        query = {
            'bool': {
                'must': {
                    'match': {
                        'f_str': 'test2'
                    }
                }
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
        doc_id = await doc.save(routing="test3__routing__")
        assert doc_id is not None
        query = {
            'bool': {
                'must': {
                    'match': {
                        'f_str': 'test3'
                    }
                }
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
        doc_id = await doc.save()

        assert doc_id is not None
        doc = await model_es_optional.get(doc_id)
        assert getattr(doc.f_binary, 'bytes') == b'\x01\x02\x03\x04'

        # Modify binary field
        doc.f_binary = b'\x05\x06\x07\x08'
        await doc.save()
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
