"""
Tests of the timezone handling of datetime and time fields, they don't need ElasticSearch
"""
from datetime import datetime, date, time, timezone, timedelta
from typing import Optional, List

import pytest

import esorm
from esorm.utils import datetime_to_es, time_to_es, time_from_es

CEST = timezone(timedelta(hours=2))


@pytest.fixture(scope="module")
def model_datetime():
    """
    Model with date and time fields, removed from the registered models after the tests
    """

    class DatetimeModel(esorm.ESModel):
        f_datetime: datetime
        f_date: date
        f_time: time

    yield DatetimeModel
    # noinspection PyProtectedMember
    esorm.model._ESModelMeta.__models__.clear()


class TestDatetimeToES:
    """
    Conversion of datetimes and times to the stored strings
    """

    @pytest.mark.parametrize('value, expected', [
        # Naive datetimes are treated as UTC
        (datetime(2024, 1, 1, 12), '2024-01-01T12:00:00Z'),
        (datetime(2024, 1, 1, 12, 30, 15, 123456), '2024-01-01T12:30:15.123456Z'),
        # Aware UTC datetimes
        (datetime(2024, 1, 1, 12, tzinfo=timezone.utc), '2024-01-01T12:00:00Z'),
        # Aware non-UTC datetimes are converted to UTC
        (datetime(2024, 1, 1, 12, tzinfo=CEST), '2024-01-01T10:00:00Z'),
        (datetime(2024, 1, 1, 1, tzinfo=CEST), '2023-12-31T23:00:00Z'),
        (datetime(2024, 1, 1, 12, tzinfo=timezone(timedelta(hours=-5))), '2024-01-01T17:00:00Z'),
    ])
    def test_datetime_to_es(self, value, expected):
        assert datetime_to_es(value) == expected

    @pytest.mark.parametrize('value, expected', [
        # Naive times are treated as UTC
        (time(12), '1970-01-01T12:00:00Z'),
        # Aware UTC times
        (time(12, tzinfo=timezone.utc), '1970-01-01T12:00:00Z'),
        # Aware non-UTC times are converted to UTC
        (time(12, tzinfo=CEST), '1970-01-01T10:00:00Z'),
        # The date stays 1970-01-01, even if the conversion crosses midnight
        (time(1, 30, tzinfo=CEST), '1970-01-01T23:30:00Z'),
        (time(22, tzinfo=timezone(timedelta(hours=-5))), '1970-01-01T03:00:00Z'),
    ])
    def test_time_to_es(self, value, expected):
        assert time_to_es(value) == expected

    @pytest.mark.parametrize('value, expected', [
        ('1970-01-01T12:00:00Z', time(12)),
        # Format of the documents stored by earlier versions
        ('1970-01-01T12:00:00', time(12)),
        ('1970-01-01T12:00:00+02:00', time(10)),
        ('1970-01-01T12:00:00.123456Z', time(12, 0, 0, 123456)),
    ])
    def test_time_from_es(self, value, expected):
        res = time_from_es(value)
        assert res == expected
        assert res.tzinfo is None

    @pytest.mark.parametrize('f_datetime, f_time', [
        (datetime(2024, 1, 1, 12), time(12)),
        (datetime(2024, 1, 1, 12, tzinfo=timezone.utc), time(12, tzinfo=timezone.utc)),
        (datetime(2024, 1, 1, 14, tzinfo=CEST), time(14, tzinfo=CEST)),
    ])
    def test_model_to_es_and_back(self, model_datetime, f_datetime, f_time):
        """
        Naive, aware UTC and aware non-UTC values are stored and loaded back as the same UTC values
        """
        doc = model_datetime(f_datetime=f_datetime, f_date=date(2024, 1, 1), f_time=f_time)
        source = doc.to_es()
        assert source == {
            'f_datetime': '2024-01-01T12:00:00Z',
            'f_date': '2024-01-01',
            'f_time': '1970-01-01T12:00:00Z',
        }

        doc = model_datetime.from_es({'_id': '1', '_source': source})
        # Loaded datetimes are aware UTC datetimes
        assert doc.f_datetime == datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
        assert doc.f_datetime.utcoffset() == timedelta(0)
        # They are the same instant as the original value (a naive one is taken as UTC)
        if f_datetime.tzinfo is not None:
            assert doc.f_datetime == f_datetime
        # Loaded times are naive UTC times
        assert doc.f_time == time(12)
        assert doc.f_time.tzinfo is None
        assert doc.f_date == date(2024, 1, 1)


@pytest.fixture(scope="module")
def models_time_fields():
    """
    Models with optional, list and nested time fields, removed from the registered models after the tests
    """

    class TimeSubModel(esorm.ESBaseModel):
        f_time: time
        f_optional_time: Optional[time] = None

    class TimeFieldsModel(esorm.ESModel):
        f_optional_time: Optional[time] = None
        f_union_time: time | None = None
        f_list_time: List[time] = []
        f_nested: Optional[TimeSubModel] = None
        f_nested_list: List[TimeSubModel] = []

    yield TimeSubModel, TimeFieldsModel
    # noinspection PyProtectedMember
    esorm.model._ESModelMeta.__models__.clear()


class TestTimeFieldsFromES:
    """
    Loading time fields that are optional, in lists or in nested models
    """

    def test_from_es(self, models_time_fields):
        _, model = models_time_fields
        doc = model.from_es({'_id': '1', '_source': {
            'f_optional_time': '1970-01-01T10:00:00Z',
            'f_union_time': '1970-01-01T11:00:00Z',
            'f_list_time': ['1970-01-01T12:00:00Z', '1970-01-01T13:00:00+02:00'],
            'f_nested': {'f_time': '1970-01-01T14:00:00Z', 'f_optional_time': '1970-01-01T15:00:00Z'},
            'f_nested_list': [
                {'f_time': '1970-01-01T16:00:00Z'},
                {'f_time': '1970-01-01T17:00:00Z', 'f_optional_time': '1970-01-01T18:00:00Z'},
            ],
        }})
        assert doc.f_optional_time == time(10)
        assert doc.f_union_time == time(11)
        assert doc.f_list_time == [time(12), time(11)]
        assert doc.f_nested is not None
        assert doc.f_nested.f_time == time(14)
        assert doc.f_nested.f_optional_time == time(15)
        assert [(n.f_time, n.f_optional_time) for n in doc.f_nested_list] == [
            (time(16), None),
            (time(17), time(18)),
        ]

    def test_from_es_none(self, models_time_fields):
        """
        None values of optional fields stay None
        """
        _, model = models_time_fields
        doc = model.from_es({'_id': '1', '_source': {
            'f_optional_time': None,
            'f_nested': None,
            'f_nested_list': [{'f_time': '1970-01-01T16:00:00Z', 'f_optional_time': None}],
        }})
        assert doc.f_optional_time is None
        assert doc.f_union_time is None
        assert doc.f_nested is None
        assert doc.f_nested_list[0].f_optional_time is None

    def test_to_es_and_back(self, models_time_fields):
        sub_model, model = models_time_fields
        doc = model(
            f_optional_time=time(10, tzinfo=CEST),
            f_union_time=time(11),
            f_list_time=[time(12), time(13)],
            f_nested=sub_model(f_time=time(14), f_optional_time=time(15)),
            f_nested_list=[sub_model(f_time=time(16)), sub_model(f_time=time(17), f_optional_time=time(18))],
        )
        source = doc.to_es()
        assert source == {
            'f_optional_time': '1970-01-01T08:00:00Z',
            'f_union_time': '1970-01-01T11:00:00Z',
            'f_list_time': ['1970-01-01T12:00:00Z', '1970-01-01T13:00:00Z'],
            'f_nested': {'f_time': '1970-01-01T14:00:00Z', 'f_optional_time': '1970-01-01T15:00:00Z'},
            'f_nested_list': [
                {'f_time': '1970-01-01T16:00:00Z', 'f_optional_time': None},
                {'f_time': '1970-01-01T17:00:00Z', 'f_optional_time': '1970-01-01T18:00:00Z'},
            ],
        }

        loaded = model.from_es({'_id': '1', '_source': source})
        assert loaded.f_optional_time == time(8)
        assert loaded.f_union_time == time(11)
        assert loaded.f_list_time == [time(12), time(13)]
        assert loaded.f_nested == doc.f_nested
        assert loaded.f_nested_list == doc.f_nested_list


@pytest.fixture(scope="module")
def models_nested_timestamp():
    """
    Models with nested ESModelTimestamp, removed from the registered models after the tests
    """

    class NestedTimestampModel(esorm.ESModelTimestamp):
        f_str: str

    class NotTimestampModel(esorm.ESBaseModel):
        created_at: Optional[datetime] = None

    class ParentModel(esorm.ESModel):
        f_nested: NestedTimestampModel
        f_nested_list: List[NestedTimestampModel] = []
        f_not_timestamp: NotTimestampModel = NotTimestampModel()

    yield NestedTimestampModel, ParentModel
    # noinspection PyProtectedMember
    esorm.model._ESModelMeta.__models__.clear()


class TestNestedTimestamps:
    """
    The created_at and modified_at fields of nested ESModelTimestamp models
    """

    def test_new_nested(self, models_nested_timestamp):
        """
        created_at of new nested models is set to modified_at
        """
        nested_model, model = models_nested_timestamp
        source = model(f_nested=nested_model(f_str='a'), f_nested_list=[nested_model(f_str='b')]).to_es()
        for nested in (source['f_nested'], source['f_nested_list'][0]):
            assert nested['modified_at'].endswith('Z')
            assert nested['created_at'] == nested['modified_at']
        # Only ESModelTimestamp models are updated
        assert source['f_not_timestamp']['created_at'] is None

    def test_existing_nested(self, models_nested_timestamp):
        """
        created_at of existing nested models is kept, modified_at is updated
        """
        nested_model, model = models_nested_timestamp
        old = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
        source = model(f_nested=nested_model(f_str='a', created_at=old, modified_at=old)).to_es()
        assert source['f_nested']['created_at'] == '2024-01-01T12:00:00Z'
        assert source['f_nested']['modified_at'] > '2024-01-01T12:00:00Z'


class TestUpdateFromES:
    """
    Updating an existing model from ElasticSearch data (used by reload())
    """

    def test_update_from_es(self, models_time_fields):
        sub_model, model = models_time_fields
        doc = model(f_union_time=time(1))
        doc.update_from_es({'_id': '1', '_version': 2, '_source': {
            # Fields which were not set explicitly are updated too
            'f_optional_time': '1970-01-01T10:00:00Z',
            'f_list_time': ['1970-01-01T12:00:00Z'],
            'f_nested': {'f_time': '1970-01-01T14:00:00Z'},
            'f_nested_list': [{'f_time': '1970-01-01T16:00:00Z', 'f_optional_time': '1970-01-01T17:00:00Z'}],
            # Unknown fields are ignored
            'f_unknown': 'x',
        }})
        assert doc.f_optional_time == time(10)
        # Fields missing from the source are kept
        assert doc.f_union_time == time(1)
        assert doc.f_list_time == [time(12)]
        assert isinstance(doc.f_nested, sub_model)
        assert doc.f_nested.f_time == time(14)
        assert doc.f_nested_list == [sub_model(f_time=time(16), f_optional_time=time(17))]
        assert doc._version == 2
        assert not hasattr(doc, 'f_unknown')

    def test_update_from_es_datetime(self, model_datetime):
        doc = model_datetime(f_datetime=datetime(2020, 1, 1), f_date=date(2020, 1, 1), f_time=time(1))
        doc.update_from_es({'_id': '1', '_source': {
            'f_datetime': '2024-01-01T12:00:00Z',
            'f_date': '2024-01-01',
            'f_time': '1970-01-01T12:00:00Z',
        }})
        assert doc.f_datetime == datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
        assert doc.f_date == date(2024, 1, 1)
        assert doc.f_time == time(12)
