"""
Tests of the timezone handling of datetime and time fields, they don't need ElasticSearch
"""
from datetime import datetime, date, time, timezone, timedelta

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
