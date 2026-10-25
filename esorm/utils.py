"""
Utility functions
"""
import re
import datetime


def snake_case(camel_str: str):
    """
    Convert to snake case

    :param camel_str: The string to convert to snake case
    :return: Converted string
    """
    if '_' in camel_str:  # If it is already snake cased
        return camel_str
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def camel_case(snake_str: str, capitalize_first: bool = False):
    """
    Convert to camel case

    :param snake_str: The string to convert to camel case
    :param capitalize_first: Capitalize the first letter
    :return: Converted string
    """
    components = snake_str.split('_')
    if capitalize_first:
        components[0] = components[0].title()
    return components[0] + ''.join((x.title()) for x in components[1:])


def utcnow():
    """
    Get current UTC time

    :return: Current UTC time
    """
    return datetime.datetime.now(datetime.timezone.utc)


def datetime_to_es(dt: datetime.datetime) -> str:
    """
    Convert datetime to the string stored in ElasticSearch

    Timezone aware datetimes are converted to UTC, naive datetimes are treated as UTC.

    :param dt: The datetime to convert
    :return: ISO 8601 string in UTC with 'Z' suffix
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt.replace(tzinfo=None).isoformat() + 'Z'


def time_to_es(t: datetime.time) -> str:
    """
    Convert time to the string stored in ElasticSearch (1970-01-01 + time)

    Timezone aware times are converted to UTC, naive times are treated as UTC.

    :param t: The time to convert
    :return: ISO 8601 datetime string in UTC with 'Z' suffix
    """
    epoch = datetime.date(1970, 1, 1)
    dt = datetime.datetime.combine(epoch, t)
    if dt.tzinfo is not None:
        # Keep the date at 1970-01-01, even if the conversion crosses midnight
        dt = datetime.datetime.combine(epoch, dt.astimezone(datetime.timezone.utc).time())
    return datetime_to_es(dt)


def time_from_es(value: str) -> datetime.time:
    """
    Convert the string stored in ElasticSearch (1970-01-01 + time) to time

    Times with timezone offset are converted to UTC, the result is a naive UTC time.

    :param value: The ISO 8601 datetime string from ElasticSearch
    :return: Naive UTC time
    """
    # Python 3.10 does not support the 'Z' suffix in fromisoformat
    if value.endswith('Z'):
        value = value[:-1] + '+00:00'
    dt = datetime.datetime.fromisoformat(value)
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt.time()
