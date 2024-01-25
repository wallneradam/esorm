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
