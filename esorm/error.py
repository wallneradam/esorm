"""
This module contains all the exceptions that can be raised by ESORM.
"""
class NotFoundError(Exception):
    """
    Raised when a model is not found.
    """


class IndexDoesNotFoundError(Exception):
    """
    Raised when an index does not exist.
    """


class InvalidResponseError(Exception):
    """
    Raised when the response from Elasticsearch is invalid.
    """


class InvalidModelError(Exception):
    """
    Raised when a model is invalid.
    """
