"""
This module contains all the exceptions that can be raised by ESORM.
"""
from typing import List, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from esorm import ESModel


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


class ConflictError(Exception):
    """
    Raised when a conflict occurs.

    You can manually raise this to retry operation with `retry_on_conflict` decorator.
    """


class BulkOperationError(TypedDict):
    """
    A dictionary type to represent an error in a bulk operation response from Elasticsearch.
    """
    # The status code of the error
    status: int
    # Type of the error
    type: str
    # A human-readable reason for the error
    reason: str
    # The model object
    model: 'ESModel'


class BulkError(Exception):
    """
    Exception for handling bulk operation errors.
    """

    failed_operations: List[BulkOperationError]

    def __init__(self, failed_operations: List[BulkOperationError]):
        super().__init__(f"Bulk operation failed for some actions: {failed_operations}")
        self.failed_operations = failed_operations
