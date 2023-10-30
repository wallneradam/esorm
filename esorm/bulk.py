"""
Bulk operation for ElasticSearch
"""
from datetime import datetime

from .model import TModel, ESModelTimestamp
from .esorm import es


class ESBulk:
    """
    Bulk operation for ElasticSearch
    """

    def __init__(self, wait_for=False, **bulk_kwargs):
        """
        Create a bulk context manager
        :param wait_for: Whether to wait for active shards
        :param bulk_kwargs: Other bulk arguments
        """
        self._actions = []
        self._bulk_kwargs = dict(bulk_kwargs)
        if wait_for:
            self._bulk_kwargs['refresh'] = 'wait_for'

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Do the bulk operation if no exception is raised
        if exc_val is None:
            await es.bulk(operations=self._actions, **self._bulk_kwargs)
        # The exceptions are not handled here
        return False

    async def save(self, model: TModel):
        """
        Add the model to the bulk for saving
        :param model: The model to add for saving
        """
        document: dict = model.to_es()
        if model.ESConfig.id_field:
            del document[model.ESConfig.id_field]

        index = {
            '_index': model.ESConfig.index,
            '_id': model.__id__
        }
        routing = model.__routing__
        if routing is not None:
            index['routing'] = routing

        if isinstance(model, ESModelTimestamp):
            document['modified_at'] = datetime.utcnow()

        self._actions.append({'index': index, })
        self._actions.append(document)

    async def delete(self, model: TModel):
        """
        Add the model to the bulk for deletion
        :param model: The model to add for deletion
        """
        delete = {
            '_index': model.ESConfig.index,
            '_id': model.__id__
        }
        routing = model.__routing__
        if routing is not None:
            delete['routing'] = routing

        self._actions.append({'delete': delete})
