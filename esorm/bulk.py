"""
Bulk operation for ElasticSearch
"""
from .error import BulkError, BulkOperationError
from .utils import utcnow

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
        self._models_to_index = []
        if wait_for:
            self._bulk_kwargs['refresh'] = 'wait_for'

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Do the bulk operation
        :param exc_type: Exception type
        :param exc_val: Exception value
        :param exc_tb: Exception traceback
        :return
        """
        # Do the bulk operation if no exception is raised
        if exc_val is None:
            res = await es.bulk(operations=self._actions, **self._bulk_kwargs)
            errors = []
            for idx, item in enumerate(res['items']):
                model = self._models_to_index[idx]
                for action, result in item.items():
                    model._id = result.get('_id', None)

                    if 'error' in result:
                        error: BulkOperationError = {
                            'status': result['status'],
                            'type': result['error']['type'],
                            'reason': result['error']['reason'],
                            'model': model
                        }
                        errors.append(error)

                    elif action == 'index':
                        model._seq_no = result['_seq_no']
                        model._primary_term = result['_primary_term']
                        model._version = result['_version']
            if errors:
                raise BulkError(errors)

        # The exceptions are not handled here, propagate them
        return False

    # noinspection PyProtectedMember
    async def save(self, model: TModel):
        """
        Add the model to the bulk for saving

        If the model is from ES (get or search, so it has _seq_no and _primary_term), it will
        use optimistic concurrency check, so it will only update the document if the _seq_no and
        _primary_term are the same as the document in the index.

        If the model is an ESModelTimestamp, it will update the modified_at field to the current
        time and if the created_at field is not already set, it will set it to the current time too.

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

        # Optimistic concurrency check
        if model._primary_term and model._seq_no:
            index['if_primary_term'] = model._primary_term
            index['if_seq_no'] = model._seq_no

        # Support for ESModelTimestamp
        if isinstance(model, ESModelTimestamp):
            document['modified_at'] = utcnow()
            # Support for created_at field
            if not model.created_at:
                document['created_at'] = document['modified_at']

        # Save model for later to update private fields after the bulk operation
        self._models_to_index.append(model)

        # Add action and document to the bulk
        self._actions.append({'index': index})
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

        # Save model for later to update private fields after the bulk operation
        self._models_to_index.append(model)

        # Add action
        self._actions.append({'delete': delete})
