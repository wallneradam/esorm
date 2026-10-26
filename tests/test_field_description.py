"""
Field descriptions from variable docstrings, they don't need ElasticSearch
"""
import pytest

from esorm import ESModel


@pytest.fixture(autouse=True)
def restore_models():
    """ The tests define models, they must not stay in the model registry """
    from esorm.model import _ESModelMeta
    models = dict(_ESModelMeta.__models__)
    yield
    _ESModelMeta.__models__.clear()
    _ESModelMeta.__models__.update(models)


def test_docstring_description():
    class DescUser(ESModel):
        name: str
        """ The name of the user """
        age: int
        email: str
        """
        The email address
        """

    fields = DescUser.model_fields
    assert fields['name'].description == 'The name of the user'
    assert fields['age'].description is None
    assert fields['email'].description == 'The email address'


def test_ellipsis_after_field():
    class EllipsisUser(ESModel):
        name: str
        ...

    assert EllipsisUser.model_fields['name'].description is None


def test_number_after_field():
    class NumberUser(ESModel):
        name: str
        42
        age: int
        """ The age """

    fields = NumberUser.model_fields
    assert fields['name'].description is None
    assert fields['age'].description == 'The age'
