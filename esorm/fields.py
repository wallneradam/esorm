from typing import Any, Optional, Union
from functools import cached_property

from base64 import b64encode, b64decode

from pydantic.fields import Field as PField, FieldInfo
from pydantic_core import core_schema
from pydantic import BaseModel

__all__ = (
    'Keyword', 'Text', 'Binary', 'Byte', 'Short', 'Integer', 'Long', 'HalfFloat', 'Float', 'Double', 'LatLon',
    'keyword', 'text', 'binary', 'byte', 'short', 'int32', 'long', 'float16', 'float32', 'double', 'geo_point',
    'integer', 'half_float', 'int64', 'boolean',
    'Field', 'NumericField', 'TextField'
)


#
# Field Type Classes
#

class Keyword(str):
    """ Keyword Field """
    __es_type__ = 'keyword'

    @classmethod
    def __get_pydantic_core_schema__(cls, _, handler):
        return core_schema.no_info_after_validator_function(cls, handler(str))


class Text(str):
    """ Text Field """
    __es_type__ = 'text'

    @classmethod
    def __get_pydantic_core_schema__(cls, _, handler):
        return core_schema.no_info_after_validator_function(cls, handler(str))


class Binary(str):
    """
    Stores binary data as base64 encoded strings
    """
    __es_type__ = 'binary'

    @classmethod
    def __get_pydantic_core_schema__(cls, _, __):
        return core_schema.with_info_plain_validator_function(cls.validate_binary)

    @classmethod
    def __get_pydantic_json_schema__(cls, _, handler):
        # Use the same schema that would be used for `str`
        return handler(core_schema.str_schema())

    @classmethod
    def validate_binary(cls, v: Union[bytes, str], _) -> str:
        if isinstance(v, bytes):
            v = b64encode(v).decode('ascii')
        return cls(v)

    @cached_property
    def bytes(self) -> bytes:
        return b64decode(self)


class Byte(int):
    """ Byte Field """
    __es_type__ = 'byte'

    @classmethod
    def __get_pydantic_core_schema__(cls, _, handler):
        return core_schema.no_info_after_validator_function(cls, handler(int))


class Short(int):
    """ Short Field """
    __es_type__ = 'short'

    @classmethod
    def __get_pydantic_core_schema__(cls, _, handler):
        return core_schema.no_info_after_validator_function(cls, handler(int))


class Integer(int):
    """ Integer Field """
    __es_type__ = 'integer'

    @classmethod
    def __get_pydantic_core_schema__(cls, _, handler):
        return core_schema.no_info_after_validator_function(cls, handler(int))


class Long(int):
    """ Long Field """
    __es_type__ = 'long'

    @classmethod
    def __get_pydantic_core_schema__(cls, _, handler):
        return core_schema.no_info_after_validator_function(cls, handler(int))


class HalfFloat(float):
    """ Half Float Field """
    __es_type__ = 'half_float'

    @classmethod
    def __get_pydantic_core_schema__(cls, _, handler):
        return core_schema.no_info_after_validator_function(cls, handler(float))


class Float(float):
    """ Float Field """
    __es_type__ = 'float'

    @classmethod
    def __get_pydantic_core_schema__(cls, _, handler):
        return core_schema.no_info_after_validator_function(cls, handler(float))


class Double(float):
    """ Double Field """
    __es_type__ = 'double'

    @classmethod
    def __get_pydantic_core_schema__(cls, _, handler):
        return core_schema.no_info_after_validator_function(cls, handler(float))


class LatLon(BaseModel):
    """
    Geo Point Field - Latitude and Longitude
    """
    __es_type__ = 'geo_point'

    lat: float
    """Latitude Coordinate"""
    lon: float
    """Longitude Coordinate"""


#
# Field Types
#

# These types should be used in model definitions, because these can be interpreted by both Python and Pydantic

keyword = Union[Keyword, str]
""" Keyword type """
text = Union[Text, str]
""" Text type """
binary = Union[Binary, str]
""" Binary type """

byte = Union[Byte, int]
""" Byte type """
short = Union[Short, int]
""" Short type """
int32 = Union[Integer, int]
""" 32 bit integer type """
long = Union[Long, int]
""" 64 bit integer (long) type """

float16 = Union[HalfFloat, float]
""" 16 bit float type """
float32 = Union[Float, float]
""" 32 bit float type """
double = Union[Double, float]
""" 64 bit float (double) type """

# Aliases
integer = int32
int64 = long
boolean = bool
half_float = float16

geo_point = LatLon
""" Geo Point type """


# noinspection PyPep8Naming
def Field(
        default: Any,
        *,
        index: bool = True,
        alias: Optional[str] = None,
        # Other pydantic args
        title: Optional[str] = None,
        description: Optional[str] = None,
        exclude: Optional[bool] = None,
        include: Optional[bool] = None,
        frozen: bool = False,
        **extra
) -> FieldInfo:
    """
    Basic Field Info

    :param default: since this is replacing the field’s default, its first argument is used
        to set the default, use ellipsis (``...``) to indicate the field is required
    :param index: if this field should be indexed or not
    :param alias: the public name of the field
    :param title: can be any string, used in the schema
    :param description: can be any string, used in the schema
    :param exclude: exclude this field while dumping.
        Takes same values as the ``include`` and ``exclude`` arguments on the ``.dict`` method.
    :param include: include this field while dumping.
        Takes same values as the ``include`` and ``exclude`` arguments on the ``.dict`` method.
    :param frozen: if this field should be frozen or not
    :param extra: any additional keyword arguments will be added as is to the schema
    :return: A field info object
    """
    return PField(default, alias=alias,
                  title=title, description=description,
                  exclude=exclude, include=include, frozen=frozen,
                  index=index, json_schema_extra=extra)


# noinspection PyPep8Naming
def NumericField(
        default: Union[int, float],
        *,
        index: Optional[bool] = None,
        alias: Optional[str] = None,
        gt: Optional[float] = None,
        ge: Optional[float] = None,
        lt: Optional[float] = None,
        le: Optional[float] = None,
        multiple_of: Optional[float] = None,
        allow_inf_nan: Optional[bool] = None,
        max_digits: Optional[int] = None,
        decimal_places: Optional[int] = None,
        # Other pydantic args
        title: Optional[str] = None,
        description: Optional[str] = None,
        exclude: Optional[bool] = None,
        include: Optional[bool] = None,
        frozen: bool = False,
        **extra
) -> FieldInfo:
    """
    Numeric Field Info

    :param default: since this is replacing the field’s default, its first argument is used
        to set the default, use ellipsis (``...``) to indicate the field is required
    :param index: if this field should be indexed or not
    :param alias: the public name of the field
    :param gt: only applies to numbers, requires the field to be "greater than". The schema
        will have an ``exclusiveMinimum`` validation keyword
    :param ge: only applies to numbers, requires the field to be "greater than or equal to". The
      schema will have a ``minimum`` validation keyword
    :param lt: only applies to numbers, requires the field to be "less than". The schema
        will have an ``exclusiveMaximum`` validation keyword
    :param le: only applies to numbers, requires the field to be "less than or equal to". The
        schema will have a ``maximum`` validation keyword
    :param multiple_of: only applies to numbers, requires the field to be "a multiple of". The
        schema will have a ``multipleOf`` validation keyword
    :param allow_inf_nan: only applies to numbers, allows the field to be NaN or infinity (+inf or -inf),
        which is a valid Python float. Default True, set to False for compatibility with JSON.
    :param max_digits: only applies to Decimals, requires the field to have a maximum number
        of digits within the decimal. It does not include a zero before the decimal point or trailing decimal zeroes.
    :param decimal_places: only applies to Decimals, requires the field to have at most a number of decimal places
        allowed. It does not include trailing decimal zeroes.
    :param title: can be any string, used in the schema
    :param description: can be any string, used in the schema
    :param exclude: exclude this field while dumping.
        Takes same values as the ``include`` and ``exclude`` arguments on the ``.dict`` method.
    :param include: include this field while dumping.
        Takes same values as the ``include`` and ``exclude`` arguments on the ``.dict`` method.
    :param frozen: if this field should be frozen or not
    :param extra: any additional keyword arguments will be added as is to the schema
    :return: A field info object
    """
    extra = dict(extra)
    if index is not None:
        extra['index'] = index
    return PField(default, alias=alias,
                  gt=gt, ge=ge, lt=lt, le=le,
                  multiple_of=multiple_of, allow_inf_nan=allow_inf_nan,
                  max_digits=max_digits, decimal_places=decimal_places,
                  title=title, description=description,
                  exclude=exclude, include=include, frozen=frozen,
                  json_schema_extra=extra)


# noinspection PyPep8Naming
def TextField(
        default: str,
        *,
        index: bool = True,
        alias: Optional[str] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        regex: Optional[str] = None,
        # Other pydantic args
        title: Optional[str] = None,
        description: Optional[str] = None,
        exclude: Optional[bool] = None,
        include: Optional[bool] = None,
        frozen: bool = False,
        **extra
) -> FieldInfo:
    """
    Text Field Info

    :param default: since this is replacing the field’s default, its first argument is used
        to set the default, use ellipsis (``...``) to indicate the field is required
    :param index: if this field should be indexed or not
    :param alias: the public name of the field
    :param min_length: only applies to strings, requires the field to have a minimum length. The
        schema will have a ``minLength`` validation keyword
    :param max_length: only applies to strings, requires the field to have a maximum length. The
        schema will have a ``maxLength`` validation keyword
    :param regex: only applies to strings, requires the field match against a regular expression
        pattern string. The schema will have a ``pattern`` validation keyword
    :param title: can be any string, used in the schema
    :param description: can be any string, used in the schema
    :param exclude: exclude this field while dumping.
        Takes same values as the ``include`` and ``exclude`` arguments on the ``.dict`` method.
    :param include: include this field while dumping.
        Takes same values as the ``include`` and ``exclude`` arguments on the ``.dict`` method.
    :param frozen: if this field should be frozen or not
    :param extra: any additional keyword arguments will be added as is to the schema
    :return: A field info object
    """
    extra = dict(extra)
    if index is not None:
        extra['index'] = index
    return PField(default, alias=alias,
                  min_length=min_length, max_length=max_length,
                  regex=regex,
                  title=title, description=description,
                  exclude=exclude, include=include, frozen=frozen,
                  json_schema_extra=extra)
