"""
Mapping comparison

The mappings generated from the models are compared with the mappings of the existing indices. The result is a list
of differences (:class:`MappingDiff`), and a mapping patch, which contains only the changes that can be applied to
the existing index without reindexing (new fields, new subfields and parameters which are updatable in place).

The mappings returned by the server contain default values and other parameters the models don't set, so only the
parameters set by the model are compared.
"""
from typing import Any, Dict, List, Literal, Optional, Tuple

from dataclasses import dataclass

__all__ = ['MappingDiff', 'MappingDiffKind', 'diff_mappings', 'format_diffs']

MappingDiffKind = Literal['new_index', 'new_field', 'update', 'conflict', 'extra']
"""
Kind of a mapping difference:

- `new_index`: the index does not exist, it will be created
- `new_field`: the field (or subfield) is not in the index, it can be added
- `update`: a parameter of the field differs, but it can be updated in place
- `conflict`: the field is incompatible with the index (e.g. its type changed), it needs a new index and reindexing
- `extra`: the field is in the index, but not in the model, it is harmless (fields can't be removed from an index)
"""

# Parameters which can be changed on an existing field
_UPDATABLE_PARAMS = frozenset((
    'search_analyzer', 'search_quote_analyzer', 'ignore_above', 'ignore_malformed', 'meta', 'dynamic',
))

# Structural keys, which are compared recursively
_STRUCTURAL_KEYS = ('type', 'properties', 'fields')

# Default values of parameters, the server doesn't return them
_PARAM_DEFAULTS: Dict[str, Any] = {
    'index': True,
    'store': False,
    'doc_values': True,
    'analyzer': 'standard',
}

_MISSING = object()


@dataclass(frozen=True)
class MappingDiff:
    """
    A difference between the mapping of a model and the mapping of its index
    """

    index: str
    """ The name of the index """

    kind: MappingDiffKind
    """ The kind of the difference """

    field: Optional[str] = None
    """ The path of the field, e.g. `address.city` or `title.keyword` (subfield), None for `new_index` """

    param: Optional[str] = None
    """ The name of the differing parameter (e.g. `type`, `analyzer`) for `update` and `conflict` """

    expected: Any = None
    """ The mapping (or parameter value) of the model """

    actual: Any = None
    """ The mapping (or parameter value) in the index """

    @property
    def is_conflict(self) -> bool:
        """ True if the difference can't be applied to the existing index """
        return self.kind == 'conflict'

    def __str__(self) -> str:
        where = f"`{self.index}`" + (f" field `{self.field}`" if self.field else '')
        if self.kind == 'new_index':
            return f"{where}: the index does not exist"
        if self.kind == 'new_field':
            return f"{where}: new field ({_short(self.expected)})"
        if self.kind == 'update':
            return f"{where}: `{self.param}` can be updated: {_short(self.actual)} -> {_short(self.expected)}"
        if self.kind == 'conflict':
            return (f"{where}: `{self.param}` is {_short(self.actual)} in the index, but {_short(self.expected)} "
                    f"in the model (needs a new index and reindexing)")
        return f"{where}: the field is in the index, but not in the model"


def _short(value: Any) -> str:
    """ Short representation of a value """
    if value is _MISSING or value is None:
        return 'not set'
    if isinstance(value, dict) and 'type' in value:
        return str(value['type'])
    if isinstance(value, dict) and 'properties' in value:
        return 'object'
    return repr(value)


def format_diffs(diffs: List[MappingDiff]) -> str:
    """
    Format mapping differences as a readable, multiline list

    :param diffs: The differences
    :return: The formatted list
    """
    return '\n'.join(f"  - [{d.kind}] {d}" for d in diffs)


def _normalize(value: Any) -> Any:
    """ Normalize a scalar parameter value, the server returns booleans and numbers instead of strings """
    if isinstance(value, str):
        if value in ('true', 'false'):
            return value == 'true'
        if value.isdigit():
            return int(value)
    return value


def _value_equal(expected: Any, actual: Any) -> bool:
    """ Compare parameter values, from dicts only the keys set in the expected one are compared """
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and _value_equal(v, actual[k])
                                                for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return (isinstance(actual, (list, tuple)) and len(expected) == len(actual)
                and all(_value_equal(e, a) for e, a in zip(expected, actual)))
    return _normalize(expected) == _normalize(actual)


def _actual_param(actual: Dict[str, Any], param: str) -> Any:
    """ Get a parameter of an existing field, or its default value if the server doesn't return it """
    if param in actual:
        return actual[param]
    if param == 'search_analyzer':
        return actual.get('analyzer', _PARAM_DEFAULTS['analyzer'])
    if param == 'space_type' and isinstance(actual.get('method'), dict):  # OpenSearch knn_vector
        return actual['method'].get('space_type', _MISSING)
    return _PARAM_DEFAULTS.get(param, _MISSING)


def _field_type(mapping: Dict[str, Any]) -> Optional[str]:
    """ The type of a field, objects have no explicit type """
    return mapping.get('type', 'object' if 'properties' in mapping else None)


def _diff_field(index: str, path: str, expected: Dict[str, Any], actual: Optional[Dict[str, Any]],
                diffs: List[MappingDiff]) -> Optional[Dict[str, Any]]:
    """
    Compare the mapping of a field

    :return: The patch of the field, which can be sent to the server, or None if nothing to change
    """
    if actual is None:
        diffs.append(MappingDiff(index, 'new_field', path, expected=expected))
        return expected

    expected_type = _field_type(expected)
    actual_type = actual.get('type', 'object')
    if expected_type is not None and expected_type != actual_type:
        diffs.append(MappingDiff(index, 'conflict', path, 'type', expected_type, actual_type))
        return None

    updated = False
    conflicts = set()
    for param, value in expected.items():
        if param in _STRUCTURAL_KEYS:
            continue
        actual_value = _actual_param(actual, param)
        if _value_equal(value, actual_value):
            continue
        if param in _UPDATABLE_PARAMS:
            diffs.append(MappingDiff(index, 'update', path, param, value, actual_value))
            updated = True
        else:
            diffs.append(MappingDiff(index, 'conflict', path, param, value, actual_value))
            conflicts.add(param)

    sub_patches = {}
    for key in ('properties', 'fields'):
        sub_patch = _diff_properties(index, path + '.', expected.get(key) or {}, actual.get(key) or {}, diffs)
        if sub_patch:
            sub_patches[key] = sub_patch

    if not updated and not sub_patches:
        return None

    # The field is sent with all of its parameters, otherwise the server would reset the missing ones to their
    # defaults, the conflicting ones are sent as they are in the index, so the compatible changes can be applied
    patch = {}
    for param, value in expected.items():
        if param in ('properties', 'fields'):
            continue
        if param in conflicts:
            if param not in actual:
                continue
            value = actual[param]
        patch[param] = value
    patch.update(sub_patches)
    return patch


def _diff_properties(index: str, prefix: str, expected: Dict[str, Any], actual: Dict[str, Any],
                     diffs: List[MappingDiff]) -> Dict[str, Any]:
    """
    Compare properties (or subfields)

    :return: The patch of the properties
    """
    patch = {}
    for name, field in expected.items():
        field_patch = _diff_field(index, prefix + name, field, actual.get(name), diffs)
        if field_patch is not None:
            patch[name] = field_patch
    for name in sorted(actual.keys() - expected.keys()):
        diffs.append(MappingDiff(index, 'extra', prefix + name, actual=actual[name]))
    return patch


def diff_mappings(index: str, expected: Dict[str, Any], actual: Dict[str, Any]) \
        -> Tuple[List[MappingDiff], Dict[str, Any]]:
    """
    Compare the properties generated from a model with the properties of an existing index

    :param index: The name of the index
    :param expected: The properties generated from the model
    :param actual: The properties of the existing index
    :return: The list of differences, and the properties to send to the put mapping API (it contains only the
             new fields and the changed fields, conflicting parameters are left as they are in the index)
    """
    diffs: List[MappingDiff] = []
    patch = _diff_properties(index, '', expected, actual, diffs)
    return diffs, patch
