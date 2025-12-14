## AI Summary

A file named validator.py.


### Function: _relax_additional_properties(obj)

**Description:** relax any `additionalProperties`

### Function: _allow_undefined(schema)

### Function: get_validator(version, version_minor, relax_add_props, name)

**Description:** Load the JSON schema into a Validator

### Function: _get_schema_json(v, version, version_minor)

**Description:** Gets the json schema from a given imported library and nbformat version.

### Function: isvalid(nbjson, ref, version, version_minor)

**Description:** Checks whether the given notebook JSON conforms to the current
notebook format schema. Returns True if the JSON is valid, and
False otherwise.

To see the individual errors that were encountered, please use the
`validate` function instead.

### Function: _format_as_index(indices)

**Description:** (from jsonschema._utils.format_as_index, copied to avoid relying on private API)

Construct a single string containing indexing operations for the indices.

For example, [1, 2, "foo"] -> [1][2]["foo"]

### Function: _truncate_obj(obj)

**Description:** Truncate objects for use in validation tracebacks

Cell and output lists are squashed, as are long strings, lists, and dicts.

## Class: NotebookValidationError

**Description:** Schema ValidationError with truncated representation

to avoid massive verbose tracebacks.

### Function: better_validation_error(error, version, version_minor)

**Description:** Get better ValidationError on oneOf failures

oneOf errors aren't informative.
if it's a cell type or output_type error,
try validating directly based on the type for a better error message

### Function: normalize(nbdict, version, version_minor)

**Description:** Normalise a notebook prior to validation.

This tries to implement a couple of normalisation steps to standardise
notebooks and make validation easier.

You should in general not rely on this function and make sure the notebooks
that reach nbformat are already in a normal form. If not you likely have a bug,
and may have security issues.

Parameters
----------
nbdict : dict
    notebook document
version : int
version_minor : int
relax_add_props : bool
    Whether to allow extra property in the Json schema validating the
    notebook.
strip_invalid_metadata : bool
    Whether to strip metadata that does not exist in the Json schema when
    validating the notebook.

Returns
-------
changes : int
    number of changes in the notebooks
notebook : dict
    deep-copy of the original object with relevant changes.

### Function: _normalize(nbdict, version, version_minor, repair_duplicate_cell_ids, relax_add_props, strip_invalid_metadata)

**Description:** Private normalisation routine.

This function attempts to normalize the `nbdict` passed to it.

As `_normalize()` is currently used both in `validate()` (for
historical reasons), and in the `normalize()` public function,
`_normalize()` does currently mutate `nbdict`.
Ideally, once `validate()` stops calling `_normalize()`, `_normalize()`
may stop mutating `nbdict`.

### Function: _dep_warn(field)

### Function: validate(nbdict, ref, version, version_minor, relax_add_props, nbjson, repair_duplicate_cell_ids, strip_invalid_metadata)

**Description:** Checks whether the given notebook dict-like object
conforms to the relevant notebook format schema.

Parameters
----------
nbdict : dict
    notebook document
ref : optional, str
    reference to the subset of the schema we want to validate against.
    for example ``"markdown_cell"``, `"code_cell"` ....
version : int
version_minor : int
relax_add_props : bool
    Whether to allow extra properties in the JSON schema validating the notebook.
    When True, all known fields are validated, but unknown fields are ignored.
nbjson
repair_duplicate_cell_ids : bool
    Deprecated since 5.5.0 - will be removed in the future.
strip_invalid_metadata : bool
    Deprecated since 5.5.0 - will be removed in the future.

Returns
-------
None

Raises
------
ValidationError if not valid.

Notes
-----
Prior to Nbformat 5.5.0 the `validate` and `isvalid` method would silently
try to fix invalid notebook and mutate arguments. This behavior is deprecated
and will be removed in a near future.

Please explicitly call `normalize` if you need to normalize notebooks.

### Function: _get_errors(nbdict, version, version_minor, relax_add_props)

### Function: _strip_invalida_metadata(nbdict, version, version_minor, relax_add_props)

**Description:** This function tries to extract metadata errors from the validator and fix
them if necessary. This mostly mean stripping unknown keys from metadata
fields, or removing metadata fields altogether.

Parameters
----------
nbdict : dict
    notebook document
version : int
version_minor : int
relax_add_props : bool
    Whether to allow extra property in the Json schema validating the
    notebook.

Returns
-------
int
    number of modifications

### Function: iter_validate(nbdict, ref, version, version_minor, relax_add_props, nbjson, strip_invalid_metadata)

**Description:** Checks whether the given notebook dict-like object conforms to the
relevant notebook format schema.

Returns a generator of all ValidationErrors if not valid.

Notes
-----
To fix: For security reasons, this function should *never* mutate its `nbdict` argument, and
should *never* try to validate a mutated or modified version of its notebook.

### Function: __init__(self, original, ref)

**Description:** Initialize the error class.

### Function: __getattr__(self, key)

**Description:** Get an attribute from the error.

### Function: __unicode__(self)

**Description:** Custom str for validation errors

avoids dumping full schema and notebook to logs
