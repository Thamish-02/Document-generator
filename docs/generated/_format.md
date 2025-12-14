## AI Summary

A file named _format.py.


## Class: FormatChecker

**Description:** A ``format`` property checker.

JSON Schema does not mandate that the ``format`` property actually do any
validation. If validation is desired however, instances of this class can
be hooked into validators to enable format validation.

`FormatChecker` objects always return ``True`` when asked about
formats that they do not know how to validate.

To add a check for a custom format use the `FormatChecker.checks`
decorator.

Arguments:

    formats:

        The known formats to validate. This argument can be used to
        limit which formats will be used during validation.

### Function: _checks_drafts(name, draft3, draft4, draft6, draft7, draft201909, draft202012, raises)

### Function: is_email(instance)

### Function: is_ipv4(instance)

### Function: is_ipv6(instance)

### Function: is_regex(instance)

### Function: is_date(instance)

### Function: is_draft3_time(instance)

### Function: is_uuid(instance)

### Function: __init__(self, formats)

### Function: __repr__(self)

### Function: checks(self, format, raises)

**Description:** Register a decorated function as validating a new format.

Arguments:

    format:

        The format that the decorated function will check.

    raises:

        The exception(s) raised by the decorated function when an
        invalid instance is found.

        The exception object will be accessible as the
        `jsonschema.exceptions.ValidationError.cause` attribute of the
        resulting validation error.

### Function: cls_checks(cls, format, raises)

### Function: _cls_checks(cls, format, raises)

### Function: check(self, instance, format)

**Description:** Check whether the instance conforms to the given format.

Arguments:

    instance (*any primitive type*, i.e. str, number, bool):

        The instance to check

    format:

        The format that instance should conform to

Raises:

    FormatError:

        if the instance does not conform to ``format``

### Function: conforms(self, instance, format)

**Description:** Check whether the instance conforms to the given format.

Arguments:

    instance (*any primitive type*, i.e. str, number, bool):

        The instance to check

    format:

        The format that instance should conform to

Returns:

    bool: whether it conformed

### Function: wrap(func)

### Function: is_host_name(instance)

### Function: is_idn_host_name(instance)

### Function: is_iri(instance)

### Function: is_iri_reference(instance)

### Function: is_uri(instance)

### Function: is_uri_reference(instance)

### Function: is_datetime(instance)

### Function: is_time(instance)

### Function: is_css21_color(instance)

### Function: is_json_pointer(instance)

### Function: is_relative_json_pointer(instance)

### Function: is_uri_template(instance)

### Function: is_duration(instance)

### Function: _checks(func)

### Function: _checks(func)

### Function: is_uri(instance)

### Function: is_uri_reference(instance)

### Function: is_iri(instance)

### Function: is_iri_reference(instance)
