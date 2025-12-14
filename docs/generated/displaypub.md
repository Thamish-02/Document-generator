## AI Summary

A file named displaypub.py.


## Class: DisplayPublisher

**Description:** A traited class that publishes display data to frontends.

Instances of this class are created by the main IPython object and should
be accessed there.

## Class: CapturingDisplayPublisher

**Description:** A DisplayPublisher that stores

### Function: __init__(self, shell)

### Function: _validate_data(self, data, metadata)

**Description:** Validate the display data.

Parameters
----------
data : dict
    The formata data dictionary.
metadata : dict
    Any metadata for the data.

### Function: publish(self, data, metadata, source)

**Description:** Publish data and metadata to all frontends.

See the ``display_data`` message in the messaging documentation for
more details about this message type.

The following MIME types are currently implemented:

* text/plain
* text/html
* text/markdown
* text/latex
* application/json
* application/javascript
* image/png
* image/jpeg
* image/svg+xml

Parameters
----------
data : dict
    A dictionary having keys that are valid MIME types (like
    'text/plain' or 'image/svg+xml') and values that are the data for
    that MIME type. The data itself must be a JSON'able data
    structure. Minimally all data should have the 'text/plain' data,
    which can be displayed by all frontends. If more than the plain
    text is given, it is up to the frontend to decide which
    representation to use.
metadata : dict
    A dictionary for metadata related to the data. This can contain
    arbitrary key, value pairs that frontends can use to interpret
    the data.  Metadata specific to each mime-type can be specified
    in the metadata dict with the same mime-type keys as
    the data itself.
source : str, deprecated
    Unused.
transient : dict, keyword-only
    A dictionary for transient data.
    Data in this dictionary should not be persisted as part of saving this output.
    Examples include 'display_id'.
update : bool, keyword-only, default: False
    If True, only update existing outputs with the same display_id,
    rather than creating a new output.

### Function: clear_output(self, wait)

**Description:** Clear the output of the cell receiving output.

### Function: publish(self, data, metadata, source)

### Function: clear_output(self, wait)
