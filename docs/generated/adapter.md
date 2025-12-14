## AI Summary

A file named adapter.py.


### Function: code_to_line(code, cursor_pos)

**Description:** Turn a multiline code block and cursor position into a single line
and new cursor position.

For adapting ``complete_`` and ``object_info_request``.

### Function: extract_oname_v4(code, cursor_pos)

**Description:** Reimplement token-finding logic from IPython 2.x javascript

for adapting object_info_request from v5 to v4

## Class: Adapter

**Description:** Base class for adapting messages

Override message_type(msg) methods to create adapters.

### Function: _version_str_to_list(version)

**Description:** convert a version string to a list of ints

non-int segments are excluded

## Class: V5toV4

**Description:** Adapt msg protocol v5 to v4

## Class: V4toV5

**Description:** Convert msg spec V4 to V5

### Function: adapt(msg, to_version)

**Description:** Adapt a single message to a target version

Parameters
----------

msg : dict
    A Jupyter message.
to_version : int, optional
    The target major version.
    If unspecified, adapt to the current version.

Returns
-------

msg : dict
    A Jupyter message appropriate in the new version.

### Function: update_header(self, msg)

**Description:** Update the header.

### Function: update_metadata(self, msg)

**Description:** Update the metadata.

### Function: update_msg_type(self, msg)

**Description:** Update the message type.

### Function: handle_reply_status_error(self, msg)

**Description:** This will be called *instead of* the regular handler

on any reply with status != ok

### Function: __call__(self, msg)

### Function: update_header(self, msg)

**Description:** Update the header.

### Function: kernel_info_reply(self, msg)

**Description:** Handle a kernel info reply.

### Function: execute_request(self, msg)

**Description:** Handle an execute request.

### Function: execute_reply(self, msg)

**Description:** Handle an execute reply.

### Function: complete_request(self, msg)

**Description:** Handle a complete request.

### Function: complete_reply(self, msg)

**Description:** Handle a complete reply.

### Function: object_info_request(self, msg)

**Description:** Handle an object info request.

### Function: object_info_reply(self, msg)

**Description:** inspect_reply can't be easily backward compatible

### Function: stream(self, msg)

**Description:** Handle a stream message.

### Function: display_data(self, msg)

**Description:** Handle a display data message.

### Function: input_request(self, msg)

**Description:** Handle an input request.

### Function: update_header(self, msg)

**Description:** Update the header.

### Function: kernel_info_reply(self, msg)

**Description:** Handle a kernel info reply.

### Function: execute_request(self, msg)

**Description:** Handle an execute request.

### Function: execute_reply(self, msg)

**Description:** Handle an execute reply.

### Function: complete_request(self, msg)

**Description:** Handle a complete request.

### Function: complete_reply(self, msg)

**Description:** Handle a complete reply.

### Function: inspect_request(self, msg)

**Description:** Handle an inspect request.

### Function: inspect_reply(self, msg)

**Description:** inspect_reply can't be easily backward compatible

### Function: stream(self, msg)

**Description:** Handle a stream message.

### Function: display_data(self, msg)

**Description:** Handle display data.

### Function: input_request(self, msg)

**Description:** Handle an input request.
