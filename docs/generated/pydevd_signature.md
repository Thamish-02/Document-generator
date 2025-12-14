## AI Summary

A file named pydevd_signature.py.


## Class: Signature

### Function: get_type_of_value(value, ignore_module_name, recursive)

### Function: _modname(path)

**Description:** Return a plausible module name for the path

## Class: SignatureFactory

### Function: get_signature_info(signature)

### Function: get_frame_info(frame)

## Class: CallSignatureCache

### Function: create_signature_message(signature)

### Function: send_signature_call_trace(dbg, frame, filename)

### Function: send_signature_return_trace(dbg, frame, filename, return_value)

### Function: __init__(self, file, name)

### Function: add_arg(self, name, type)

### Function: set_args(self, frame, recursive)

### Function: __str__(self)

### Function: __init__(self)

### Function: create_signature(self, frame, filename, with_args)

### Function: file_module_function_of(self, frame)

### Function: __init__(self)

### Function: add(self, signature)

### Function: is_in_cache(self, signature)
