## AI Summary

A file named psLib.py.


## Class: PSTokenError

## Class: PSError

## Class: PSTokenizer

## Class: PSInterpreter

### Function: unpack_item(item)

### Function: suckfont(data, encoding)

### Function: __init__(self, buf, encoding)

### Function: read(self, n)

**Description:** Read at most 'n' bytes from the buffer, or less if the read
hits EOF before obtaining 'n' bytes.
If 'n' is negative or omitted, read all data until EOF is reached.

### Function: close(self)

### Function: getnexttoken(self, len, ps_special, stringmatch, hexstringmatch, commentmatch, endmatch)

### Function: skipwhite(self, whitematch)

### Function: starteexec(self)

### Function: stopeexec(self)

### Function: __init__(self, encoding)

### Function: fillsystemdict(self)

### Function: suckoperators(self, systemdict, klass)

### Function: interpret(self, data, getattr)

### Function: handle_object(self, object)

### Function: call_procedure(self, proc)

### Function: resolve_name(self, name)

### Function: do_token(self, token, int, float, ps_name, ps_integer, ps_real)

### Function: do_comment(self, token)

### Function: do_literal(self, token)

### Function: do_string(self, token)

### Function: do_hexstring(self, token)

### Function: do_special(self, token)

### Function: push(self, object)

### Function: pop(self)

### Function: do_makearray(self)

### Function: close(self)

**Description:** Remove circular references.
