## AI Summary

A file named _pydev_completer.py.


## Class: _StartsWithFilter

**Description:** Used because we can't create a lambda that'll use an outer scope in jython 2.1

## Class: Completer

### Function: generate_completions(frame, act_tok)

**Description:** :return list(tuple(method_name, docstring, parameters, completion_type))

method_name: str
docstring: str
parameters: str -- i.e.: "(a, b)"
completion_type is an int
    See: _pydev_bundle._pydev_imports_tipper for TYPE_ constants

### Function: generate_completions_as_xml(frame, act_tok)

### Function: completions_to_xml(completions)

### Function: isidentifier(s)

### Function: extract_token_and_qualifier(text, line, column)

**Description:** Extracts the token a qualifier from the text given the line/colum
(see test_extract_token_and_qualifier for examples).

:param unicode text:
:param int line: 0-based
:param int column: 0-based

### Function: __init__(self, start_with)

### Function: __call__(self, name)

### Function: __init__(self, namespace, global_namespace)

**Description:** Create a new completer for the command line.

Completer([namespace,global_namespace]) -> completer instance.

If unspecified, the default namespace where completions are performed
is __main__ (technically, __main__.__dict__). Namespaces should be
given as dictionaries.

An optional second namespace can be given.  This allows the completer
to handle cases where both the local and global scopes need to be
distinguished.

Completer instances should be used as the completion mechanism of
readline via the set_completer() call:

readline.set_completer(Completer(my_namespace).complete)

### Function: complete(self, text)

**Description:** Return the next possible completion for 'text'.

This is called successively with state == 0, 1, 2, ... until it
returns None.  The completion should begin with 'text'.

### Function: global_matches(self, text)

**Description:** Compute matches when text is a simple name.

Return a list of all keywords, built-in functions and names currently
defined in self.namespace or self.global_namespace that match.

### Function: attr_matches(self, text)

**Description:** Compute matches when text contains a dot.

Assuming the text is of the form NAME.NAME....[NAME], and is
evaluatable in self.namespace or self.global_namespace, it will be
evaluated and its attributes (as revealed by dir()) are used as
possible completions.  (For class instances, class members are are
also considered.)

WARNING: this can still invoke arbitrary C code, if an object
with a __getattr__ hook is evaluated.

### Function: get_item(obj, attr)
