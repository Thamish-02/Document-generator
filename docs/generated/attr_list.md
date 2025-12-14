## AI Summary

A file named attr_list.py.


### Function: _handle_double_quote(s, t)

### Function: _handle_single_quote(s, t)

### Function: _handle_key_value(s, t)

### Function: _handle_word(s, t)

### Function: get_attrs_and_remainder(attrs_string)

**Description:** Parse attribute list and return a list of attribute tuples.

Additionally, return any text that remained after a curly brace. In typical cases, its presence
should mean that the input does not match the intended attribute list syntax.

### Function: get_attrs(str)

**Description:** Soft-deprecated. Prefer `get_attrs_and_remainder`. 

### Function: isheader(elem)

## Class: AttrListTreeprocessor

## Class: AttrListExtension

**Description:** Attribute List extension for Python-Markdown 

### Function: makeExtension()

### Function: run(self, doc)

### Function: assign_attrs(self, elem, attrs_string)

**Description:** Assign `attrs` to element.

If the `attrs_string` has an extra closing curly brace, the remaining text is returned.

The `strict` argument controls whether to still assign `attrs` if there is a remaining `}`.

### Function: sanitize_name(self, name)

**Description:** Sanitize name as 'an XML Name, minus the `:`.'
See <https://www.w3.org/TR/REC-xml-names/#NT-NCName>.

### Function: extendMarkdown(self, md)
