## AI Summary

A file named html5lib_shim.py.


## Class: InputStreamWithMemory

**Description:** Wraps an HTMLInputStream to remember characters since last <

This wraps existing HTMLInputStream classes to keep track of the stream
since the last < which marked an open tag state.

## Class: BleachHTMLTokenizer

**Description:** Tokenizer that doesn't consume character entities

## Class: BleachHTMLParser

**Description:** Parser that uses BleachHTMLTokenizer

### Function: convert_entity(value)

**Description:** Convert an entity (minus the & and ; part) into what it represents

This handles numeric, hex, and text entities.

:arg value: the string (minus the ``&`` and ``;`` part) to convert

:returns: unicode character or None if it's an ambiguous ampersand that
    doesn't match a character entity

### Function: convert_entities(text)

**Description:** Converts all found entities in the text

:arg text: the text to convert entities in

:returns: unicode text with converted entities

### Function: match_entity(stream)

**Description:** Returns first entity in stream or None if no entity exists

Note: For Bleach purposes, entities must start with a "&" and end with a
";". This ignores ambiguous character entities that have no ";" at the end.

:arg stream: the character stream

:returns: the entity string without "&" or ";" if it's a valid character
    entity; ``None`` otherwise

### Function: next_possible_entity(text)

**Description:** Takes a text and generates a list of possible entities

:arg text: the text to look at

:returns: generator where each part (except the first) starts with an
    "&"

## Class: BleachHTMLSerializer

**Description:** HTMLSerializer that undoes & -> &amp; in attributes and sets
escape_rcdata to True

### Function: __init__(self, inner_stream)

### Function: errors(self)

### Function: charEncoding(self)

### Function: changeEncoding(self)

### Function: char(self)

### Function: charsUntil(self, characters, opposite)

### Function: unget(self, char)

### Function: get_tag(self)

**Description:** Returns the stream history since last '<'

Since the buffer starts at the last '<' as as seen by tagOpenState(),
we know that everything from that point to when this method is called
is the "tag" that is being tokenized.

### Function: start_tag(self)

**Description:** Resets stream history to just '<'

This gets called by tagOpenState() which marks a '<' that denotes an
open tag. Any time we see that, we reset the buffer.

### Function: __init__(self, consume_entities)

### Function: __iter__(self)

### Function: consumeEntity(self, allowedChar, fromAttribute)

### Function: tagOpenState(self)

### Function: emitCurrentToken(self)

### Function: __init__(self, tags, strip, consume_entities)

**Description:** :arg tags: set of allowed tags--everything else is either stripped or
    escaped; if None, then this doesn't look at tags at all
:arg strip: whether to strip disallowed tags (True) or escape them (False);
    if tags=None, then this doesn't have any effect
:arg consume_entities: whether to consume entities (default behavior) or
    leave them as is when tokenizing (BleachHTMLTokenizer-added behavior)

### Function: _parse(self, stream, innerHTML, container, scripting)

### Function: escape_base_amp(self, stoken)

**Description:** Escapes just bare & in HTML attribute values

### Function: serialize(self, treewalker, encoding)

**Description:** Wrap HTMLSerializer.serialize and conver & to &amp; in attribute values

Note that this converts & to &amp; in attribute values where the & isn't
already part of an unambiguous character entity.
