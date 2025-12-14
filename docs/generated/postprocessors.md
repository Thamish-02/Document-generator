## AI Summary

A file named postprocessors.py.


### Function: build_postprocessors(md)

**Description:** Build the default postprocessors for Markdown. 

## Class: Postprocessor

**Description:** Postprocessors are run after the ElementTree it converted back into text.

Each Postprocessor implements a `run` method that takes a pointer to a
text string, modifies it as necessary and returns a text string.

Postprocessors must extend `Postprocessor`.

## Class: RawHtmlPostprocessor

**Description:** Restore raw html to the document. 

## Class: AndSubstitutePostprocessor

**Description:** Restore valid entities 

## Class: UnescapePostprocessor

**Description:** Restore escaped chars. 

### Function: run(self, text)

**Description:** Subclasses of `Postprocessor` should implement a `run` method, which
takes the html document as a single text string and returns a
(possibly modified) string.

### Function: run(self, text)

**Description:** Iterate over html stash and restore html. 

### Function: isblocklevel(self, html)

**Description:** Check is block of HTML is block-level. 

### Function: stash_to_string(self, text)

**Description:** Convert a stashed object to a string. 

### Function: run(self, text)

### Function: unescape(self, m)

### Function: run(self, text)

### Function: substitute_match(m)
