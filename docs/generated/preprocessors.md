## AI Summary

A file named preprocessors.py.


### Function: build_preprocessors(md)

**Description:** Build and return the default set of preprocessors used by Markdown. 

## Class: Preprocessor

**Description:** Preprocessors are run after the text is broken into lines.

Each preprocessor implements a `run` method that takes a pointer to a
list of lines of the document, modifies it as necessary and returns
either the same pointer or a pointer to a new list.

Preprocessors must extend `Preprocessor`.

## Class: NormalizeWhitespace

**Description:** Normalize whitespace for consistent parsing. 

## Class: HtmlBlockPreprocessor

**Description:** Remove html blocks from the text and store them for later retrieval.

The raw HTML is stored in the [`htmlStash`][markdown.util.HtmlStash] of the
[`Markdown`][markdown.Markdown] instance.

### Function: run(self, lines)

**Description:** Each subclass of `Preprocessor` should override the `run` method, which
takes the document as a list of strings split by newlines and returns
the (possibly modified) list of lines.

### Function: run(self, lines)

### Function: run(self, lines)
