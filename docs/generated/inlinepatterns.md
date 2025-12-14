## AI Summary

A file named inlinepatterns.py.


### Function: build_inlinepatterns(md)

**Description:** Build the default set of inline patterns for Markdown.

The order in which processors and/or patterns are applied is very important - e.g. if we first replace
`http://.../` links with `<a>` tags and _then_ try to replace inline HTML, we would end up with a mess. So, we
apply the expressions in the following order:

* backticks and escaped characters have to be handled before everything else so that we can preempt any markdown
  patterns by escaping them;

* then we handle the various types of links (auto-links must be handled before inline HTML);

* then we handle inline HTML.  At this point we will simply replace all inline HTML strings with a placeholder
  and add the actual HTML to a stash;

* finally we apply strong, emphasis, etc.

### Function: dequote(string)

**Description:** Remove quotes from around a string.

## Class: EmStrongItem

**Description:** Emphasis/strong pattern item.

## Class: Pattern

**Description:** Base class that inline patterns subclass.

Inline patterns are handled by means of `Pattern` subclasses, one per regular expression.
Each pattern object uses a single regular expression and must support the following methods:
[`getCompiledRegExp`][markdown.inlinepatterns.Pattern.getCompiledRegExp] and
[`handleMatch`][markdown.inlinepatterns.Pattern.handleMatch].

All the regular expressions used by `Pattern` subclasses must capture the whole block.  For this
reason, they all start with `^(.*)` and end with `(.*)!`.  When passing a regular expression on
class initialization, the `^(.*)` and `(.*)!` are added automatically and the regular expression
is pre-compiled.

It is strongly suggested that the newer style [`markdown.inlinepatterns.InlineProcessor`][] that
use a more efficient and flexible search approach be used instead. However, the older style
`Pattern` remains for backward compatibility with many existing third-party extensions.

## Class: InlineProcessor

**Description:** Base class that inline processors subclass.

This is the newer style inline processor that uses a more
efficient and flexible search approach.

## Class: SimpleTextPattern

**Description:** Return a simple text of `group(2)` of a Pattern. 

## Class: SimpleTextInlineProcessor

**Description:** Return a simple text of `group(1)` of a Pattern. 

## Class: EscapeInlineProcessor

**Description:** Return an escaped character. 

## Class: SimpleTagPattern

**Description:** Return element of type `tag` with a text attribute of `group(3)`
of a Pattern.

## Class: SimpleTagInlineProcessor

**Description:** Return element of type `tag` with a text attribute of `group(2)`
of a Pattern.

## Class: SubstituteTagPattern

**Description:** Return an element of type `tag` with no children. 

## Class: SubstituteTagInlineProcessor

**Description:** Return an element of type `tag` with no children. 

## Class: BacktickInlineProcessor

**Description:** Return a `<code>` element containing the escaped matching text. 

## Class: DoubleTagPattern

**Description:** Return a ElementTree element nested in tag2 nested in tag1.

Useful for strong emphasis etc.

## Class: DoubleTagInlineProcessor

**Description:** Return a ElementTree element nested in tag2 nested in tag1.

Useful for strong emphasis etc.

## Class: HtmlInlineProcessor

**Description:** Store raw inline html and return a placeholder. 

## Class: AsteriskProcessor

**Description:** Emphasis processor for handling strong and em matches inside asterisks.

## Class: UnderscoreProcessor

**Description:** Emphasis processor for handling strong and em matches inside underscores.

## Class: LinkInlineProcessor

**Description:** Return a link element from the given match. 

## Class: ImageInlineProcessor

**Description:** Return a `img` element from the given match. 

## Class: ReferenceInlineProcessor

**Description:** Match to a stored reference and return link element. 

## Class: ShortReferenceInlineProcessor

**Description:** Short form of reference: `[google]`. 

## Class: ImageReferenceInlineProcessor

**Description:** Match to a stored reference and return `img` element. 

## Class: ShortImageReferenceInlineProcessor

**Description:** Short form of image reference: `![ref]`. 

## Class: AutolinkInlineProcessor

**Description:** Return a link Element given an auto-link (`<http://example/com>`). 

## Class: AutomailInlineProcessor

**Description:** Return a `mailto` link Element given an auto-mail link (`<foo@example.com>`).

### Function: __init__(self, pattern, md)

**Description:** Create an instant of an inline pattern.

Arguments:
    pattern: A regular expression that matches a pattern.
    md: An optional pointer to the instance of `markdown.Markdown` and is available as
        `self.md` on the class instance.

### Function: getCompiledRegExp(self)

**Description:** Return a compiled regular expression. 

### Function: handleMatch(self, m)

**Description:** Return a ElementTree element from the given match.

Subclasses should override this method.

Arguments:
    m: A match object containing a match of the pattern.

Returns: An ElementTree Element object.

### Function: type(self)

**Description:** Return class name, to define pattern type 

### Function: unescape(self, text)

**Description:** Return unescaped text given text with an inline placeholder. 

### Function: __init__(self, pattern, md)

**Description:** Create an instant of an inline processor.

Arguments:
    pattern: A regular expression that matches a pattern.
    md: An optional pointer to the instance of `markdown.Markdown` and is available as
        `self.md` on the class instance.

### Function: handleMatch(self, m, data)

**Description:** Return a ElementTree element from the given match and the
start and end index of the matched text.

If `start` and/or `end` are returned as `None`, it will be
assumed that the processor did not find a valid region of text.

Subclasses should override this method.

Arguments:
    m: A re match object containing a match of the pattern.
    data: The buffer currently under analysis.

Returns:
    el: The ElementTree element, text or None.
    start: The start of the region that has been matched or None.
    end: The end of the region that has been matched or None.

### Function: handleMatch(self, m)

**Description:** Return string content of `group(2)` of a matching pattern. 

### Function: handleMatch(self, m, data)

**Description:** Return string content of `group(1)` of a matching pattern. 

### Function: handleMatch(self, m, data)

**Description:** If the character matched by `group(1)` of a pattern is in [`ESCAPED_CHARS`][markdown.Markdown.ESCAPED_CHARS]
then return the integer representing the character's Unicode code point (as returned by [`ord`][]) wrapped
in [`util.STX`][markdown.util.STX] and [`util.ETX`][markdown.util.ETX].

If the matched character is not in [`ESCAPED_CHARS`][markdown.Markdown.ESCAPED_CHARS], then return `None`.

### Function: __init__(self, pattern, tag)

**Description:** Create an instant of an simple tag pattern.

Arguments:
    pattern: A regular expression that matches a pattern.
    tag: Tag of element.

### Function: handleMatch(self, m)

**Description:** Return [`Element`][xml.etree.ElementTree.Element] of type `tag` with the string in `group(3)` of a
matching pattern as the Element's text.

### Function: __init__(self, pattern, tag)

**Description:** Create an instant of an simple tag processor.

Arguments:
    pattern: A regular expression that matches a pattern.
    tag: Tag of element.

### Function: handleMatch(self, m, data)

**Description:** Return [`Element`][xml.etree.ElementTree.Element] of type `tag` with the string in `group(2)` of a
matching pattern as the Element's text.

### Function: handleMatch(self, m)

**Description:** Return empty [`Element`][xml.etree.ElementTree.Element] of type `tag`. 

### Function: handleMatch(self, m, data)

**Description:** Return empty [`Element`][xml.etree.ElementTree.Element] of type `tag`. 

### Function: __init__(self, pattern)

### Function: handleMatch(self, m, data)

**Description:** If the match contains `group(3)` of a pattern, then return a `code`
[`Element`][xml.etree.ElementTree.Element] which contains HTML escaped text (with
[`code_escape`][markdown.util.code_escape]) as an [`AtomicString`][markdown.util.AtomicString].

If the match does not contain `group(3)` then return the text of `group(1)` backslash escaped.

### Function: handleMatch(self, m)

**Description:** Return [`Element`][xml.etree.ElementTree.Element] in following format:
`<tag1><tag2>group(3)</tag2>group(4)</tag2>` where `group(4)` is optional.

### Function: handleMatch(self, m, data)

**Description:** Return [`Element`][xml.etree.ElementTree.Element] in following format:
`<tag1><tag2>group(2)</tag2>group(3)</tag2>` where `group(3)` is optional.

### Function: handleMatch(self, m, data)

**Description:** Store the text of `group(1)` of a pattern and return a placeholder string. 

### Function: unescape(self, text)

**Description:** Return unescaped text given text with an inline placeholder. 

### Function: backslash_unescape(self, text)

**Description:** Return text with backslash escapes undone (backslashes are restored). 

### Function: build_single(self, m, tag, idx)

**Description:** Return single tag.

### Function: build_double(self, m, tags, idx)

**Description:** Return double tag.

### Function: build_double2(self, m, tags, idx)

**Description:** Return double tags (variant 2): `<strong>text <em>text</em></strong>`.

### Function: parse_sub_patterns(self, data, parent, last, idx)

**Description:** Parses sub patterns.

`data`: text to evaluate.

`parent`: Parent to attach text and sub elements to.

`last`: Last appended child to parent. Can also be None if parent has no children.

`idx`: Current pattern index that was used to evaluate the parent.

### Function: build_element(self, m, builder, tags, index)

**Description:** Element builder.

### Function: handleMatch(self, m, data)

**Description:** Parse patterns.

### Function: handleMatch(self, m, data)

**Description:** Return an `a` [`Element`][xml.etree.ElementTree.Element] or `(None, None, None)`. 

### Function: getLink(self, data, index)

**Description:** Parse data between `()` of `[Text]()` allowing recursive `()`. 

### Function: getText(self, data, index)

**Description:** Parse the content between `[]` of the start of an image or link
resolving nested square brackets.

### Function: handleMatch(self, m, data)

**Description:** Return an `img` [`Element`][xml.etree.ElementTree.Element] or `(None, None, None)`. 

### Function: handleMatch(self, m, data)

**Description:** Return [`Element`][xml.etree.ElementTree.Element] returned by `makeTag` method or `(None, None, None)`.

### Function: evalId(self, data, index, text)

**Description:** Evaluate the id portion of `[ref][id]`.

If `[ref][]` use `[ref]`.

### Function: makeTag(self, href, title, text)

**Description:** Return an `a` [`Element`][xml.etree.ElementTree.Element]. 

### Function: evalId(self, data, index, text)

**Description:** Evaluate the id of `[ref]`.  

### Function: makeTag(self, href, title, text)

**Description:** Return an `img` [`Element`][xml.etree.ElementTree.Element]. 

### Function: evalId(self, data, index, text)

**Description:** Evaluate the id of `[ref]`.  

### Function: handleMatch(self, m, data)

**Description:** Return an `a` [`Element`][xml.etree.ElementTree.Element] of `group(1)`. 

### Function: handleMatch(self, m, data)

**Description:** Return an [`Element`][xml.etree.ElementTree.Element] containing a `mailto` link  of `group(1)`. 

### Function: get_stash(m)

### Function: get_stash(m)

### Function: _unescape(m)

### Function: codepoint2name(code)

**Description:** Return entity definition by code, or the code if not defined.
