## AI Summary

A file named html5parser.py.


### Function: parse(doc, treebuilder, namespaceHTMLElements)

**Description:** Parse an HTML document as a string or file-like object into a tree

:arg doc: the document to parse as a string or file-like object

:arg treebuilder: the treebuilder to use when parsing

:arg namespaceHTMLElements: whether or not to namespace HTML elements

:returns: parsed tree

Example:

>>> from html5lib.html5parser import parse
>>> parse('<html><body><p>This is a doc</p></body></html>')
<Element u'{http://www.w3.org/1999/xhtml}html' at 0x7feac4909db0>

### Function: parseFragment(doc, container, treebuilder, namespaceHTMLElements)

**Description:** Parse an HTML fragment as a string or file-like object into a tree

:arg doc: the fragment to parse as a string or file-like object

:arg container: the container context to parse the fragment in

:arg treebuilder: the treebuilder to use when parsing

:arg namespaceHTMLElements: whether or not to namespace HTML elements

:returns: parsed tree

Example:

>>> from html5lib.html5libparser import parseFragment
>>> parseFragment('<b>this is a fragment</b>')
<Element u'DOCUMENT_FRAGMENT' at 0x7feac484b090>

### Function: method_decorator_metaclass(function)

## Class: HTMLParser

**Description:** HTML parser

Generates a tree structure from a stream of (possibly malformed) HTML.

### Function: getPhases(debug)

### Function: adjust_attributes(token, replacements)

### Function: impliedTagToken(name, type, attributes, selfClosing)

## Class: ParseError

**Description:** Error in parsed document

## Class: Decorated

### Function: __init__(self, tree, strict, namespaceHTMLElements, debug)

**Description:** :arg tree: a treebuilder class controlling the type of tree that will be
    returned. Built in treebuilders can be accessed through
    html5lib.treebuilders.getTreeBuilder(treeType)

:arg strict: raise an exception when a parse error is encountered

:arg namespaceHTMLElements: whether or not to namespace HTML elements

:arg debug: whether or not to enable debug mode which logs things

Example:

>>> from html5lib.html5parser import HTMLParser
>>> parser = HTMLParser()                     # generates parser with etree builder
>>> parser = HTMLParser('lxml', strict=True)  # generates parser with lxml builder which is strict

### Function: _parse(self, stream, innerHTML, container, scripting)

### Function: reset(self)

### Function: documentEncoding(self)

**Description:** Name of the character encoding that was used to decode the input stream, or
:obj:`None` if that is not determined yet

### Function: isHTMLIntegrationPoint(self, element)

### Function: isMathMLTextIntegrationPoint(self, element)

### Function: mainLoop(self)

### Function: parse(self, stream)

**Description:** Parse a HTML document into a well-formed tree

:arg stream: a file-like object or string containing the HTML to be parsed

    The optional encoding parameter must be a string that indicates
    the encoding.  If specified, that encoding will be used,
    regardless of any BOM or later declaration (such as in a meta
    element).

:arg scripting: treat noscript elements as if JavaScript was turned on

:returns: parsed tree

Example:

>>> from html5lib.html5parser import HTMLParser
>>> parser = HTMLParser()
>>> parser.parse('<html><body><p>This is a doc</p></body></html>')
<Element u'{http://www.w3.org/1999/xhtml}html' at 0x7feac4909db0>

### Function: parseFragment(self, stream)

**Description:** Parse a HTML fragment into a well-formed tree fragment

:arg container: name of the element we're setting the innerHTML
    property if set to None, default to 'div'

:arg stream: a file-like object or string containing the HTML to be parsed

    The optional encoding parameter must be a string that indicates
    the encoding.  If specified, that encoding will be used,
    regardless of any BOM or later declaration (such as in a meta
    element)

:arg scripting: treat noscript elements as if JavaScript was turned on

:returns: parsed tree

Example:

>>> from html5lib.html5libparser import HTMLParser
>>> parser = HTMLParser()
>>> parser.parseFragment('<b>this is a fragment</b>')
<Element u'DOCUMENT_FRAGMENT' at 0x7feac484b090>

### Function: parseError(self, errorcode, datavars)

### Function: adjustMathMLAttributes(self, token)

### Function: adjustSVGAttributes(self, token)

### Function: adjustForeignAttributes(self, token)

### Function: reparseTokenNormal(self, token)

### Function: resetInsertionMode(self)

### Function: parseRCDataRawtext(self, token, contentType)

### Function: log(function)

**Description:** Logger that records which phase processes each token

### Function: getMetaclass(use_metaclass, metaclass_func)

## Class: Phase

**Description:** Base class for helper object that implements each phase of processing
        

## Class: InitialPhase

## Class: BeforeHtmlPhase

## Class: BeforeHeadPhase

## Class: InHeadPhase

## Class: InHeadNoscriptPhase

## Class: AfterHeadPhase

## Class: InBodyPhase

## Class: TextPhase

## Class: InTablePhase

## Class: InTableTextPhase

## Class: InCaptionPhase

## Class: InColumnGroupPhase

## Class: InTableBodyPhase

## Class: InRowPhase

## Class: InCellPhase

## Class: InSelectPhase

## Class: InSelectInTablePhase

## Class: InForeignContentPhase

## Class: AfterBodyPhase

## Class: InFramesetPhase

## Class: AfterFramesetPhase

## Class: AfterAfterBodyPhase

## Class: AfterAfterFramesetPhase

### Function: __new__(meta, classname, bases, classDict)

### Function: wrapped(self)

### Function: __init__(self, parser, tree)

### Function: processEOF(self)

### Function: processComment(self, token)

### Function: processDoctype(self, token)

### Function: processCharacters(self, token)

### Function: processSpaceCharacters(self, token)

### Function: processStartTag(self, token)

### Function: startTagHtml(self, token)

### Function: processEndTag(self, token)

### Function: processSpaceCharacters(self, token)

### Function: processComment(self, token)

### Function: processDoctype(self, token)

### Function: anythingElse(self)

### Function: processCharacters(self, token)

### Function: processStartTag(self, token)

### Function: processEndTag(self, token)

### Function: processEOF(self)

### Function: insertHtmlElement(self)

### Function: processEOF(self)

### Function: processComment(self, token)

### Function: processSpaceCharacters(self, token)

### Function: processCharacters(self, token)

### Function: processStartTag(self, token)

### Function: processEndTag(self, token)

### Function: processEOF(self)

### Function: processSpaceCharacters(self, token)

### Function: processCharacters(self, token)

### Function: startTagHtml(self, token)

### Function: startTagHead(self, token)

### Function: startTagOther(self, token)

### Function: endTagImplyHead(self, token)

### Function: endTagOther(self, token)

### Function: processEOF(self)

### Function: processCharacters(self, token)

### Function: startTagHtml(self, token)

### Function: startTagHead(self, token)

### Function: startTagBaseLinkCommand(self, token)

### Function: startTagMeta(self, token)

### Function: startTagTitle(self, token)

### Function: startTagNoFramesStyle(self, token)

### Function: startTagNoscript(self, token)

### Function: startTagScript(self, token)

### Function: startTagOther(self, token)

### Function: endTagHead(self, token)

### Function: endTagHtmlBodyBr(self, token)

### Function: endTagOther(self, token)

### Function: anythingElse(self)

### Function: processEOF(self)

### Function: processComment(self, token)

### Function: processCharacters(self, token)

### Function: processSpaceCharacters(self, token)

### Function: startTagHtml(self, token)

### Function: startTagBaseLinkCommand(self, token)

### Function: startTagHeadNoscript(self, token)

### Function: startTagOther(self, token)

### Function: endTagNoscript(self, token)

### Function: endTagBr(self, token)

### Function: endTagOther(self, token)

### Function: anythingElse(self)

### Function: processEOF(self)

### Function: processCharacters(self, token)

### Function: startTagHtml(self, token)

### Function: startTagBody(self, token)

### Function: startTagFrameset(self, token)

### Function: startTagFromHead(self, token)

### Function: startTagHead(self, token)

### Function: startTagOther(self, token)

### Function: endTagHtmlBodyBr(self, token)

### Function: endTagOther(self, token)

### Function: anythingElse(self)

### Function: __init__(self)

### Function: isMatchingFormattingElement(self, node1, node2)

### Function: addFormattingElement(self, token)

### Function: processEOF(self)

### Function: processSpaceCharactersDropNewline(self, token)

### Function: processCharacters(self, token)

### Function: processSpaceCharactersNonPre(self, token)

### Function: startTagProcessInHead(self, token)

### Function: startTagBody(self, token)

### Function: startTagFrameset(self, token)

### Function: startTagCloseP(self, token)

### Function: startTagPreListing(self, token)

### Function: startTagForm(self, token)

### Function: startTagListItem(self, token)

### Function: startTagPlaintext(self, token)

### Function: startTagHeading(self, token)

### Function: startTagA(self, token)

### Function: startTagFormatting(self, token)

### Function: startTagNobr(self, token)

### Function: startTagButton(self, token)

### Function: startTagAppletMarqueeObject(self, token)

### Function: startTagXmp(self, token)

### Function: startTagTable(self, token)

### Function: startTagVoidFormatting(self, token)

### Function: startTagInput(self, token)

### Function: startTagParamSource(self, token)

### Function: startTagHr(self, token)

### Function: startTagImage(self, token)

### Function: startTagIsIndex(self, token)

### Function: startTagTextarea(self, token)

### Function: startTagIFrame(self, token)

### Function: startTagNoscript(self, token)

### Function: startTagRawtext(self, token)

**Description:** iframe, noembed noframes, noscript(if scripting enabled)

### Function: startTagOpt(self, token)

### Function: startTagSelect(self, token)

### Function: startTagRpRt(self, token)

### Function: startTagMath(self, token)

### Function: startTagSvg(self, token)

### Function: startTagMisplaced(self, token)

**Description:** Elements that should be children of other elements that have a
different insertion mode; here they are ignored
"caption", "col", "colgroup", "frame", "frameset", "head",
"option", "optgroup", "tbody", "td", "tfoot", "th", "thead",
"tr", "noscript"

### Function: startTagOther(self, token)

### Function: endTagP(self, token)

### Function: endTagBody(self, token)

### Function: endTagHtml(self, token)

### Function: endTagBlock(self, token)

### Function: endTagForm(self, token)

### Function: endTagListItem(self, token)

### Function: endTagHeading(self, token)

### Function: endTagFormatting(self, token)

**Description:** The much-feared adoption agency algorithm

### Function: endTagAppletMarqueeObject(self, token)

### Function: endTagBr(self, token)

### Function: endTagOther(self, token)

### Function: processCharacters(self, token)

### Function: processEOF(self)

### Function: startTagOther(self, token)

### Function: endTagScript(self, token)

### Function: endTagOther(self, token)

### Function: clearStackToTableContext(self)

### Function: processEOF(self)

### Function: processSpaceCharacters(self, token)

### Function: processCharacters(self, token)

### Function: insertText(self, token)

### Function: startTagCaption(self, token)

### Function: startTagColgroup(self, token)

### Function: startTagCol(self, token)

### Function: startTagRowGroup(self, token)

### Function: startTagImplyTbody(self, token)

### Function: startTagTable(self, token)

### Function: startTagStyleScript(self, token)

### Function: startTagInput(self, token)

### Function: startTagForm(self, token)

### Function: startTagOther(self, token)

### Function: endTagTable(self, token)

### Function: endTagIgnore(self, token)

### Function: endTagOther(self, token)

### Function: __init__(self)

### Function: flushCharacters(self)

### Function: processComment(self, token)

### Function: processEOF(self)

### Function: processCharacters(self, token)

### Function: processSpaceCharacters(self, token)

### Function: processStartTag(self, token)

### Function: processEndTag(self, token)

### Function: ignoreEndTagCaption(self)

### Function: processEOF(self)

### Function: processCharacters(self, token)

### Function: startTagTableElement(self, token)

### Function: startTagOther(self, token)

### Function: endTagCaption(self, token)

### Function: endTagTable(self, token)

### Function: endTagIgnore(self, token)

### Function: endTagOther(self, token)

### Function: ignoreEndTagColgroup(self)

### Function: processEOF(self)

### Function: processCharacters(self, token)

### Function: startTagCol(self, token)

### Function: startTagOther(self, token)

### Function: endTagColgroup(self, token)

### Function: endTagCol(self, token)

### Function: endTagOther(self, token)

### Function: clearStackToTableBodyContext(self)

### Function: processEOF(self)

### Function: processSpaceCharacters(self, token)

### Function: processCharacters(self, token)

### Function: startTagTr(self, token)

### Function: startTagTableCell(self, token)

### Function: startTagTableOther(self, token)

### Function: startTagOther(self, token)

### Function: endTagTableRowGroup(self, token)

### Function: endTagTable(self, token)

### Function: endTagIgnore(self, token)

### Function: endTagOther(self, token)

### Function: clearStackToTableRowContext(self)

### Function: ignoreEndTagTr(self)

### Function: processEOF(self)

### Function: processSpaceCharacters(self, token)

### Function: processCharacters(self, token)

### Function: startTagTableCell(self, token)

### Function: startTagTableOther(self, token)

### Function: startTagOther(self, token)

### Function: endTagTr(self, token)

### Function: endTagTable(self, token)

### Function: endTagTableRowGroup(self, token)

### Function: endTagIgnore(self, token)

### Function: endTagOther(self, token)

### Function: closeCell(self)

### Function: processEOF(self)

### Function: processCharacters(self, token)

### Function: startTagTableOther(self, token)

### Function: startTagOther(self, token)

### Function: endTagTableCell(self, token)

### Function: endTagIgnore(self, token)

### Function: endTagImply(self, token)

### Function: endTagOther(self, token)

### Function: processEOF(self)

### Function: processCharacters(self, token)

### Function: startTagOption(self, token)

### Function: startTagOptgroup(self, token)

### Function: startTagSelect(self, token)

### Function: startTagInput(self, token)

### Function: startTagScript(self, token)

### Function: startTagOther(self, token)

### Function: endTagOption(self, token)

### Function: endTagOptgroup(self, token)

### Function: endTagSelect(self, token)

### Function: endTagOther(self, token)

### Function: processEOF(self)

### Function: processCharacters(self, token)

### Function: startTagTable(self, token)

### Function: startTagOther(self, token)

### Function: endTagTable(self, token)

### Function: endTagOther(self, token)

### Function: adjustSVGTagNames(self, token)

### Function: processCharacters(self, token)

### Function: processStartTag(self, token)

### Function: processEndTag(self, token)

### Function: processEOF(self)

### Function: processComment(self, token)

### Function: processCharacters(self, token)

### Function: startTagHtml(self, token)

### Function: startTagOther(self, token)

### Function: endTagHtml(self, name)

### Function: endTagOther(self, token)

### Function: processEOF(self)

### Function: processCharacters(self, token)

### Function: startTagFrameset(self, token)

### Function: startTagFrame(self, token)

### Function: startTagNoframes(self, token)

### Function: startTagOther(self, token)

### Function: endTagFrameset(self, token)

### Function: endTagOther(self, token)

### Function: processEOF(self)

### Function: processCharacters(self, token)

### Function: startTagNoframes(self, token)

### Function: startTagOther(self, token)

### Function: endTagHtml(self, token)

### Function: endTagOther(self, token)

### Function: processEOF(self)

### Function: processComment(self, token)

### Function: processSpaceCharacters(self, token)

### Function: processCharacters(self, token)

### Function: startTagHtml(self, token)

### Function: startTagOther(self, token)

### Function: processEndTag(self, token)

### Function: processEOF(self)

### Function: processComment(self, token)

### Function: processSpaceCharacters(self, token)

### Function: processCharacters(self, token)

### Function: startTagHtml(self, token)

### Function: startTagNoFrames(self, token)

### Function: startTagOther(self, token)

### Function: processEndTag(self, token)
