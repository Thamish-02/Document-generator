## AI Summary

A file named _warnings.py.


## Class: GuessedAtParserWarning

**Description:** The warning issued when BeautifulSoup has to guess what parser to
use -- probably because no parser was specified in the constructor.

## Class: UnusualUsageWarning

**Description:** A superclass for warnings issued when Beautiful Soup sees
something that is typically the result of a mistake in the calling
code, but might be intentional on the part of the user. If it is
in fact intentional, you can filter the individual warning class
to get rid of the warning. If you don't like Beautiful Soup
second-guessing what you are doing, you can filter the
UnusualUsageWarningclass itself and get rid of these entirely.

## Class: MarkupResemblesLocatorWarning

**Description:** The warning issued when BeautifulSoup is given 'markup' that
actually looks like a resource locator -- a URL or a path to a file
on disk.

## Class: AttributeResemblesVariableWarning

**Description:** The warning issued when Beautiful Soup suspects a provided
attribute name may actually be the misspelled name of a Beautiful
Soup variable. Generally speaking, this is only used in cases like
"_class" where it's very unlikely the user would be referencing an
XML attribute with that name.

## Class: XMLParsedAsHTMLWarning

**Description:** The warning issued when an HTML parser is used to parse
XML that is not (as far as we can tell) XHTML.
