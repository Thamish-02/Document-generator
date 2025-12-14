## AI Summary

A file named jslexer.py.


## Class: Token

### Function: get_rules(jsx, dotted, template_string)

**Description:** Get a tokenization rule list given the passed syntax options.

Internal to this module.

### Function: indicates_division(token)

**Description:** A helper function that helps the tokenizer to decide if the current
token may be followed by a division operator.

### Function: unquote_string(string)

**Description:** Unquote a string with JavaScript rules.  The string has to start with
string delimiters (``'``, ``"`` or the back-tick/grave accent (for template strings).)

### Function: tokenize(source, jsx, dotted, template_string, lineno)

**Description:** Tokenize JavaScript/JSX source.  Returns a generator of tokens.

:param source: The JavaScript source to tokenize.
:param jsx: Enable (limited) JSX parsing.
:param dotted: Read dotted names as single name token.
:param template_string: Support ES6 template strings
:param lineno: starting line number (optional)
