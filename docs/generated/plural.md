## AI Summary

A file named plural.py.


### Function: extract_operands(source)

**Description:** Extract operands from a decimal, a float or an int, according to `CLDR rules`_.

The result is an 8-tuple (n, i, v, w, f, t, c, e), where those symbols are as follows:

====== ===============================================================
Symbol Value
------ ---------------------------------------------------------------
n      absolute value of the source number (integer and decimals).
i      integer digits of n.
v      number of visible fraction digits in n, with trailing zeros.
w      number of visible fraction digits in n, without trailing zeros.
f      visible fractional digits in n, with trailing zeros.
t      visible fractional digits in n, without trailing zeros.
c      compact decimal exponent value: exponent of the power of 10 used in compact decimal formatting.
e      currently, synonym for ‘c’. however, may be redefined in the future.
====== ===============================================================

.. _`CLDR rules`: https://www.unicode.org/reports/tr35/tr35-61/tr35-numbers.html#Operands

:param source: A real number
:type source: int|float|decimal.Decimal
:return: A n-i-v-w-f-t-c-e tuple
:rtype: tuple[decimal.Decimal, int, int, int, int, int, int, int]

## Class: PluralRule

**Description:** Represents a set of language pluralization rules.  The constructor
accepts a list of (tag, expr) tuples or a dict of `CLDR rules`_. The
resulting object is callable and accepts one parameter with a positive or
negative number (both integer and float) for the number that indicates the
plural form for a string and returns the tag for the format:

>>> rule = PluralRule({'one': 'n is 1'})
>>> rule(1)
'one'
>>> rule(2)
'other'

Currently the CLDR defines these tags: zero, one, two, few, many and
other where other is an implicit default.  Rules should be mutually
exclusive; for a given numeric value, only one rule should apply (i.e.
the condition should only be true for one of the plural rule elements.

.. _`CLDR rules`: https://www.unicode.org/reports/tr35/tr35-33/tr35-numbers.html#Language_Plural_Rules

### Function: to_javascript(rule)

**Description:** Convert a list/dict of rules or a `PluralRule` object into a JavaScript
function.  This function depends on no external library:

>>> to_javascript({'one': 'n is 1'})
"(function(n) { return (n == 1) ? 'one' : 'other'; })"

Implementation detail: The function generated will probably evaluate
expressions involved into range operations multiple times.  This has the
advantage that external helper functions are not required and is not a
big performance hit for these simple calculations.

:param rule: the rules as list or dict, or a `PluralRule` object
:raise RuleError: if the expression is malformed

### Function: to_python(rule)

**Description:** Convert a list/dict of rules or a `PluralRule` object into a regular
Python function.  This is useful in situations where you need a real
function and don't are about the actual rule object:

>>> func = to_python({'one': 'n is 1', 'few': 'n in 2..4'})
>>> func(1)
'one'
>>> func(3)
'few'
>>> func = to_python({'one': 'n in 1,11', 'few': 'n in 3..10,13..19'})
>>> func(11)
'one'
>>> func(15)
'few'

:param rule: the rules as list or dict, or a `PluralRule` object
:raise RuleError: if the expression is malformed

### Function: to_gettext(rule)

**Description:** The plural rule as gettext expression.  The gettext expression is
technically limited to integers and returns indices rather than tags.

>>> to_gettext({'one': 'n is 1', 'two': 'n is 2'})
'nplurals=3; plural=((n == 1) ? 0 : (n == 2) ? 1 : 2);'

:param rule: the rules as list or dict, or a `PluralRule` object
:raise RuleError: if the expression is malformed

### Function: in_range_list(num, range_list)

**Description:** Integer range list test.  This is the callback for the "in" operator
of the UTS #35 pluralization rule language:

>>> in_range_list(1, [(1, 3)])
True
>>> in_range_list(3, [(1, 3)])
True
>>> in_range_list(3, [(1, 3), (5, 8)])
True
>>> in_range_list(1.2, [(1, 4)])
False
>>> in_range_list(10, [(1, 4)])
False
>>> in_range_list(10, [(1, 4), (6, 8)])
False

### Function: within_range_list(num, range_list)

**Description:** Float range test.  This is the callback for the "within" operator
of the UTS #35 pluralization rule language:

>>> within_range_list(1, [(1, 3)])
True
>>> within_range_list(1.0, [(1, 3)])
True
>>> within_range_list(1.2, [(1, 4)])
True
>>> within_range_list(8.8, [(1, 4), (7, 15)])
True
>>> within_range_list(10, [(1, 4)])
False
>>> within_range_list(10.5, [(1, 4), (20, 30)])
False

### Function: cldr_modulo(a, b)

**Description:** Javaish modulo.  This modulo operator returns the value with the sign
of the dividend rather than the divisor like Python does:

>>> cldr_modulo(-3, 5)
-3
>>> cldr_modulo(-3, -5)
-3
>>> cldr_modulo(3, 5)
3

## Class: RuleError

**Description:** Raised if a rule is malformed.

### Function: tokenize_rule(s)

### Function: test_next_token(tokens, type_, value)

### Function: skip_token(tokens, type_, value)

### Function: value_node(value)

### Function: ident_node(name)

### Function: range_list_node(range_list)

### Function: negate(rv)

## Class: _Parser

**Description:** Internal parser.  This class can translate a single rule into an abstract
tree of tuples. It implements the following grammar::

    condition     = and_condition ('or' and_condition)*
                    ('@integer' samples)?
                    ('@decimal' samples)?
    and_condition = relation ('and' relation)*
    relation      = is_relation | in_relation | within_relation
    is_relation   = expr 'is' ('not')? value
    in_relation   = expr (('not')? 'in' | '=' | '!=') range_list
    within_relation = expr ('not')? 'within' range_list
    expr          = operand (('mod' | '%') value)?
    operand       = 'n' | 'i' | 'f' | 't' | 'v' | 'w'
    range_list    = (range | value) (',' range_list)*
    value         = digit+
    digit         = 0|1|2|3|4|5|6|7|8|9
    range         = value'..'value
    samples       = sampleRange (',' sampleRange)* (',' ('…'|'...'))?
    sampleRange   = decimalValue '~' decimalValue
    decimalValue  = value ('.' value)?

- Whitespace can occur between or around any of the above tokens.
- Rules should be mutually exclusive; for a given numeric value, only one
  rule should apply (i.e. the condition should only be true for one of
  the plural rule elements).
- The in and within relations can take comma-separated lists, such as:
  'n in 3,5,7..15'.
- Samples are ignored.

The translator parses the expression on instantiation into an attribute
called `ast`.

### Function: _binary_compiler(tmpl)

**Description:** Compiler factory for the `_Compiler`.

### Function: _unary_compiler(tmpl)

**Description:** Compiler factory for the `_Compiler`.

## Class: _Compiler

**Description:** The compilers are able to transform the expressions into multiple
output formats.

## Class: _PythonCompiler

**Description:** Compiles an expression to Python.

## Class: _GettextCompiler

**Description:** Compile into a gettext plural expression.

## Class: _JavaScriptCompiler

**Description:** Compiles the expression to plain of JavaScript.

## Class: _UnicodeCompiler

**Description:** Returns a unicode pluralization rule again.

### Function: __init__(self, rules)

**Description:** Initialize the rule instance.

:param rules: a list of ``(tag, expr)``) tuples with the rules
              conforming to UTS #35 or a dict with the tags as keys
              and expressions as values.
:raise RuleError: if the expression is malformed

### Function: __repr__(self)

### Function: parse(cls, rules)

**Description:** Create a `PluralRule` instance for the given rules.  If the rules
are a `PluralRule` object, that object is returned.

:param rules: the rules as list or dict, or a `PluralRule` object
:raise RuleError: if the expression is malformed

### Function: rules(self)

**Description:** The `PluralRule` as a dict of unicode plural rules.

>>> rule = PluralRule({'one': 'n is 1'})
>>> rule.rules
{'one': 'n is 1'}

### Function: tags(self)

**Description:** A set of explicitly defined tags in this rule.  The implicit default
``'other'`` rules is not part of this set unless there is an explicit
rule for it.

### Function: __getstate__(self)

### Function: __setstate__(self, abstract)

### Function: __call__(self, n)

### Function: __init__(self, string)

### Function: expect(self, type_, value, term)

### Function: condition(self)

### Function: and_condition(self)

### Function: relation(self)

### Function: newfangled_relation(self, left)

### Function: range_or_value(self)

### Function: range_list(self)

### Function: expr(self)

### Function: value(self)

### Function: compile(self, arg)

### Function: compile_relation(self, method, expr, range_list)

### Function: compile_relation(self, method, expr, range_list)

### Function: compile_relation(self, method, expr, range_list)

### Function: compile_relation(self, method, expr, range_list)

### Function: compile_not(self, relation)

### Function: compile_relation(self, method, expr, range_list, negated)
