## AI Summary

A file named ipdoctest.py.


## Class: DocTestFinder

## Class: IPDoctestOutputChecker

**Description:** Second-chance checker with support for random tests.

If the default comparison doesn't pass, this checker looks in the expected
output string for flags that tell us to ignore the output.

## Class: IPExample

## Class: IPDocTestParser

**Description:** A class used to parse strings containing doctest examples.

Note: This is a version modified to properly recognize IPython input and
convert any IPython examples into valid Python ones.

## Class: IPDocTestRunner

**Description:** Test runner that synchronizes the IPython namespace with test globals.
    

### Function: _get_test(self, obj, name, module, globs, source_lines)

### Function: check_output(self, want, got, optionflags)

**Description:** Check output, accepting special markers embedded in the output.

If the output didn't pass the default validation but the special string
'#random' is included, we accept it.

### Function: ip2py(self, source)

**Description:** Convert input IPython source into valid Python.

### Function: parse(self, string, name)

**Description:** Divide the given string into examples and intervening text,
and return them as a list of alternating Examples and strings.
Line numbers for the Examples are 0-based.  The optional
argument `name` is a name identifying this string, and is only
used for error messages.

### Function: _parse_example(self, m, name, lineno, ip2py)

**Description:** Given a regular expression match from `_EXAMPLE_RE` (`m`),
return a pair `(source, want)`, where `source` is the matched
example's source code (with prompts and indentation stripped);
and `want` is the example's expected output (with indentation
stripped).

`name` is the string's name, and `lineno` is the line number
where the example starts; both are used for error messages.

Optional:
`ip2py`: if true, filter the input via IPython to convert the syntax
into valid python.

### Function: _check_prompt_blank(self, lines, indent, name, lineno, ps1_len)

**Description:** Given the lines of a source string (including prompts and
leading indentation), check to make sure that every prompt is
followed by a space character.  If any line is not followed by
a space character, then raise ValueError.

Note: IPython-modified version which takes the input prompt length as a
parameter, so that prompts of variable length can be dealt with.

### Function: run(self, test, compileflags, out, clear_globs)
