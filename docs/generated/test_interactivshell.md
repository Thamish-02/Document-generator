## AI Summary

A file named test_interactivshell.py.


## Class: TestAutoSuggest

## Class: TestElide

## Class: TestContextAwareCompletion

## Class: mock_input_helper

**Description:** Machinery for tests of the main interact loop.

Used by the mock_input decorator.

### Function: mock_input(testfunc)

**Description:** Decorator for tests of the main interact loop.

Write the test as a generator, yield-ing the input strings, which IPython
will see as if they were typed in at the prompt.

## Class: InteractiveShellTestCase

### Function: syntax_error_transformer(lines)

**Description:** Transformer that throws SyntaxError if 'syntaxerror' is in the code.

## Class: TerminalMagicsTestCase

### Function: test_changing_provider(self)

### Function: test_elide(self)

### Function: test_elide_typed_normal(self)

### Function: test_elide_typed_short_match(self)

**Description:** if the match is too short we don't elide.
avoid the "the...the"

### Function: test_elide_typed_no_match(self)

**Description:** if the match is too short we don't elide.
avoid the "the...the"

### Function: test_adjust_completion_text_based_on_context(self)

### Function: __init__(self, testgen)

### Function: __enter__(self)

### Function: __exit__(self, etype, value, tb)

### Function: fake_input(self)

### Function: test_method(self)

### Function: rl_hist_entries(self, rl, n)

**Description:** Get last n readline history entries as a list

### Function: test_inputtransformer_syntaxerror(self)

### Function: test_repl_not_plain_text(self)

### Function: test_paste_magics_blankline(self)

**Description:** Test that code with a blank line doesn't get split (gh-3246).

## Class: Test

## Class: Test2

### Function: handler(data, metadata)

### Function: __repr__(self)

### Function: _repr_html_(self)

### Function: _ipython_display_(self)
