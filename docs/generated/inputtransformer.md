## AI Summary

A file named inputtransformer.py.


## Class: InputTransformer

**Description:** Abstract base class for line-based input transformers.

## Class: StatelessInputTransformer

**Description:** Wrapper for a stateless input transformer implemented as a function.

## Class: CoroutineInputTransformer

**Description:** Wrapper for an input transformer implemented as a coroutine.

## Class: TokenInputTransformer

**Description:** Wrapper for a token-based input transformer.

func should accept a list of tokens (5-tuples, see tokenize docs), and
return an iterable which can be passed to tokenize.untokenize().

## Class: assemble_python_lines

### Function: assemble_logical_lines()

**Description:** Join lines following explicit line continuations (\)

### Function: _make_help_call(target, esc, lspace)

**Description:** Prepares a pinfo(2)/psearch call from a target name and the escape
(i.e. ? or ??)

### Function: _tr_system(line_info)

**Description:** Translate lines escaped with: !

### Function: _tr_system2(line_info)

**Description:** Translate lines escaped with: !!

### Function: _tr_help(line_info)

**Description:** Translate lines escaped with: ?/??

### Function: _tr_magic(line_info)

**Description:** Translate lines escaped with: %

### Function: _tr_quote(line_info)

**Description:** Translate lines escaped with: ,

### Function: _tr_quote2(line_info)

**Description:** Translate lines escaped with: ;

### Function: _tr_paren(line_info)

**Description:** Translate lines escaped with: /

### Function: escaped_commands(line)

**Description:** Transform escaped commands - %magic, !system, ?help + various autocalls.

### Function: _line_tokens(line)

**Description:** Helper for has_comment and ends_in_comment_or_string.

### Function: has_comment(src)

**Description:** Indicate whether an input line has (i.e. ends in, or is) a comment.

This uses tokenize, so it can distinguish comments from # inside strings.

Parameters
----------
src : string
    A single line input string.

Returns
-------
comment : bool
    True if source has a comment.

### Function: ends_in_comment_or_string(src)

**Description:** Indicates whether or not an input line ends in a comment or within
a multiline string.

Parameters
----------
src : string
    A single line input string.

Returns
-------
comment : bool
    True if source ends in a comment or multiline string.

### Function: help_end(line)

**Description:** Translate lines with ?/?? at the end

### Function: cellmagic(end_on_blank_line)

**Description:** Captures & transforms cell magics.

After a cell magic is started, this stores up any lines it gets until it is
reset (sent None).

### Function: _strip_prompts(prompt_re, initial_re, turnoff_re)

**Description:** Remove matching input prompts from a block of input.

Parameters
----------
prompt_re : regular expression
    A regular expression matching any input prompt (including continuation)
initial_re : regular expression, optional
    A regular expression matching only the initial prompt, but not continuation.
    If no initial expression is given, prompt_re will be used everywhere.
    Used mainly for plain Python prompts, where the continuation prompt
    ``...`` is a valid Python expression in Python 3, so shouldn't be stripped.

Notes
-----
If `initial_re` and `prompt_re differ`,
only `initial_re` will be tested against the first line.
If any prompt is found on the first two lines,
prompts will be stripped from the rest of the block.

### Function: classic_prompt()

**Description:** Strip the >>>/... prompts of the Python interactive shell.

### Function: ipy_prompt()

**Description:** Strip IPython's In [1]:/...: prompts.

### Function: leading_indent()

**Description:** Remove leading indentation.

If the first line starts with a spaces or tabs, the same whitespace will be
removed from each following line until it is reset.

### Function: assign_from_system(line)

**Description:** Transform assignment from system commands (e.g. files = !ls)

### Function: assign_from_magic(line)

**Description:** Transform assignment from magic commands (e.g. a = %who_ls)

### Function: __init__(self)

### Function: push(self, line)

**Description:** Send a line of input to the transformer, returning the transformed
input or None if the transformer is waiting for more input.

Must be overridden by subclasses.

Implementations may raise ``SyntaxError`` if the input is invalid. No
other exceptions may be raised.

### Function: reset(self)

**Description:** Return, transformed any lines that the transformer has accumulated,
and reset its internal state.

Must be overridden by subclasses.

### Function: wrap(cls, func)

**Description:** Can be used by subclasses as a decorator, to return a factory that
will allow instantiation with the decorated object.

### Function: __init__(self, func)

### Function: __repr__(self)

### Function: push(self, line)

**Description:** Send a line of input to the transformer, returning the
transformed input.

### Function: reset(self)

**Description:** No-op - exists for compatibility.

### Function: __init__(self, coro)

### Function: __repr__(self)

### Function: push(self, line)

**Description:** Send a line of input to the transformer, returning the
transformed input or None if the transformer is waiting for more
input.

### Function: reset(self)

**Description:** Return, transformed any lines that the transformer has
accumulated, and reset its internal state.

### Function: __init__(self, func)

### Function: reset_tokenizer(self)

### Function: push(self, line)

### Function: output(self, tokens)

### Function: reset(self)

### Function: __init__(self)

### Function: output(self, tokens)

### Function: transformer_factory()
