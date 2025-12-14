## AI Summary

A file named auto_suggest.py.


### Function: _get_query(document)

## Class: AppendAutoSuggestionInAnyLine

**Description:** Append the auto suggestion to lines other than the last (appending to the
last line is natively supported by the prompt toolkit).

This has a private `_debug` attribute that can be set to True to display
debug information as virtual suggestion on the end of any line. You can do
so with:

    >>> from IPython.terminal.shortcuts.auto_suggest import AppendAutoSuggestionInAnyLine
    >>> AppendAutoSuggestionInAnyLine._debug = True

## Class: NavigableAutoSuggestFromHistory

**Description:** A subclass of AutoSuggestFromHistory that allow navigation to next/previous
suggestion from history. To do so it remembers the current position, but it
state need to carefully be cleared on the right events.

### Function: accept_or_jump_to_end(event)

**Description:** Apply autosuggestion or jump to end of line.

### Function: _deprected_accept_in_vi_insert_mode(event)

**Description:** Accept autosuggestion or jump to end of line.

.. deprecated:: 8.12
    Use `accept_or_jump_to_end` instead.

### Function: accept(event)

**Description:** Accept autosuggestion

### Function: discard(event)

**Description:** Discard autosuggestion

### Function: accept_word(event)

**Description:** Fill partial autosuggestion by word

### Function: accept_character(event)

**Description:** Fill partial autosuggestion by character

### Function: accept_and_keep_cursor(event)

**Description:** Accept autosuggestion and keep cursor in place

### Function: accept_and_move_cursor_left(event)

**Description:** Accept autosuggestion and move cursor left in place

### Function: _update_hint(buffer)

### Function: backspace_and_resume_hint(event)

**Description:** Resume autosuggestions after deleting last character

### Function: resume_hinting(event)

**Description:** Resume autosuggestions

### Function: up_and_update_hint(event)

**Description:** Go up and update hint

### Function: down_and_update_hint(event)

**Description:** Go down and update hint

### Function: accept_token(event)

**Description:** Fill partial autosuggestion by token

### Function: _swap_autosuggestion(buffer, provider, direction_method)

**Description:** We skip most recent history entry (in either direction) if it equals the
current autosuggestion because if user cycles when auto-suggestion is shown
they most likely want something else than what was suggested (otherwise
they would have accepted the suggestion).

### Function: swap_autosuggestion_up(event)

**Description:** Get next autosuggestion from history.

### Function: swap_autosuggestion_down(event)

**Description:** Get previous autosuggestion from history.

### Function: __getattr__(key)

### Function: __init__(self, style)

### Function: apply_transformation(self, ti)

**Description:**  Apply transformation to the line that is currently being edited.

 This is a variation of the original implementation in prompt toolkit
 that allows to not only append suggestions to any line, but also to show
 multi-line suggestions.

 As transformation are applied on a line-by-line basis; we need to trick
 a bit, and elide any line that is after the line we are currently
 editing, until we run out of completions. We cannot shift the existing
 lines

 There are multiple cases to handle:

 The completions ends before the end of the buffer:
     We can resume showing the normal line, and say that some code may
     be hidden.

The completions ends at the end of the buffer
     We can just say that some code may be hidden.

 And separately:

 The completions ends beyond the end of the buffer
     We need to both say that some code may be hidden, and that some
     lines are not shown.

### Function: __init__(self)

### Function: reset_history_position(self, _)

### Function: disconnect(self)

### Function: connect(self, pt_app)

### Function: get_suggestion(self, buffer, document)

### Function: _dismiss(self, buffer)

### Function: _find_match(self, text, skip_lines, history, previous)

**Description:** text : str
    Text content to find a match for, the user cursor is most of the
    time at the end of this text.
skip_lines : float
    number of items to skip in the search, this is used to indicate how
    far in the list the user has navigated by pressing up or down.
    The float type is used as the base value is +inf
history : History
    prompt_toolkit History instance to fetch previous entries from.
previous : bool
    Direction of the search, whether we are looking previous match
    (True), or next match (False).

Yields
------
Tuple with:
str:
    current suggestion.
float:
    will actually yield only ints, which is passed back via skip_lines,
    which may be a +inf (float)

### Function: _find_next_match(self, text, skip_lines, history)

### Function: _find_previous_match(self, text, skip_lines, history)

### Function: up(self, query, other_than, history)

### Function: down(self, query, other_than, history)

### Function: _cancel_running_llm_task(self)

**Description:** Try to cancel the currently running llm_task if exists, and set it to None.

### Function: _llm_provider(self)

**Description:** Lazy-initialized instance of the LLM provider.

Do not use in the constructor, as `_init_llm_provider` can trigger slow side-effects.
