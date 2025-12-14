## AI Summary

A file named interactiveshell.py.


## Class: _NoStyle

### Function: _backward_compat_continuation_prompt_tokens(method, width)

**Description:** Sagemath use custom prompt and we broke them in 8.19.

### Function: get_default_editor()

### Function: black_reformat_handler(text_before_cursor)

**Description:** We do not need to protect against error,
this is taken care at a higher level where any reformat error is ignored.
Indeed we may call reformatting on incomplete code.

### Function: yapf_reformat_handler(text_before_cursor)

## Class: PtkHistoryAdapter

**Description:** Prompt toolkit has it's own way of handling history, Where it assumes it can
Push/pull from history.

## Class: TerminalInteractiveShell

### Function: __init__(self, shell)

### Function: append_string(self, string)

### Function: _refresh(self)

### Function: load_history_strings(self)

### Function: store_string(self, string)

### Function: debugger_cls(self)

### Function: _validate_editing_mode(self, proposal)

### Function: _editing_mode(self, change)

### Function: _set_formatter(self, formatter)

### Function: _autoformatter_changed(self, change)

### Function: _highlighting_style_changed(self, change)

### Function: refresh_style(self)

### Function: _prompts_default(self)

### Function: _displayhook_class_default(self)

### Function: _llm_prefix_from_history_changed(self, change)

### Function: _llm_provider_class_changed(self, change)

### Function: _set_autosuggestions(self, provider)

### Function: _autosuggestions_provider_changed(self, change)

### Function: _shortcuts_changed(self, change)

### Function: _merge_shortcuts(self, user_shortcuts)

### Function: init_term_title(self, change)

### Function: restore_term_title(self)

### Function: init_display_formatter(self)

### Function: init_prompt_toolkit_cli(self)

### Function: _make_style_from_name_or_cls(self, name_or_cls)

**Description:** Small wrapper that make an IPython compatible style from a style name

We need that to add style for prompt ... etc.

### Function: pt_complete_style(self)

### Function: color_depth(self)

### Function: _extra_prompt_options(self)

**Description:** Return the current layout option for the current Terminal InteractiveShell

### Function: prompt_for_code(self)

### Function: enable_win_unicode_console(self)

### Function: init_io(self)

### Function: init_magics(self)

### Function: init_alias(self)

### Function: __init__(self)

### Function: ask_exit(self)

### Function: interact(self)

### Function: mainloop(self)

### Function: inputhook(self, context)

### Function: enable_gui(self, gui)

### Function: auto_rewrite_input(self, cmd)

**Description:** Overridden from the parent class to use fancy rewriting prompt

### Function: switch_doctest_mode(self, mode)

**Description:** Switch prompts to classic for %doctest_mode

### Function: get_message()

### Function: prompt()

### Function: init_llm_provider()

### Function: no_prefix(history_manager)

### Function: input_history(history_manager)
