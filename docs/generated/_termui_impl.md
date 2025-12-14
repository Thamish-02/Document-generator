## AI Summary

A file named _termui_impl.py.


## Class: ProgressBar

### Function: pager(generator, color)

**Description:** Decide what method to use for paging through text.

### Function: _pipepager(generator, cmd_parts, color)

**Description:** Page through text by feeding it to another program. Invoking a
pager through this might support colors.

Returns `True` if the command was found, `False` otherwise and thus another
pager should be attempted.

### Function: _tempfilepager(generator, cmd_parts, color)

**Description:** Page through text by invoking a program on a temporary file.

Returns `True` if the command was found, `False` otherwise and thus another
pager should be attempted.

### Function: _nullpager(stream, generator, color)

**Description:** Simply print unformatted text.  This is the ultimate fallback.

## Class: Editor

### Function: open_url(url, wait, locate)

### Function: _translate_ch_to_exc(ch)

### Function: __init__(self, iterable, length, fill_char, empty_char, bar_template, info_sep, hidden, show_eta, show_percent, show_pos, item_show_func, label, file, color, update_min_steps, width)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_value, tb)

### Function: __iter__(self)

### Function: __next__(self)

### Function: render_finish(self)

### Function: pct(self)

### Function: time_per_iteration(self)

### Function: eta(self)

### Function: format_eta(self)

### Function: format_pos(self)

### Function: format_pct(self)

### Function: format_bar(self)

### Function: format_progress_line(self)

### Function: render_progress(self)

### Function: make_step(self, n_steps)

### Function: update(self, n_steps, current_item)

**Description:** Update the progress bar by advancing a specified number of
steps, and optionally set the ``current_item`` for this new
position.

:param n_steps: Number of steps to advance.
:param current_item: Optional item to set as ``current_item``
    for the updated position.

.. versionchanged:: 8.0
    Added the ``current_item`` optional parameter.

.. versionchanged:: 8.0
    Only render when the number of steps meets the
    ``update_min_steps`` threshold.

### Function: finish(self)

### Function: generator(self)

**Description:** Return a generator which yields the items added to the bar
during construction, and updates the progress bar *after* the
yielded block returns.

### Function: __init__(self, editor, env, require_save, extension)

### Function: get_editor(self)

### Function: edit_files(self, filenames)

### Function: edit(self, text)

### Function: edit(self, text)

### Function: edit(self, text)

### Function: _unquote_file(url)

### Function: raw_terminal()

### Function: getchar(echo)

### Function: raw_terminal()

### Function: getchar(echo)
