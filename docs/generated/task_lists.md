## AI Summary

A file named task_lists.py.


### Function: task_lists_hook(md, state)

### Function: render_task_list_item(renderer, text, checked)

### Function: task_lists(md)

**Description:** A mistune plugin to support task lists. Spec defined by
GitHub flavored Markdown and commonly used by many parsers:

.. code-block:: text

    - [ ] unchecked task
    - [x] checked task

:param md: Markdown instance

### Function: _rewrite_all_list_items(tokens)

### Function: _rewrite_list_item(tok)
