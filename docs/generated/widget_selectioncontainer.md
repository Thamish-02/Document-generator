## AI Summary

A file named widget_selectioncontainer.py.


### Function: pad(iterable, padding, length)

**Description:** Returns the sequence elements and then returns None up to the given size (or indefinitely if size is None).

## Class: _SelectionContainer

**Description:** Base class used to display multiple child widgets.

## Class: Accordion

**Description:** Displays children each on a separate accordion page.

## Class: Tab

**Description:** Displays children each on a separate accordion tab.

## Class: Stack

**Description:** Displays only the selected child.

### Function: _validated_index(self, proposal)

### Function: _validate_titles(self, proposal)

### Function: _observe_children(self, change)

### Function: _reset_selected_index(self)

### Function: _reset_titles(self)

### Function: set_title(self, index, title)

**Description:** Sets the title of a container page.
Parameters
----------
index : int
    Index of the container page
title : unicode
    New title

### Function: get_title(self, index)

**Description:** Gets the title of a container page.
Parameters
----------
index : int
    Index of the container page

### Function: __init__(self, children)

### Function: _reset_selected_index(self)
