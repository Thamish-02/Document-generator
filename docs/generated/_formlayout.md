## AI Summary

A file named _formlayout.py.


## Class: ColorButton

**Description:** Color choosing push button

### Function: to_qcolor(color)

**Description:** Create a QColor from a matplotlib color

## Class: ColorLayout

**Description:** Color-specialized QLineEdit layout

### Function: font_is_installed(font)

**Description:** Check if font is installed

### Function: tuple_to_qfont(tup)

**Description:** Create a QFont from tuple:
    (family [string], size [int], italic [bool], bold [bool])

### Function: qfont_to_tuple(font)

## Class: FontLayout

**Description:** Font selection

### Function: is_edit_valid(edit)

## Class: FormWidget

## Class: FormComboWidget

## Class: FormTabWidget

## Class: FormDialog

**Description:** Form Dialog

### Function: fedit(data, title, comment, icon, parent, apply)

**Description:** Create form dialog

data: datalist, datagroup
title: str
comment: str
icon: QIcon instance
parent: parent QWidget
apply: apply callback (function)

datalist: list/tuple of (field_name, field_value)
datagroup: list/tuple of (datalist *or* datagroup, title, comment)

-> one field for each member of a datalist
-> one tab for each member of a top-level datagroup
-> one page (of a multipage widget, each page can be selected with a combo
   box) for each member of a datagroup inside a datagroup

Supported types for field_value:
  - int, float, str, bool
  - colors: in Qt-compatible text form, i.e. in hex format or name
            (red, ...) (automatically detected from a string)
  - list/tuple:
      * the first element will be the selected index (or value)
      * the other elements can be couples (key, value) or only values

### Function: __init__(self, parent)

### Function: choose_color(self)

### Function: get_color(self)

### Function: set_color(self, color)

### Function: __init__(self, color, parent)

### Function: update_color(self)

### Function: update_text(self, color)

### Function: text(self)

### Function: __init__(self, value, parent)

### Function: get_font(self)

### Function: __init__(self, data, comment, with_margin, parent)

**Description:** Parameters
----------
data : list of (label, value) pairs
    The data to be edited in the form.
comment : str, optional
with_margin : bool, default: False
    If False, the form elements reach to the border of the widget.
    This is the desired behavior if the FormWidget is used as a widget
    alongside with other widgets such as a QComboBox, which also do
    not have a margin around them.
    However, a margin can be desired if the FormWidget is the only
    widget within a container, e.g. a tab in a QTabWidget.
parent : QWidget or None
    The parent widget.

### Function: get_dialog(self)

**Description:** Return FormDialog instance

### Function: setup(self)

### Function: get(self)

### Function: __init__(self, datalist, comment, parent)

### Function: setup(self)

### Function: get(self)

### Function: __init__(self, datalist, comment, parent)

### Function: setup(self)

### Function: get(self)

### Function: __init__(self, data, title, comment, icon, parent, apply)

### Function: register_float_field(self, field)

### Function: update_buttons(self)

### Function: accept(self)

### Function: reject(self)

### Function: apply(self)

### Function: get(self)

**Description:** Return form result

### Function: create_datalist_example()

### Function: create_datagroup_example()

### Function: apply_test(data)
