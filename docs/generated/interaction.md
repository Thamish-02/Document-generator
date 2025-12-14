## AI Summary

A file named interaction.py.


### Function: show_inline_matplotlib_plots()

**Description:** Show matplotlib plots immediately if using the inline backend.

With ipywidgets 6.0, matplotlib plots don't work well with interact when
using the inline backend that comes with ipykernel. Basically, the inline
backend only shows the plot after the entire cell executes, which does not
play well with drawing plots inside of an interact function. See
https://github.com/jupyter-widgets/ipywidgets/issues/1181/ and
https://github.com/ipython/ipython/issues/10376 for more details. This
function displays any matplotlib plots if the backend is the inline backend.

### Function: interactive_output(f, controls)

**Description:** Connect widget controls to a function.

This function does not generate a user interface for the widgets (unlike `interact`).
This enables customisation of the widget user interface layout.
The user interface layout must be defined and displayed manually.

### Function: _matches(o, pattern)

**Description:** Match a pattern of types in a sequence.

### Function: _get_min_max_value(min, max, value, step)

**Description:** Return min, max, value given input values with possible None.

### Function: _yield_abbreviations_for_parameter(param, kwargs)

**Description:** Get an abbreviation for a function parameter.

## Class: interactive

**Description:** A VBox container containing a group of interactive widgets tied to a
function.

Parameters
----------
__interact_f : function
    The function to which the interactive widgets are tied. The `**kwargs`
    should match the function signature.
__options : dict
    A dict of options. Currently, the only supported keys are
    ``"manual"`` (defaults to ``False``), ``"manual_name"`` (defaults
    to ``"Run Interact"``) and ``"auto_display"`` (defaults to ``False``).
**kwargs : various, optional
    An interactive widget is created for each keyword argument that is a
    valid widget abbreviation.

Note that the first two parameters intentionally start with a double
underscore to avoid being mixed up with keyword arguments passed by
``**kwargs``.

## Class: _InteractFactory

**Description:** Factory for instances of :class:`interactive`.

This class is needed to support options like::

    >>> @interact.options(manual=True)
    ... def greeting(text="World"):
    ...     print("Hello {}".format(text))

Parameters
----------
cls : class
    The subclass of :class:`interactive` to construct.
options : dict
    A dict of options used to construct the interactive
    function. By default, this is returned by
    ``cls.default_options()``.
kwargs : dict
    A dict of **kwargs to use for widgets.

## Class: fixed

**Description:** A pseudo-widget whose value is fixed and never synced to the client.

### Function: observer(change)

### Function: __init__(self, __interact_f, __options)

### Function: update(self)

**Description:** Call the interact function and update the output widget with
the result of the function call.

Parameters
----------
*args : ignored
    Required for this method to be used as traitlets callback.

### Function: signature(self)

### Function: find_abbreviations(self, kwargs)

**Description:** Find the abbreviations for the given function and kwargs.
Return (name, abbrev, default) tuples.

### Function: widgets_from_abbreviations(self, seq)

**Description:** Given a sequence of (name, abbrev, default) tuples, return a sequence of Widgets.

### Function: widget_from_abbrev(cls, abbrev, default)

**Description:** Build a ValueWidget instance given an abbreviation or Widget.

### Function: widget_from_single_value(o)

**Description:** Make widgets from single values, which can be used as parameter defaults.

### Function: widget_from_annotation(t)

**Description:** Make widgets from type annotation and optional default value.

### Function: widget_from_tuple(o)

**Description:** Make widgets from a tuple abbreviation.

### Function: widget_from_iterable(o)

**Description:** Make widgets from an iterable. This should not be done for
a string or tuple.

### Function: factory(cls)

### Function: __init__(self, cls, options, kwargs)

### Function: widget(self, f)

**Description:** Return an interactive function widget for the given function.

The widget is only constructed, not displayed nor attached to
the function.

Returns
-------
An instance of ``self.cls`` (typically :class:`interactive`).

Parameters
----------
f : function
    The function to which the interactive widgets are tied.

### Function: __call__(self, __interact_f)

**Description:** Make the given function interactive by adding and displaying
the corresponding :class:`interactive` widget.

Expects the first argument to be a function. Parameters to this
function are widget abbreviations passed in as keyword arguments
(``**kwargs``). Can be used as a decorator (see examples).

Returns
-------
f : __interact_f with interactive widget attached to it.

Parameters
----------
__interact_f : function
    The function to which the interactive widgets are tied. The `**kwargs`
    should match the function signature. Passed to :func:`interactive()`
**kwargs : various, optional
    An interactive widget is created for each keyword argument that is a
    valid widget abbreviation. Passed to :func:`interactive()`

Examples
--------
Render an interactive text field that shows the greeting with the passed in
text::

    # 1. Using interact as a function
    def greeting(text="World"):
        print("Hello {}".format(text))
    interact(greeting, text="Jupyter Widgets")

    # 2. Using interact as a decorator
    @interact
    def greeting(text="World"):
        print("Hello {}".format(text))

    # 3. Using interact as a decorator with named parameters
    @interact(text="Jupyter Widgets")
    def greeting(text="World"):
        print("Hello {}".format(text))

Render an interactive slider widget and prints square of number::

    # 1. Using interact as a function
    def square(num=1):
        print("{} squared is {}".format(num, num*num))
    interact(square, num=5)

    # 2. Using interact as a decorator
    @interact
    def square(num=2):
        print("{} squared is {}".format(num, num*num))

    # 3. Using interact as a decorator with named parameters
    @interact(num=5)
    def square(num=2):
        print("{} squared is {}".format(num, num*num))

### Function: options(self)

**Description:** Change options for interactive functions.

Returns
-------
A new :class:`_InteractFactory` which will apply the
options when called.

### Function: __init__(self, value)

### Function: get_interact_value(self)

**Description:** Return the value for this widget which should be passed to
interactive functions. Custom widgets can change this method
to process the raw value ``self.value``.
