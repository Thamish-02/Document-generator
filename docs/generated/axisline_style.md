## AI Summary

A file named axisline_style.py.


## Class: _FancyAxislineStyle

## Class: AxislineStyle

**Description:** A container class which defines style classes for AxisArtists.

An instance of any axisline style class is a callable object,
whose call signature is ::

   __call__(self, axis_artist, path, transform)

When called, this should return an `.Artist` with the following methods::

  def set_path(self, path):
      # set the path for axisline.

  def set_line_mutation_scale(self, scale):
      # set the scale

  def draw(self, renderer):
      # draw

## Class: SimpleArrow

**Description:** The artist class that will be returned for SimpleArrow style.

## Class: FilledArrow

**Description:** The artist class that will be returned for FilledArrow style.

## Class: _Base

## Class: SimpleArrow

**Description:** A simple arrow.

## Class: FilledArrow

**Description:** An arrow with a filled head.

### Function: __init__(self, axis_artist, line_path, transform, line_mutation_scale)

### Function: set_line_mutation_scale(self, scale)

### Function: _extend_path(self, path, mutation_size)

**Description:** Extend the path to make a room for drawing arrow.

### Function: set_path(self, path)

### Function: draw(self, renderer)

**Description:** Draw the axis line.
 1) Transform the path to the display coordinate.
 2) Extend the path to make a room for arrow.
 3) Update the path of the FancyArrowPatch.
 4) Draw.

### Function: get_window_extent(self, renderer)

### Function: __init__(self, axis_artist, line_path, transform, line_mutation_scale, facecolor)

### Function: __init__(self)

**Description:** initialization.

### Function: __call__(self, axis_artist, transform)

**Description:** Given the AxisArtist instance, and transform for the path (set_path
method), return the Matplotlib artist for drawing the axis line.

### Function: __init__(self, size)

**Description:** Parameters
----------
size : float
    Size of the arrow as a fraction of the ticklabel size.

### Function: new_line(self, axis_artist, transform)

### Function: __init__(self, size, facecolor)

**Description:** Parameters
----------
size : float
    Size of the arrow as a fraction of the ticklabel size.
facecolor : :mpltype:`color`, default: :rc:`axes.edgecolor`
    Fill color.

    .. versionadded:: 3.7

### Function: new_line(self, axis_artist, transform)
