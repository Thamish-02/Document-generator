## AI Summary

A file named scale.py.


## Class: ScaleBase

**Description:** The base class for all scales.

Scales are separable transformations, working on a single dimension.

Subclasses should override

:attr:`name`
    The scale's name.
:meth:`get_transform`
    A method returning a `.Transform`, which converts data coordinates to
    scaled coordinates.  This transform should be invertible, so that e.g.
    mouse positions can be converted back to data coordinates.
:meth:`set_default_locators_and_formatters`
    A method that sets default locators and formatters for an `~.axis.Axis`
    that uses this scale.
:meth:`limit_range_for_scale`
    An optional method that "fixes" the axis range to acceptable values,
    e.g. restricting log-scaled axes to positive values.

## Class: LinearScale

**Description:** The default linear scale.

## Class: FuncTransform

**Description:** A simple transform that takes and arbitrary function for the
forward and inverse transform.

## Class: FuncScale

**Description:** Provide an arbitrary scale with user-supplied function for the axis.

## Class: LogTransform

## Class: InvertedLogTransform

## Class: LogScale

**Description:** A standard logarithmic scale.  Care is taken to only plot positive values.

## Class: FuncScaleLog

**Description:** Provide an arbitrary scale with user-supplied function for the axis and
then put on a logarithmic axes.

## Class: SymmetricalLogTransform

## Class: InvertedSymmetricalLogTransform

## Class: SymmetricalLogScale

**Description:** The symmetrical logarithmic scale is logarithmic in both the
positive and negative directions from the origin.

Since the values close to zero tend toward infinity, there is a
need to have a range around zero that is linear.  The parameter
*linthresh* allows the user to specify the size of this range
(-*linthresh*, *linthresh*).

See :doc:`/gallery/scales/symlog_demo` for a detailed description.

Parameters
----------
base : float, default: 10
    The base of the logarithm.

linthresh : float, default: 2
    Defines the range ``(-x, x)``, within which the plot is linear.
    This avoids having the plot go to infinity around zero.

subs : sequence of int
    Where to place the subticks between each major tick.
    For example, in a log10 scale: ``[2, 3, 4, 5, 6, 7, 8, 9]`` will place
    8 logarithmically spaced minor ticks between each major tick.

linscale : float, optional
    This allows the linear range ``(-linthresh, linthresh)`` to be
    stretched relative to the logarithmic range. Its value is the number of
    decades to use for each half of the linear range. For example, when
    *linscale* == 1.0 (the default), the space used for the positive and
    negative halves of the linear range will be equal to one decade in
    the logarithmic range.

## Class: AsinhTransform

**Description:** Inverse hyperbolic-sine transformation used by `.AsinhScale`

## Class: InvertedAsinhTransform

**Description:** Hyperbolic sine transformation used by `.AsinhScale`

## Class: AsinhScale

**Description:** A quasi-logarithmic scale based on the inverse hyperbolic sine (asinh)

For values close to zero, this is essentially a linear scale,
but for large magnitude values (either positive or negative)
it is asymptotically logarithmic. The transition between these
linear and logarithmic regimes is smooth, and has no discontinuities
in the function gradient in contrast to
the `.SymmetricalLogScale` ("symlog") scale.

Specifically, the transformation of an axis coordinate :math:`a` is
:math:`a \rightarrow a_0 \sinh^{-1} (a / a_0)` where :math:`a_0`
is the effective width of the linear region of the transformation.
In that region, the transformation is
:math:`a \rightarrow a + \mathcal{O}(a^3)`.
For large values of :math:`a` the transformation behaves as
:math:`a \rightarrow a_0 \, \mathrm{sgn}(a) \ln |a| + \mathcal{O}(1)`.

.. note::

   This API is provisional and may be revised in the future
   based on early user feedback.

## Class: LogitTransform

## Class: LogisticTransform

## Class: LogitScale

**Description:** Logit scale for data between zero and one, both excluded.

This scale is similar to a log scale close to zero and to one, and almost
linear around 0.5. It maps the interval ]0, 1[ onto ]-infty, +infty[.

### Function: get_scale_names()

**Description:** Return the names of the available scales.

### Function: scale_factory(scale, axis)

**Description:** Return a scale class by name.

Parameters
----------
scale : {%(names)s}
axis : `~matplotlib.axis.Axis`

### Function: register_scale(scale_class)

**Description:** Register a new kind of scale.

Parameters
----------
scale_class : subclass of `ScaleBase`
    The scale to register.

### Function: _get_scale_docs()

**Description:** Helper function for generating docstrings related to scales.

### Function: __init__(self, axis)

**Description:** Construct a new scale.

Notes
-----
The following note is for scale implementers.

For back-compatibility reasons, scales take an `~matplotlib.axis.Axis`
object as first argument.  However, this argument should not
be used: a single scale object should be usable by multiple
`~matplotlib.axis.Axis`\es at the same time.

### Function: get_transform(self)

**Description:** Return the `.Transform` object associated with this scale.

### Function: set_default_locators_and_formatters(self, axis)

**Description:** Set the locators and formatters of *axis* to instances suitable for
this scale.

### Function: limit_range_for_scale(self, vmin, vmax, minpos)

**Description:** Return the range *vmin*, *vmax*, restricted to the
domain supported by this scale (if any).

*minpos* should be the minimum positive value in the data.
This is used by log scales to determine a minimum value.

### Function: __init__(self, axis)

**Description:**         

### Function: set_default_locators_and_formatters(self, axis)

### Function: get_transform(self)

**Description:** Return the transform for linear scaling, which is just the
`~matplotlib.transforms.IdentityTransform`.

### Function: __init__(self, forward, inverse)

**Description:** Parameters
----------
forward : callable
    The forward function for the transform.  This function must have
    an inverse and, for best behavior, be monotonic.
    It must have the signature::

       def forward(values: array-like) -> array-like

inverse : callable
    The inverse of the forward function.  Signature as ``forward``.

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: __init__(self, axis, functions)

**Description:** Parameters
----------
axis : `~matplotlib.axis.Axis`
    The axis for the scale.
functions : (callable, callable)
    two-tuple of the forward and inverse functions for the scale.
    The forward function must be monotonic.

    Both functions must have the signature::

       def forward(values: array-like) -> array-like

### Function: get_transform(self)

**Description:** Return the `.FuncTransform` associated with this scale.

### Function: set_default_locators_and_formatters(self, axis)

### Function: __init__(self, base, nonpositive)

### Function: __str__(self)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: __init__(self, base)

### Function: __str__(self)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: __init__(self, axis)

**Description:** Parameters
----------
axis : `~matplotlib.axis.Axis`
    The axis for the scale.
base : float, default: 10
    The base of the logarithm.
nonpositive : {'clip', 'mask'}, default: 'clip'
    Determines the behavior for non-positive values. They can either
    be masked as invalid, or clipped to a very small positive number.
subs : sequence of int, default: None
    Where to place the subticks between each major tick.  For example,
    in a log10 scale, ``[2, 3, 4, 5, 6, 7, 8, 9]`` will place 8
    logarithmically spaced minor ticks between each major tick.

### Function: set_default_locators_and_formatters(self, axis)

### Function: get_transform(self)

**Description:** Return the `.LogTransform` associated with this scale.

### Function: limit_range_for_scale(self, vmin, vmax, minpos)

**Description:** Limit the domain to positive values.

### Function: __init__(self, axis, functions, base)

**Description:** Parameters
----------
axis : `~matplotlib.axis.Axis`
    The axis for the scale.
functions : (callable, callable)
    two-tuple of the forward and inverse functions for the scale.
    The forward function must be monotonic.

    Both functions must have the signature::

        def forward(values: array-like) -> array-like

base : float, default: 10
    Logarithmic base of the scale.

### Function: base(self)

### Function: get_transform(self)

**Description:** Return the `.Transform` associated with this scale.

### Function: __init__(self, base, linthresh, linscale)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: __init__(self, base, linthresh, linscale)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: __init__(self, axis)

### Function: set_default_locators_and_formatters(self, axis)

### Function: get_transform(self)

**Description:** Return the `.SymmetricalLogTransform` associated with this scale.

### Function: __init__(self, linear_width)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: __init__(self, linear_width)

### Function: transform_non_affine(self, values)

### Function: inverted(self)

### Function: __init__(self, axis)

**Description:** Parameters
----------
linear_width : float, default: 1
    The scale parameter (elsewhere referred to as :math:`a_0`)
    defining the extent of the quasi-linear region,
    and the coordinate values beyond which the transformation
    becomes asymptotically logarithmic.
base : int, default: 10
    The number base used for rounding tick locations
    on a logarithmic scale. If this is less than one,
    then rounding is to the nearest integer multiple
    of powers of ten.
subs : sequence of int
    Multiples of the number base used for minor ticks.
    If set to 'auto', this will use built-in defaults,
    e.g. (2, 5) for base=10.

### Function: get_transform(self)

### Function: set_default_locators_and_formatters(self, axis)

### Function: __init__(self, nonpositive)

### Function: transform_non_affine(self, values)

**Description:** logit transform (base 10), masked or clipped

### Function: inverted(self)

### Function: __str__(self)

### Function: __init__(self, nonpositive)

### Function: transform_non_affine(self, values)

**Description:** logistic transform (base 10)

### Function: inverted(self)

### Function: __str__(self)

### Function: __init__(self, axis, nonpositive)

**Description:** Parameters
----------
axis : `~matplotlib.axis.Axis`
    Currently unused.
nonpositive : {'mask', 'clip'}
    Determines the behavior for values beyond the open interval ]0, 1[.
    They can either be masked as invalid, or clipped to a number very
    close to 0 or 1.
use_overline : bool, default: False
    Indicate the usage of survival notation (\overline{x}) in place of
    standard notation (1-x) for probability close to one.
one_half : str, default: r"\frac{1}{2}"
    The string used for ticks formatter to represent 1/2.

### Function: get_transform(self)

**Description:** Return the `.LogitTransform` associated with this scale.

### Function: set_default_locators_and_formatters(self, axis)

### Function: limit_range_for_scale(self, vmin, vmax, minpos)

**Description:** Limit the domain to values between 0 and 1 (excluded).
