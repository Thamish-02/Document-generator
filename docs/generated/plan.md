## AI Summary

A file named plan.py.


### Function: normalizeLinear(value, rangeMin, rangeMax)

**Description:** Linearly normalize value in [rangeMin, rangeMax] to [0, 1], with extrapolation.

### Function: interpolateLinear(t, a, b)

**Description:** Linear interpolation between a and b, with t typically in [0, 1].

### Function: normalizeLog(value, rangeMin, rangeMax)

**Description:** Logarithmically normalize value in [rangeMin, rangeMax] to [0, 1], with extrapolation.

### Function: interpolateLog(t, a, b)

**Description:** Logarithmic interpolation between a and b, with t typically in [0, 1].

### Function: normalizeDegrees(value, rangeMin, rangeMax)

**Description:** Angularly normalize value in [rangeMin, rangeMax] to [0, 1], with extrapolation.

### Function: measureWeight(glyphset, glyphs)

**Description:** Measure the perceptual average weight of the given glyphs.

### Function: measureWidth(glyphset, glyphs)

**Description:** Measure the average width of the given glyphs.

### Function: measureSlant(glyphset, glyphs)

**Description:** Measure the perceptual average slant angle of the given glyphs.

### Function: sanitizeWidth(userTriple, designTriple, pins, measurements)

**Description:** Sanitize the width axis limits.

### Function: sanitizeWeight(userTriple, designTriple, pins, measurements)

**Description:** Sanitize the weight axis limits.

### Function: sanitizeSlant(userTriple, designTriple, pins, measurements)

**Description:** Sanitize the slant axis limits.

### Function: planAxis(measureFunc, normalizeFunc, interpolateFunc, glyphSetFunc, axisTag, axisLimits, values, samples, glyphs, designLimits, pins, sanitizeFunc)

**Description:** Plan an axis.

measureFunc: callable that takes a glyphset and an optional
list of glyphnames, and returns the glyphset-wide measurement
to be used for the axis.

normalizeFunc: callable that takes a measurement and a minimum
and maximum, and normalizes the measurement into the range 0..1,
possibly extrapolating too.

interpolateFunc: callable that takes a normalized t value, and a
minimum and maximum, and returns the interpolated value,
possibly extrapolating too.

glyphSetFunc: callable that takes a variations "location" dictionary,
and returns a glyphset.

axisTag: the axis tag string.

axisLimits: a triple of minimum, default, and maximum values for
the axis. Or an `fvar` Axis object.

values: a list of output values to map for this axis.

samples: the number of samples to use when sampling. Default 8.

glyphs: a list of glyph names to use when sampling. Defaults to None,
which will process all glyphs.

designLimits: an optional triple of minimum, default, and maximum values
represenging the "design" limits for the axis. If not provided, the
axisLimits will be used.

pins: an optional dictionary of before/after mapping entries to pin in
the output.

sanitizeFunc: an optional callable to call to sanitize the axis limits.

### Function: planWeightAxis(glyphSetFunc, axisLimits, weights, samples, glyphs, designLimits, pins, sanitize)

**Description:** Plan a weight (`wght`) axis.

weights: A list of weight values to plan for. If None, the default
values are used.

This function simply calls planAxis with values=weights, and the appropriate
arguments. See documenation for planAxis for more information.

### Function: planWidthAxis(glyphSetFunc, axisLimits, widths, samples, glyphs, designLimits, pins, sanitize)

**Description:** Plan a width (`wdth`) axis.

widths: A list of width values (percentages) to plan for. If None, the default
values are used.

This function simply calls planAxis with values=widths, and the appropriate
arguments. See documenation for planAxis for more information.

### Function: planSlantAxis(glyphSetFunc, axisLimits, slants, samples, glyphs, designLimits, pins, sanitize)

**Description:** Plan a slant (`slnt`) axis.

slants: A list slant angles to plan for. If None, the default
values are used.

This function simply calls planAxis with values=slants, and the appropriate
arguments. See documenation for planAxis for more information.

### Function: planOpticalSizeAxis(glyphSetFunc, axisLimits, sizes, samples, glyphs, designLimits, pins, sanitize)

**Description:** Plan a optical-size (`opsz`) axis.

sizes: A list of optical size values to plan for. If None, the default
values are used.

This function simply calls planAxis with values=sizes, and the appropriate
arguments. See documenation for planAxis for more information.

### Function: makeDesignspaceSnippet(axisTag, axisName, axisLimit, mapping)

**Description:** Make a designspace snippet for a single axis.

### Function: addEmptyAvar(font)

**Description:** Add an empty `avar` table to the font.

### Function: processAxis(font, planFunc, axisTag, axisName, values, samples, glyphs, designLimits, pins, sanitize, plot)

**Description:** Process a single axis.

### Function: main(args)

**Description:** Plan the standard axis mappings for a variable font
