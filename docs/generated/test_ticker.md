## AI Summary

A file named test_ticker.py.


## Class: TestMaxNLocator

## Class: TestLinearLocator

## Class: TestMultipleLocator

## Class: TestAutoMinorLocator

## Class: TestLogLocator

## Class: TestNullLocator

## Class: _LogitHelper

## Class: TestLogitLocator

## Class: TestFixedLocator

## Class: TestIndexLocator

## Class: TestSymmetricalLogLocator

## Class: TestAsinhLocator

## Class: TestScalarFormatter

## Class: TestLogFormatterExponent

## Class: TestLogFormatterMathtext

## Class: TestLogFormatterSciNotation

## Class: TestLogFormatter

## Class: TestLogitFormatter

## Class: TestFormatStrFormatter

## Class: TestStrMethodFormatter

## Class: TestEngFormatter

### Function: test_engformatter_usetex_useMathText()

### Function: test_engformatter_offset_oom(data_offset, noise, oom_center_desired, oom_noise_desired)

## Class: TestPercentFormatter

### Function: _impl_locale_comma()

### Function: test_locale_comma()

### Function: test_majformatter_type()

### Function: test_minformatter_type()

### Function: test_majlocator_type()

### Function: test_minlocator_type()

### Function: test_minorticks_rc()

### Function: test_minorticks_toggle()

**Description:** Test toggling minor ticks

Test `.Axis.minorticks_on()` and `.Axis.minorticks_off()`. Testing is
limited to a subset of built-in scales - `'linear'`, `'log'`, `'asinh'`
and `'logit'`. `symlog` scale does not seem to have a working minor
locator and is omitted. In future, this test should cover all scales in
`matplotlib.scale.get_scale_names()`.

### Function: test_remove_overlap(remove_overlapping_locs, expected_num)

### Function: test_bad_locator_subs(sub)

### Function: test_small_range_loglocator(numticks)

### Function: test_NullFormatter()

### Function: test_set_offset_string(formatter)

### Function: test_minorticks_on_multi_fig()

**Description:** Turning on minor gridlines in a multi-Axes Figure
that contains more than one boxplot and shares the x-axis
should not raise an exception.

### Function: test_basic(self, vmin, vmax, expected)

### Function: test_integer(self, vmin, vmax, steps, expected)

### Function: test_errors(self, kwargs, errortype, match)

### Function: test_padding(self, steps, result)

### Function: test_basic(self)

### Function: test_zero_numticks(self)

### Function: test_set_params(self)

**Description:** Create linear locator with presets={}, numticks=2 and change it to
something else. See if change was successful. Should not exception.

### Function: test_presets(self)

### Function: test_basic(self)

### Function: test_basic_with_offset(self)

### Function: test_view_limits(self)

**Description:** Test basic behavior of view limits.

### Function: test_view_limits_round_numbers(self)

**Description:** Test that everything works properly with 'round_numbers' for auto
limit.

### Function: test_view_limits_round_numbers_with_offset(self)

**Description:** Test that everything works properly with 'round_numbers' for auto
limit.

### Function: test_view_limits_single_bin(self)

**Description:** Test that 'round_numbers' works properly with a single bin.

### Function: test_set_params(self)

**Description:** Create multiple locator with 0.7 base, and change it to something else.
See if change was successful.

### Function: test_basic(self)

### Function: test_first_and_last_minorticks(self)

**Description:** Test that first and last minor tick appear as expected.

### Function: test_low_number_of_majorticks(self, nb_majorticks, expected_nb_minorticks)

### Function: test_using_all_default_major_steps(self)

### Function: test_number_of_minor_ticks(self, major_step, expected_nb_minordivisions)

### Function: test_additional(self, lim, ref)

### Function: test_number_of_minor_ticks_auto(self, lim, ref, use_rcparam)

### Function: test_number_of_minor_ticks_int(self, n, lim, ref, use_rcparam)

### Function: test_basic(self)

### Function: test_polar_axes(self)

**Description:** Polar Axes have a different ticking logic.

### Function: test_switch_to_autolocator(self)

### Function: test_set_params(self)

**Description:** Create log locator with default value, base=10.0, subs=[1.0],
numticks=15 and change it to something else.
See if change was successful. Should not raise exception.

### Function: test_tick_values_correct(self)

### Function: test_tick_values_not_empty(self)

### Function: test_multiple_shared_axes(self)

### Function: test_set_params(self)

**Description:** Create null locator, and attempt to call set_params() on it.
Should not exception, and should raise a warning.

### Function: isclose(x, y)

### Function: assert_almost_equal(x, y)

### Function: test_basic_major(self, lims, expected_low_ticks)

**Description:** Create logit locator with huge number of major, and tests ticks.

### Function: test_maxn_major(self, lims)

**Description:** When the axis is zoomed, the locator must have the same behavior as
MaxNLocator.

### Function: test_nbins_major(self, lims)

**Description:** Assert logit locator for respecting nbins param.

### Function: test_minor(self, lims, expected_low_ticks)

**Description:** In large scale, test the presence of minor,
and assert no minor when major are subsampled.

### Function: test_minor_attr(self)

### Function: test_nonsingular_ok(self, lims)

**Description:** Create logit locator, and test the nonsingular method for acceptable
value

### Function: test_nonsingular_nok(self, okval)

**Description:** Create logit locator, and test the nonsingular method for non
acceptable value

### Function: test_set_params(self)

**Description:** Create fixed locator with 5 nbins, and change it to something else.
See if change was successful.
Should not exception.

### Function: test_set_params(self)

**Description:** Create index locator with 3 base, 4 offset. and change it to something
else. See if change was successful.
Should not exception.

### Function: test_set_params(self)

**Description:** Create symmetrical log locator with default subs =[1.0] numticks = 15,
and change it to something else.
See if change was successful.
Should not exception.

### Function: test_values(self, vmin, vmax, expected)

### Function: test_subs(self)

### Function: test_extending(self)

### Function: test_init(self)

### Function: test_set_params(self)

### Function: test_linear_values(self)

### Function: test_wide_values(self)

### Function: test_near_zero(self)

**Description:** Check that manually injected zero will supersede nearby tick

### Function: test_fallback(self)

### Function: test_symmetrizing(self)

### Function: test_base_rounding(self)

### Function: test_unicode_minus(self, unicode_minus, result)

### Function: test_offset_value(self, left, right, offset)

### Function: test_use_offset(self, use_offset)

### Function: test_useMathText(self, use_math_text)

### Function: test_set_use_offset_float(self)

### Function: test_use_locale(self)

### Function: test_scilimits(self, sci_type, scilimits, lim, orderOfMag, fewticks)

### Function: test_format_data(self, value, expected)

### Function: test_cursor_precision(self, data, expected)

### Function: test_cursor_dummy_axis(self, data, expected)

### Function: test_mathtext_ticks(self)

### Function: test_cmr10_substitutions(self, caplog)

### Function: test_empty_locs(self)

### Function: test_basic(self, labelOnlyBase, base, exponent, locs, positions, expected)

### Function: test_blank(self)

### Function: test_min_exponent(self, min_exponent, value, expected)

### Function: test_basic(self, base, value, expected)

### Function: test_pprint(self, value, domain, expected)

### Function: test_format_data(self, value, long, short)

### Function: _sub_labels(self, axis, subs)

**Description:** Test whether locator marks subs to be labeled.

### Function: test_sublabel(self)

### Function: test_LogFormatter_call(self, val)

### Function: test_LogFormatter_call_tiny(self, val)

### Function: logit_deformatter(string)

**Description:** Parser to convert string as r'$\mathdefault{1.41\cdot10^{-4}}$' in
float 1.41e-4, as '0.5' or as r'$\mathdefault{\frac{1}{2}}$' in float
0.5,

### Function: test_logit_deformater(self, fx, x)

### Function: test_basic(self, x)

**Description:** Test the formatted value correspond to the value for ideal ticks in
logit space.

### Function: test_invalid(self, x)

**Description:** Test that invalid value are formatted with empty string without
raising exception.

### Function: test_variablelength(self, x)

**Description:** The format length should change depending on the neighbor labels.

### Function: test_minor_vs_major(self, method, lims, cases)

**Description:** Test minor/major displays.

### Function: test_minor_number(self)

**Description:** Test the parameter minor_number

### Function: test_use_overline(self)

**Description:** Test the parameter use_overline

### Function: test_one_half(self)

**Description:** Test the parameter one_half

### Function: test_format_data_short(self, N)

### Function: test_basic(self)

### Function: test_basic(self, format, input, unicode_minus, expected)

### Function: test_params(self, unicode_minus, input, expected)

**Description:** Test the formatting of EngFormatter for various values of the 'places'
argument, in several cases:

0. without a unit symbol but with a (default) space separator;
1. with both a unit symbol and a (default) space separator;
2. with both a unit symbol and some non default separators;
3. without a unit symbol but with some non default separators.

Note that cases 2. and 3. are looped over several separator strings.

### Function: test_basic(self, xmax, decimals, symbol, x, display_range, expected)

### Function: test_latex(self, is_latex, usetex, expected)

### Function: minorticksubplot(xminor, yminor, i)

### Function: minortickstoggle(xminor, yminor, scale, i)
