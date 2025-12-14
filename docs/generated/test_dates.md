## AI Summary

A file named test_dates.py.


### Function: test_date_numpyx()

### Function: test_date_date2num_numpy(t0, dtype)

### Function: test_date2num_NaT(dtype)

### Function: test_date2num_NaT_scalar(units)

### Function: test_date2num_masked()

### Function: test_date_empty()

### Function: test_date_not_empty()

### Function: test_axhline()

### Function: test_date_axhspan()

### Function: test_date_axvspan()

### Function: test_date_axhline()

### Function: test_date_axvline()

### Function: test_too_many_date_ticks(caplog)

### Function: _new_epoch_decorator(thefunc)

### Function: test_RRuleLocator()

### Function: test_RRuleLocator_dayrange()

### Function: test_RRuleLocator_close_minmax()

### Function: test_DateFormatter()

### Function: test_locator_set_formatter()

**Description:** Test if setting the locator only will update the AutoDateFormatter to use
the new locator.

### Function: test_date_formatter_callable()

### Function: test_date_formatter_usetex(delta, expected)

### Function: test_drange()

**Description:** This test should check if drange works as expected, and if all the
rounding errors are fixed

### Function: test_auto_date_locator()

### Function: test_auto_date_locator_intmult()

### Function: test_concise_formatter_subsecond()

### Function: test_concise_formatter()

### Function: test_concise_formatter_show_offset(t_delta, expected)

### Function: test_concise_formatter_show_offset_inverted()

### Function: test_concise_converter_stays()

### Function: test_offset_changes()

### Function: test_concise_formatter_usetex(t_delta, expected)

### Function: test_concise_formatter_formats()

### Function: test_concise_formatter_zformats()

### Function: test_concise_formatter_tz()

### Function: test_auto_date_locator_intmult_tz()

### Function: test_date_inverted_limit()

### Function: _test_date2num_dst(date_range, tz_convert)

### Function: test_date2num_dst()

### Function: test_date2num_dst_pandas(pd)

### Function: _test_rrulewrapper(attach_tz, get_tz)

### Function: test_rrulewrapper()

### Function: test_rrulewrapper_pytz()

### Function: test_yearlocator_pytz()

### Function: test_YearLocator()

### Function: test_DayLocator()

### Function: test_tz_utc()

### Function: test_num2timedelta(x, tdelta)

### Function: test_datetime64_in_list()

### Function: test_change_epoch()

### Function: test_warn_notintervals()

### Function: test_change_converter()

### Function: test_change_interval_multiples()

### Function: test_DateLocator()

### Function: test_datestr2num()

### Function: test_concise_formatter_exceptions(kwarg)

### Function: test_concise_formatter_call()

### Function: test_datetime_masked()

### Function: test_num2date_error(val)

### Function: test_num2date_roundoff()

### Function: test_DateFormatter_settz()

### Function: wrapper()

## Class: _Locator

### Function: callable_formatting_function(dates, _)

### Function: _create_auto_date_locator(date1, date2)

### Function: _create_auto_date_locator(date1, date2)

### Function: _create_auto_date_locator(date1, date2)

### Function: _create_auto_date_locator(date1, date2)

### Function: _create_auto_date_locator(date1, date2)

### Function: _create_auto_date_locator(date1, date2, tz)

### Function: _create_auto_date_locator(date1, date2, tz)

## Class: dt_tzaware

**Description:** This bug specifically occurs because of the normalization behavior of
pandas Timestamp objects, so in order to replicate it, we need a
datetime-like object that applies timezone normalization after
subtraction.

### Function: date_range(start, freq, periods)

### Function: tz_convert(dt_list, tzinfo)

### Function: tz_convert()

### Function: attach_tz(dt, zi)

### Function: attach_tz(dt, zi)

### Function: _create_year_locator(date1, date2)

### Function: _get_unit(self)

### Function: __sub__(self, other)

### Function: __add__(self, other)

### Function: astimezone(self, tzinfo)

### Function: mk_tzaware(cls, datetime_obj)
