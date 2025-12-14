## AI Summary

A file named testutils.py.


### Function: approx(a, b, fill_value, rtol, atol)

**Description:** Returns true if all components of a and b are equal to given tolerances.

If fill_value is True, masked values considered equal. Otherwise,
masked values are considered unequal.  The relative error rtol should
be positive and << 1.0 The absolute error atol comes into play for
those elements of b that are very small or zero; it says how small a
must be also.

### Function: almost(a, b, decimal, fill_value)

**Description:** Returns True if a and b are equal up to decimal places.

If fill_value is True, masked values considered equal. Otherwise,
masked values are considered unequal.

### Function: _assert_equal_on_sequences(actual, desired, err_msg)

**Description:** Asserts the equality of two non-array sequences.

### Function: assert_equal_records(a, b)

**Description:** Asserts that two records are equal.

Pretty crude for now.

### Function: assert_equal(actual, desired, err_msg)

**Description:** Asserts that two items are equal.

### Function: fail_if_equal(actual, desired, err_msg)

**Description:** Raises an assertion error if two items are equal.

### Function: assert_almost_equal(actual, desired, decimal, err_msg, verbose)

**Description:** Asserts that two items are almost equal.

The test is equivalent to abs(desired-actual) < 0.5 * 10**(-decimal).

### Function: assert_array_compare(comparison, x, y, err_msg, verbose, header, fill_value)

**Description:** Asserts that comparison between two masked arrays is satisfied.

The comparison is elementwise.

### Function: assert_array_equal(x, y, err_msg, verbose)

**Description:** Checks the elementwise equality of two masked arrays.

### Function: fail_if_array_equal(x, y, err_msg, verbose)

**Description:** Raises an assertion error if two masked arrays are not equal elementwise.

### Function: assert_array_approx_equal(x, y, decimal, err_msg, verbose)

**Description:** Checks the equality of two masked arrays, up to given number odecimals.

The equality is checked elementwise.

### Function: assert_array_almost_equal(x, y, decimal, err_msg, verbose)

**Description:** Checks the equality of two masked arrays, up to given number odecimals.

The equality is checked elementwise.

### Function: assert_array_less(x, y, err_msg, verbose)

**Description:** Checks that x is smaller than y elementwise.

### Function: assert_mask_equal(m1, m2, err_msg)

**Description:** Asserts the equality of two masks.

### Function: compare(x, y)

### Function: compare(x, y)

**Description:** Returns the result of the loose comparison between x and y).

### Function: compare(x, y)

**Description:** Returns the result of the loose comparison between x and y).
