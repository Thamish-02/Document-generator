## AI Summary

A file named Epoch.py.


## Class: Epoch

### Function: __init__(self, frame, sec, jd, daynum, dt)

**Description:** Create a new Epoch object.

Build an epoch 1 of 2 ways:

Using seconds past a Julian date:
#   Epoch('ET', sec=1e8, jd=2451545)

or using a matplotlib day number
#   Epoch('ET', daynum=730119.5)

= ERROR CONDITIONS
- If the input units are not in the allowed list, an error is thrown.

= INPUT VARIABLES
- frame     The frame of the epoch.  Must be 'ET' or 'UTC'
- sec        The number of seconds past the input JD.
- jd         The Julian date of the epoch.
- daynum    The matplotlib day number of the epoch.
- dt         A python datetime instance.

### Function: convert(self, frame)

### Function: frame(self)

### Function: julianDate(self, frame)

### Function: secondsPast(self, frame, jd)

### Function: _cmp(self, op, rhs)

**Description:** Compare Epochs *self* and *rhs* using operator *op*.

### Function: __add__(self, rhs)

**Description:** Add a duration to an Epoch.

= INPUT VARIABLES
- rhs     The Epoch to subtract.

= RETURN VALUE
- Returns the difference of ourselves and the input Epoch.

### Function: __sub__(self, rhs)

**Description:** Subtract two Epoch's or a Duration from an Epoch.

Valid:
Duration = Epoch - Epoch
Epoch = Epoch - Duration

= INPUT VARIABLES
- rhs     The Epoch to subtract.

= RETURN VALUE
- Returns either the duration between to Epoch's or the a new
  Epoch that is the result of subtracting a duration from an epoch.

### Function: __str__(self)

**Description:** Print the Epoch.

### Function: __repr__(self)

**Description:** Print the Epoch.

### Function: range(start, stop, step)

**Description:** Generate a range of Epoch objects.

Similar to the Python range() method.  Returns the range [
start, stop) at the requested step.  Each element will be a
Epoch object.

= INPUT VARIABLES
- start     The starting value of the range.
- stop      The stop value of the range.
- step      Step to use.

= RETURN VALUE
- Returns a list containing the requested Epoch values.
