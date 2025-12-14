## AI Summary

A file named Duration.py.


## Class: Duration

**Description:** Class Duration in development.

### Function: __init__(self, frame, seconds)

**Description:** Create a new Duration object.

= ERROR CONDITIONS
- If the input frame is not in the allowed list, an error is thrown.

= INPUT VARIABLES
- frame     The frame of the duration.  Must be 'ET' or 'UTC'
- seconds  The number of seconds in the Duration.

### Function: frame(self)

**Description:** Return the frame the duration is in.

### Function: __abs__(self)

**Description:** Return the absolute value of the duration.

### Function: __neg__(self)

**Description:** Return the negative value of this Duration.

### Function: seconds(self)

**Description:** Return the number of seconds in the Duration.

### Function: __bool__(self)

### Function: _cmp(self, op, rhs)

**Description:** Check that *self* and *rhs* share frames; compare them using *op*.

### Function: __add__(self, rhs)

**Description:** Add two Durations.

= ERROR CONDITIONS
- If the input rhs is not in the same frame, an error is thrown.

= INPUT VARIABLES
- rhs     The Duration to add.

= RETURN VALUE
- Returns the sum of ourselves and the input Duration.

### Function: __sub__(self, rhs)

**Description:** Subtract two Durations.

= ERROR CONDITIONS
- If the input rhs is not in the same frame, an error is thrown.

= INPUT VARIABLES
- rhs     The Duration to subtract.

= RETURN VALUE
- Returns the difference of ourselves and the input Duration.

### Function: __mul__(self, rhs)

**Description:** Scale a UnitDbl by a value.

= INPUT VARIABLES
- rhs     The scalar to multiply by.

= RETURN VALUE
- Returns the scaled Duration.

### Function: __str__(self)

**Description:** Print the Duration.

### Function: __repr__(self)

**Description:** Print the Duration.

### Function: checkSameFrame(self, rhs, func)

**Description:** Check to see if frames are the same.

= ERROR CONDITIONS
- If the frame of the rhs Duration is not the same as our frame,
  an error is thrown.

= INPUT VARIABLES
- rhs     The Duration to check for the same frame
- func    The name of the function doing the check.
