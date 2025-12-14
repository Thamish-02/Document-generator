## AI Summary

A file named array.py.


### Function: codes_from_offsets(offsets)

**Description:** Determine codes from offsets, assuming they all correspond to closed polygons.
    

### Function: codes_from_offsets_and_points(offsets, points)

**Description:** Determine codes from offsets and points, using the equality of the start and end points of
each line to determine if lines are closed or not.

### Function: codes_from_points(points)

**Description:** Determine codes for a single line, using the equality of the start and end points to
determine if the line is closed or not.

### Function: concat_codes(list_of_codes)

**Description:** Concatenate a list of codes arrays into a single code array.
    

### Function: concat_codes_or_none(list_of_codes_or_none)

**Description:** Concatenate a list of codes arrays or None into a single code array or None.
    

### Function: concat_offsets(list_of_offsets)

**Description:** Concatenate a list of offsets arrays into a single offset array.
    

### Function: concat_offsets_or_none(list_of_offsets_or_none)

**Description:** Concatenate a list of offsets arrays or None into a single offset array or None.
    

### Function: concat_points(list_of_points)

**Description:** Concatenate a list of point arrays into a single point array.
    

### Function: concat_points_or_none(list_of_points_or_none)

**Description:** Concatenate a list of point arrays or None into a single point array or None.
    

### Function: concat_points_or_none_with_nan(list_of_points_or_none)

**Description:** Concatenate a list of points or None into a single point array or None, with NaNs used to
separate each line.

### Function: concat_points_with_nan(list_of_points)

**Description:** Concatenate a list of points into a single point array with NaNs used to separate each line.
    

### Function: insert_nan_at_offsets(points, offsets)

**Description:** Insert NaNs into a point array at locations specified by an offset array.
    

### Function: offsets_from_codes(codes)

**Description:** Determine offsets from codes using locations of MOVETO codes.
    

### Function: offsets_from_lengths(list_of_points)

**Description:** Determine offsets from lengths of point arrays.
    

### Function: outer_offsets_from_list_of_codes(list_of_codes)

**Description:** Determine outer offsets from codes using locations of MOVETO codes.
    

### Function: outer_offsets_from_list_of_offsets(list_of_offsets)

**Description:** Determine outer offsets from a list of offsets.
    

### Function: remove_nan(points)

**Description:** Remove NaN from a points array, also return the offsets corresponding to the NaN removed.
    

### Function: split_codes_by_offsets(codes, offsets)

**Description:** Split a code array at locations specified by an offset array into a list of code arrays.
    

### Function: split_points_by_offsets(points, offsets)

**Description:** Split a point array at locations specified by an offset array into a list of point arrays.
    

### Function: split_points_at_nan(points)

**Description:** Split a points array at NaNs into a list of point arrays.
    
