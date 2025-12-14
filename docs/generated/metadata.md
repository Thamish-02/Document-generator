## AI Summary

A file named metadata.py.


### Function: get_metadata(output, key, mimetype)

**Description:** Resolve an output metadata key

If mimetype given, resolve at mimetype level first,
then fallback to top-level.
Otherwise, just resolve at top-level.
Returns None if no data found.
