## AI Summary

A file named conv_template.py.


### Function: parse_structure(astr, level)

**Description:** The returned line number is from the beginning of the string, starting
at zero. Returns an empty list if no loops found.

### Function: paren_repl(obj)

### Function: parse_values(astr)

### Function: parse_loop_header(loophead)

**Description:** Find all named replacements in the header

Returns a list of dictionaries, one for each loop iteration,
where each key is a name to be substituted and the corresponding
value is the replacement string.

Also return a list of exclusions.  The exclusions are dictionaries
 of key value pairs. There can be more than one exclusion.
 [{'var1':'value1', 'var2', 'value2'[,...]}, ...]

### Function: parse_string(astr, env, level, line)

### Function: process_str(astr)

### Function: resolve_includes(source)

### Function: process_file(source)

### Function: unique_key(adict)

### Function: main()

### Function: replace(match)
