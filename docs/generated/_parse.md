## AI Summary

A file named _parse.py.


## Class: ParsedLine

### Function: parse_ini_data(path, data)

**Description:** Parse INI data and return sections and sources mappings.

Args:
    path: Path for error messages
    data: INI content as string
    strip_inline_comments: Whether to strip inline comments from values
    strip_section_whitespace: Whether to strip whitespace from section and key names
        (default: False). When True, addresses issue #4 by stripping Unicode whitespace.

Returns:
    Tuple of (sections_data, sources) where:
    - sections_data: mapping of section -> {name -> value}
    - sources: mapping of (section, name) -> line number

### Function: parse_lines(path, line_iter)

### Function: _parseline(path, line, lineno, strip_inline_comments, strip_section_whitespace)

### Function: iscommentline(line)
