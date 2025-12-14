## AI Summary

A file named payloadpage.py.


### Function: page(strng, start, screen_lines, pager_cmd)

**Description:** Print a string, piping through a pager.

This version ignores the screen_lines and pager_cmd arguments and uses
IPython's payload system instead.

Parameters
----------
strng : str or mime-dict
    Text to page, or a mime-type keyed dict of already formatted data.
start : int
    Starting line at which to place the display.

### Function: install_payload_page()

**Description:** DEPRECATED, use show_in_pager hook

Install this version of page as IPython.core.page.page.
