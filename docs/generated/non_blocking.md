## AI Summary

A file named non_blocking.py.


### Function: make_non_blocking(file_obj)

**Description:** make file object non-blocking

Windows doesn't have the fcntl module, but someone on
stack overflow supplied this code as an answer, and it works
http://stackoverflow.com/a/34504971/2893090
