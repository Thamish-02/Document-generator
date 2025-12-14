## AI Summary

A file named pydev_localhost.py.


### Function: get_localhost()

**Description:** Should return 127.0.0.1 in ipv4 and ::1 in ipv6

localhost is not used because on windows vista/windows 7, there can be issues where the resolving doesn't work
properly and takes a lot of time (had this issue on the pyunit server).

Using the IP directly solves the problem.

### Function: get_socket_names(n_sockets, close)

### Function: get_socket_name(close)
