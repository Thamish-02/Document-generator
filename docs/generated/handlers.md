## AI Summary

A file named handlers.py.


## Class: TermSocket

**Description:** A terminal websocket.

### Function: initialize(self, name, term_manager)

**Description:** Initialize the socket.

### Function: origin_check(self, origin)

**Description:** Terminado adds redundant origin_check
Tornado already calls check_origin, so don't do anything here.

### Function: write_message(self, message, binary)

**Description:** Write a message to the socket.

### Function: _update_activity(self)
