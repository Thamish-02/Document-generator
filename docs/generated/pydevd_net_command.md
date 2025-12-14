## AI Summary

A file named pydevd_net_command.py.


## Class: _BaseNetCommand

## Class: _NullNetCommand

## Class: _NullExitCommand

## Class: NetCommand

**Description:** Commands received/sent over the network.

Command can represent command received from the debugger,
or one to be sent by daemon.

### Function: send(self)

### Function: call_after_send(self, callback)

### Function: __init__(self, cmd_id, seq, text, is_json)

**Description:** If sequence is 0, new sequence will be generated (otherwise, this was the response
to a command from the client).

### Function: send(self, sock)

### Function: call_after_send(self, callback)

### Function: _show_debug_info(cls, cmd_id, seq, text)
