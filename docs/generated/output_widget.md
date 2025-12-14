## AI Summary

A file named output_widget.py.


## Class: OutputWidget

**Description:** This class mimics a front end output widget

### Function: __init__(self, comm_id, state, kernel_client, executor)

**Description:** Initialize the widget.

### Function: clear_output(self, outs, msg, cell_index)

**Description:** Clear output.

### Function: sync_state(self)

**Description:** Sync state.

### Function: _publish_msg(self, msg_type, data, metadata, buffers)

**Description:** Helper for sending a comm message on IOPub

### Function: send(self, data, metadata, buffers)

**Description:** Send a comm message.

### Function: output(self, outs, msg, display_id, cell_index)

**Description:** Handle output.

### Function: set_state(self, state)

**Description:** Set the state.

### Function: handle_msg(self, msg)

**Description:** Handle a message.
