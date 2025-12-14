## AI Summary

A file named blocking.py.


## Class: BlockingInProcessChannel

**Description:** A blocking in-process channel.

## Class: BlockingInProcessStdInChannel

**Description:** A blocking in-process stdin channel.

## Class: BlockingInProcessKernelClient

**Description:** A blocking in-process kernel client.

### Function: __init__(self)

**Description:** Initialize the channel.

### Function: call_handlers(self, msg)

**Description:** Call the handlers for a message.

### Function: get_msg(self, block, timeout)

**Description:** Gets a message if there is one that is ready.

### Function: get_msgs(self)

**Description:** Get all messages that are currently ready.

### Function: msg_ready(self)

**Description:** Is there a message that has been received?

### Function: call_handlers(self, msg)

**Description:** Overridden for the in-process channel.

This methods simply calls raw_input directly.

### Function: wait_for_ready(self)

**Description:** Wait for kernel info reply on shell channel.
