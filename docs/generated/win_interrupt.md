## AI Summary

A file named win_interrupt.py.


### Function: create_interrupt_event()

**Description:** Create an interrupt event handle.

The parent process should call this to create the
interrupt event that is passed to the child process. It should store
this handle and use it with ``send_interrupt`` to interrupt the child
process.

### Function: send_interrupt(interrupt_handle)

**Description:** Sends an interrupt event using the specified handle.

## Class: SECURITY_ATTRIBUTES
