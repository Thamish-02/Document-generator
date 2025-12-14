## AI Summary

A file named threaded.py.


## Class: ThreadedZMQSocketChannel

**Description:** A ZMQ socket invoking a callback in the ioloop

## Class: IOLoopThread

**Description:** Run a pyzmq ioloop in a thread to send and receive messages

## Class: ThreadedKernelClient

**Description:** A KernelClient that provides thread-safe sockets with async callbacks on message replies.

### Function: __init__(self, socket, session, loop)

**Description:** Create a channel.

Parameters
----------
socket : :class:`zmq.Socket`
    The ZMQ socket to use.
session : :class:`session.Session`
    The session to use.
loop
    A tornado ioloop to connect the socket to using a ZMQStream

### Function: is_alive(self)

**Description:** Whether the channel is alive.

### Function: start(self)

**Description:** Start the channel.

### Function: stop(self)

**Description:** Stop the channel.

### Function: close(self)

**Description:** Close the channel.

### Function: send(self, msg)

**Description:** Queue a message to be sent from the IOLoop's thread.

Parameters
----------
msg : message to send

This is threadsafe, as it uses IOLoop.add_callback to give the loop's
thread control of the action.

### Function: _handle_recv(self, msg_list)

**Description:** Callback for stream.on_recv.

Unpacks message, and calls handlers with it.

### Function: call_handlers(self, msg)

**Description:** This method is called in the ioloop thread when a message arrives.

Subclasses should override this method to handle incoming messages.
It is important to remember that this method is called in the thread
so that some logic must be done to ensure that the application level
handlers are called in the application thread.

### Function: process_events(self)

**Description:** Subclasses should override this with a method
processing any pending GUI events.

### Function: flush(self, timeout)

**Description:** Immediately processes all pending messages on this channel.

This is only used for the IOPub channel.

Callers should use this method to ensure that :meth:`call_handlers`
has been called for all messages that have been received on the
0MQ SUB socket of this channel.

This method is thread safe.

Parameters
----------
timeout : float, optional
    The maximum amount of time to spend flushing, in seconds. The
    default is one second.

### Function: _flush(self)

**Description:** Callback for :method:`self.flush`.

### Function: __init__(self)

**Description:** Initialize an io loop thread.

### Function: _notice_exit()

### Function: start(self)

**Description:** Start the IOLoop thread

Don't return until self.ioloop is defined,
which is created in the thread

### Function: run(self)

**Description:** Run my loop, ignoring EINTR events in the poller

### Function: stop(self)

**Description:** Stop the channel's event loop and join its thread.

This calls :meth:`~threading.Thread.join` and returns when the thread
terminates. :class:`RuntimeError` will be raised if
:meth:`~threading.Thread.start` is called again.

### Function: __del__(self)

### Function: close(self)

**Description:** Close the io loop thread.

### Function: ioloop(self)

### Function: start_channels(self, shell, iopub, stdin, hb, control)

**Description:** Start the channels on the client.

### Function: _check_kernel_info_reply(self, msg)

**Description:** This is run in the ioloop thread when the kernel info reply is received

### Function: stop_channels(self)

**Description:** Stop the channels on the client.

### Function: is_alive(self)

**Description:** Is the kernel process still running?

### Function: setup_stream()

### Function: thread_send()

### Function: flush(f)

### Function: close_stream()
