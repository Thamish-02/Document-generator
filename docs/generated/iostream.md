## AI Summary

A file named iostream.py.


## Class: IOPubThread

**Description:** An object for sending IOPub messages in a background thread

Prevents a blocking main thread from delaying output from threads.

IOPubThread(pub_socket).background_socket is a Socket-API-providing object
whose IO is always run in a thread.

## Class: BackgroundSocket

**Description:** Wrapper around IOPub thread that provides zmq send[_multipart]

## Class: OutStream

**Description:** A file like object that publishes the stream to a 0MQ PUB socket.

Output is handed off to an IO Thread

### Function: __init__(self, socket, pipe)

**Description:** Create IOPub thread

Parameters
----------
socket : zmq.PUB Socket
    the socket on which messages will be sent.
pipe : bool
    Whether this process should listen for IOPub messages
    piped from subprocesses.

### Function: _thread_main(self)

**Description:** The inner loop that's actually run in a thread

### Function: _setup_event_pipe(self)

**Description:** Create the PULL socket listening for events that should fire in this thread.

### Function: _event_pipe(self)

**Description:** thread-local event pipe for signaling events that should be processed in the thread

### Function: _handle_event(self, msg)

**Description:** Handle an event on the event pipe

Content of the message is ignored.

Whenever *an* event arrives on the event stream,
*all* waiting events are processed in order.

### Function: _setup_pipe_in(self)

**Description:** setup listening pipe for IOPub from forked subprocesses

### Function: _handle_pipe_msg(self, msg)

**Description:** handle a pipe message from a subprocess

### Function: _setup_pipe_out(self)

### Function: _is_master_process(self)

### Function: _check_mp_mode(self)

**Description:** check for forks, and switch to zmq pipeline if necessary

### Function: start(self)

**Description:** Start the IOPub thread

### Function: stop(self)

**Description:** Stop the IOPub thread

### Function: close(self)

**Description:** Close the IOPub thread.

### Function: closed(self)

### Function: schedule(self, f)

**Description:** Schedule a function to be called in our IO thread.

If the thread is not running, call immediately.

### Function: send_multipart(self)

**Description:** send_multipart schedules actual zmq send in my thread.

If my thread isn't running (e.g. forked process), send immediately.

### Function: _really_send(self, msg)

**Description:** The callback that actually sends messages

### Function: __init__(self, io_thread)

**Description:** Initialize the socket.

### Function: __getattr__(self, attr)

**Description:** Wrap socket attr access for backward-compatibility

### Function: __setattr__(self, attr, value)

**Description:** Set an attribute on the socket.

### Function: send(self, msg)

**Description:** Send a message to the socket.

### Function: send_multipart(self)

**Description:** Schedule send in IO thread

### Function: fileno(self)

**Description:** Things like subprocess will peak and write to the fileno() of stderr/stdout.

### Function: _watch_pipe_fd(self)

**Description:** We've redirected standards streams 0 and 1 into a pipe.

We need to watch in a thread and redirect them to the right places.

1) the ZMQ channels to show in notebook interfaces,
2) the original stdout/err, to capture errors in terminals.

We cannot schedule this on the ioloop thread, as this might be blocking.

### Function: __init__(self, session, pub_thread, name, pipe, echo)

**Description:** Parameters
----------
session : object
    the session object
pub_thread : threading.Thread
    the publication thread
name : str {'stderr', 'stdout'}
    the name of the standard stream to replace
pipe : object
    the pipe object
echo : bool
    whether to echo output
watchfd : bool (default, True)
    Watch the file descriptor corresponding to the replaced stream.
    This is useful if you know some underlying code will write directly
    the file descriptor by its number. It will spawn a watching thread,
    that will swap the give file descriptor for a pipe, read from the
    pipe, and insert this into the current Stream.
isatty : bool (default, False)
    Indication of whether this stream has terminal capabilities (e.g. can handle colors)

### Function: parent_header(self)

### Function: parent_header(self, value)

### Function: isatty(self)

**Description:** Return a bool indicating whether this is an 'interactive' stream.

Returns:
    Boolean

### Function: _setup_stream_redirects(self, name)

### Function: _is_master_process(self)

### Function: set_parent(self, parent)

**Description:** Set the parent header.

### Function: close(self)

**Description:** Close the stream.

### Function: closed(self)

### Function: _schedule_flush(self)

**Description:** schedule a flush in the IO thread

call this on write, to indicate that flush should be called soon.

### Function: flush(self)

**Description:** trigger actual zmq send

send will happen in the background thread

### Function: _flush(self)

**Description:** This is where the actual send happens.

_flush should generally be called in the IO thread,
unless the thread has been destroyed (e.g. forked subprocess).

### Function: write(self, string)

**Description:** Write to current stream after encoding if necessary

Returns
-------
len : int
    number of items from input parameter written to stream.

### Function: writelines(self, sequence)

**Description:** Write lines to the stream.

### Function: writable(self)

**Description:** Test whether the stream is writable.

### Function: _flush_buffers(self)

**Description:** clear the current buffer and return the current buffer data.

### Function: _rotate_buffers(self)

**Description:** Returns the current buffer and replaces it with an empty buffer.

### Function: _hooks(self)

### Function: register_hook(self, hook)

**Description:** Registers a hook with the thread-local storage.

Parameters
----------
hook : Any callable object

Returns
-------
Either a publishable message, or `None`.
The hook callable must return a message from
the __call__ method if they still require the
`session.send` method to be called after transformation.
Returning `None` will halt that execution path, and
session.send will not be called.

### Function: unregister_hook(self, hook)

**Description:** Un-registers a hook with the thread-local storage.

Parameters
----------
hook : Any callable object which has previously been
    registered as a hook.

Returns
-------
bool - `True` if the hook was removed, `False` if it wasn't
    found.

### Function: _start_event_gc()

### Function: _schedule_in_thread()
