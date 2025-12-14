## AI Summary

A file named eventloops.py.


### Function: _use_appnope()

**Description:** Should we use appnope for dealing with OS X app nap?

Checks if we are on OS X 10.9 or greater.

### Function: register_integration()

**Description:** Decorator to register an event loop to integrate with the IPython kernel

The decorator takes names to register the event loop as for the %gui magic.
You can provide alternative names for the same toolkit.

The decorated function should take a single argument, the IPython kernel
instance, arrange for the event loop to yield the asyncio loop when a
message is received by the main shell zmq stream or at least every
``kernel._poll_interval`` seconds, and start the event loop.

:mod:`ipykernel.eventloops` provides and registers such functions
for a few common event loops.

### Function: get_shell_stream(kernel)

### Function: _notify_stream_qt(kernel)

### Function: loop_qt(kernel)

**Description:** Event loop for all supported versions of Qt.

### Function: loop_qt_exit(kernel)

### Function: _loop_wx(app)

**Description:** Inner-loop for running the Wx eventloop

Pulled from guisupport.start_event_loop in IPython < 5.2,
since IPython 5.2 only checks `get_ipython().active_eventloop` is defined,
rather than if the eventloop is actually running.

### Function: loop_wx(kernel)

**Description:** Start a kernel with wx event loop support.

### Function: loop_wx_exit(kernel)

**Description:** Exit the wx loop.

### Function: loop_tk(kernel)

**Description:** Start a kernel with the Tk event loop.

### Function: loop_tk_exit(kernel)

**Description:** Exit the tk loop.

### Function: loop_gtk(kernel)

**Description:** Start the kernel, coordinating with the GTK event loop

### Function: loop_gtk_exit(kernel)

**Description:** Exit the gtk loop.

### Function: loop_gtk3(kernel)

**Description:** Start the kernel, coordinating with the GTK event loop

### Function: loop_gtk3_exit(kernel)

**Description:** Exit the gtk3 loop.

### Function: loop_cocoa(kernel)

**Description:** Start the kernel, coordinating with the Cocoa CFRunLoop event loop
via the matplotlib MacOSX backend.

### Function: loop_cocoa_exit(kernel)

**Description:** Exit the cocoa loop.

### Function: loop_asyncio(kernel)

**Description:** Start a kernel with asyncio event loop support.

### Function: loop_asyncio_exit(kernel)

**Description:** Exit hook for asyncio

### Function: set_qt_api_env_from_gui(gui)

**Description:** Sets the QT_API environment variable by trying to import PyQtx or PySidex.

The user can generically request `qt` or a specific Qt version, e.g. `qt6`.
For a generic Qt request, we let the mechanism in IPython choose the best
available version by leaving the `QT_API` environment variable blank.

For specific versions, we check to see whether the PyQt or PySide
implementations are present and set `QT_API` accordingly to indicate to
IPython which version we want. If neither implementation is present, we
leave the environment variable set so IPython will generate a helpful error
message.

Notes
-----
- If the environment variable is already set, it will be used unchanged,
  regardless of what the user requested.

### Function: make_qt_app_for_kernel(gui, kernel)

**Description:** Sets the `QT_API` environment variable if it isn't already set.

### Function: enable_gui(gui, kernel)

**Description:** Enable integration with a given GUI

### Function: decorator(func)

**Description:** Integration registration decorator.

### Function: exit_loop()

**Description:** fall back to main loop

### Function: process_stream_events_wrap(shell_stream)

**Description:** fall back to main loop when there's a socket event

### Function: _schedule_exit(delay)

**Description:** schedule fall back to main loop in [delay] seconds

### Function: wake(shell_stream)

**Description:** wake from wx

## Class: TimerFrame

## Class: IPWxApp

### Function: handle_int(etype, value, tb)

**Description:** don't let KeyboardInterrupts look like crashes

### Function: process_stream_events(shell_stream)

**Description:** fall back to main loop when there's a socket event

### Function: exit_decorator(exit_func)

**Description:** @func.exit is now a decorator

to register a function to be called on exit

### Function: __init__(self, func)

### Function: on_timer(self, event)

### Function: OnInit(self)

## Class: BasicAppWrapper

### Function: exit_loop()

**Description:** fall back to main loop

### Function: process_stream_events_wrap(shell_stream)

**Description:** fall back to main loop when there's a socket event

### Function: _schedule_exit(delay)

**Description:** schedule fall back to main loop in [delay] seconds

## Class: TimedAppWrapper

### Function: enum_helper(name)

### Function: __init__(self, app)

### Function: __init__(self, app, shell_stream)

### Function: on_timer(self)

### Function: start(self)
