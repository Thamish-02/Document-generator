## AI Summary

A file named inputhookqt5.py.


## Class: InteractiveShell

### Function: create_inputhook_qt5(mgr, app)

**Description:** Create an input hook for running the Qt5 application event loop.

Parameters
----------
mgr : an InputHookManager

app : Qt Application, optional.
    Running application to use.  If not given, we probe Qt for an
    existing application object, and create a new one if none is found.

Returns
-------
A pair consisting of a Qt Application (either the one given or the
one found or created) and a inputhook.

Notes
-----
We use a custom input hook instead of PyQt5's default one, as it
interacts better with the readline packages (issue #481).

The inputhook function works in tandem with a 'pre_prompt_hook'
which automatically restores the hook as an inputhook in case the
latter has been temporarily disabled after having intercepted a
KeyboardInterrupt.

### Function: instance(cls)

### Function: set_hook(self)

### Function: inputhook_qt5()

**Description:** PyOS_InputHook python hook for Qt5.

Process pending Qt events and if there's no pending keyboard
input, spend a short slice of time (50ms) running the Qt event
loop.

As a Python ctypes callback can't raise an exception, we catch
the KeyboardInterrupt and temporarily deactivate the hook,
which will let a *second* CTRL+C be processed normally and go
back to a clean prompt line.

### Function: preprompthook_qt5(ishell)

**Description:** 'pre_prompt_hook' used to restore the Qt5 input hook

(in case the latter was temporarily deactivated after a
CTRL+C)
