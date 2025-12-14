## AI Summary

A file named inputhook.py.


### Function: ignore_CTRL_C()

**Description:** Ignore CTRL+C (not implemented).

### Function: allow_CTRL_C()

**Description:** Take CTRL+C into account (not implemented).

## Class: InputHookManager

**Description:** Manage PyOS_InputHook for different GUI toolkits.

This class installs various hooks under ``PyOSInputHook`` to handle
GUI event loop integration.

### Function: enable_gui(gui, app)

**Description:** Switch amongst GUI input hooks by name.

This is just a utility wrapper around the methods of the InputHookManager
object.

Parameters
----------
gui : optional, string or None
  If None (or 'none'), clears input hook, otherwise it must be one
  of the recognized GUI names (see ``GUI_*`` constants in module).

app : optional, existing application object.
  For toolkits that have the concept of a global app, you can supply an
  existing one.  If not given, the toolkit will be probed for one, and if
  none is found, a new one will be created.  Note that GTK does not have
  this concept, and passing an app if ``gui=="GTK"`` will raise an error.

Returns
-------
The output of the underlying gui switch routine, typically the actual
PyOS_InputHook wrapper object or the GUI toolkit app created, if there was
one.

### Function: __init__(self)

### Function: _reset(self)

### Function: set_return_control_callback(self, return_control_callback)

### Function: get_return_control_callback(self)

### Function: return_control(self)

### Function: get_inputhook(self)

### Function: set_inputhook(self, callback)

**Description:** Set inputhook to callback.

### Function: clear_inputhook(self, app)

**Description:** Clear input hook.

Parameters
----------
app : optional, ignored
  This parameter is allowed only so that clear_inputhook() can be
  called with a similar interface as all the ``enable_*`` methods.  But
  the actual value of the parameter is ignored.  This uniform interface
  makes it easier to have user-level entry points in the main IPython
  app like :meth:`enable_gui`.

### Function: clear_app_refs(self, gui)

**Description:** Clear IPython's internal reference to an application instance.

Whenever we create an app for a user on qt4 or wx, we hold a
reference to the app.  This is needed because in some cases bad things
can happen if a user doesn't hold a reference themselves.  This
method is provided to clear the references we are holding.

Parameters
----------
gui : None or str
    If None, clear all app references.  If ('wx', 'qt4') clear
    the app for that toolkit.  References are not held for gtk or tk
    as those toolkits don't have the notion of an app.

### Function: enable_wx(self, app)

**Description:** Enable event loop integration with wxPython.

Parameters
----------
app : WX Application, optional.
    Running application to use.  If not given, we probe WX for an
    existing application object, and create a new one if none is found.

Notes
-----
This methods sets the ``PyOS_InputHook`` for wxPython, which allows
the wxPython to integrate with terminal based applications like
IPython.

If ``app`` is not given we probe for an existing one, and return it if
found.  If no existing app is found, we create an :class:`wx.App` as
follows::

    import wx
    app = wx.App(redirect=False, clearSigInt=False)

### Function: disable_wx(self)

**Description:** Disable event loop integration with wxPython.

This merely sets PyOS_InputHook to NULL.

### Function: enable_qt(self, app)

### Function: enable_qt4(self, app)

**Description:** Enable event loop integration with PyQt4.

Parameters
----------
app : Qt Application, optional.
    Running application to use.  If not given, we probe Qt for an
    existing application object, and create a new one if none is found.

Notes
-----
This methods sets the PyOS_InputHook for PyQt4, which allows
the PyQt4 to integrate with terminal based applications like
IPython.

If ``app`` is not given we probe for an existing one, and return it if
found.  If no existing app is found, we create an :class:`QApplication`
as follows::

    from PyQt4 import QtCore
    app = QtGui.QApplication(sys.argv)

### Function: disable_qt4(self)

**Description:** Disable event loop integration with PyQt4.

This merely sets PyOS_InputHook to NULL.

### Function: enable_qt5(self, app)

### Function: disable_qt5(self)

### Function: enable_qt6(self, app)

### Function: disable_qt6(self)

### Function: enable_gtk(self, app)

**Description:** Enable event loop integration with PyGTK.

Parameters
----------
app : ignored
   Ignored, it's only a placeholder to keep the call signature of all
   gui activation methods consistent, which simplifies the logic of
   supporting magics.

Notes
-----
This methods sets the PyOS_InputHook for PyGTK, which allows
the PyGTK to integrate with terminal based applications like
IPython.

### Function: disable_gtk(self)

**Description:** Disable event loop integration with PyGTK.

This merely sets PyOS_InputHook to NULL.

### Function: enable_tk(self, app)

**Description:** Enable event loop integration with Tk.

Parameters
----------
app : toplevel :class:`Tkinter.Tk` widget, optional.
    Running toplevel widget to use.  If not given, we probe Tk for an
    existing one, and create a new one if none is found.

Notes
-----
If you have already created a :class:`Tkinter.Tk` object, the only
thing done by this method is to register with the
:class:`InputHookManager`, since creating that object automatically
sets ``PyOS_InputHook``.

### Function: disable_tk(self)

**Description:** Disable event loop integration with Tkinter.

This merely sets PyOS_InputHook to NULL.

### Function: enable_glut(self, app)

**Description:** Enable event loop integration with GLUT.

Parameters
----------

app : ignored
    Ignored, it's only a placeholder to keep the call signature of all
    gui activation methods consistent, which simplifies the logic of
    supporting magics.

Notes
-----

This methods sets the PyOS_InputHook for GLUT, which allows the GLUT to
integrate with terminal based applications like IPython. Due to GLUT
limitations, it is currently not possible to start the event loop
without first creating a window. You should thus not create another
window but use instead the created one. See 'gui-glut.py' in the
docs/examples/lib directory.

The default screen mode is set to:
glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH

### Function: disable_glut(self)

**Description:** Disable event loop integration with glut.

This sets PyOS_InputHook to NULL and set the display function to a
dummy one and set the timer to a dummy timer that will be triggered
very far in the future.

### Function: enable_pyglet(self, app)

**Description:** Enable event loop integration with pyglet.

Parameters
----------
app : ignored
   Ignored, it's only a placeholder to keep the call signature of all
   gui activation methods consistent, which simplifies the logic of
   supporting magics.

Notes
-----
This methods sets the ``PyOS_InputHook`` for pyglet, which allows
pyglet to integrate with terminal based applications like
IPython.

### Function: disable_pyglet(self)

**Description:** Disable event loop integration with pyglet.

This merely sets PyOS_InputHook to NULL.

### Function: enable_gtk3(self, app)

**Description:** Enable event loop integration with Gtk3 (gir bindings).

Parameters
----------
app : ignored
   Ignored, it's only a placeholder to keep the call signature of all
   gui activation methods consistent, which simplifies the logic of
   supporting magics.

Notes
-----
This methods sets the PyOS_InputHook for Gtk3, which allows
the Gtk3 to integrate with terminal based applications like
IPython.

### Function: disable_gtk3(self)

**Description:** Disable event loop integration with PyGTK.

This merely sets PyOS_InputHook to NULL.

### Function: enable_mac(self, app)

**Description:** Enable event loop integration with MacOSX.

We call function pyplot.pause, which updates and displays active
figure during pause. It's not MacOSX-specific, but it enables to
avoid inputhooks in native MacOSX backend.
Also we shouldn't import pyplot, until user does it. Cause it's
possible to choose backend before importing pyplot for the first
time only.

### Function: disable_mac(self)

### Function: current_gui(self)

**Description:** Return a string indicating the currently active GUI or None.

### Function: inputhook_mac(app)
