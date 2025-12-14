## AI Summary

A file named pydev_monkey_qt.py.


### Function: set_trace_in_qt()

### Function: patch_qt(qt_support_mode)

**Description:** This method patches qt (PySide2, PySide, PyQt4, PyQt5) so that we have hooks to set the tracing for QThread.

### Function: _patch_import_to_patch_pyqt_on_import(patch_qt_on_import, get_qt_core_module)

### Function: _internal_patch_qt(QtCore, qt_support_mode)

### Function: patched_import(name)

## Class: FuncWrapper

## Class: StartedSignalWrapper

## Class: ThreadWrapper

## Class: RunnableWrapper

### Function: __init__(self, original)

### Function: __call__(self)

### Function: __init__(self, thread, original_started)

### Function: connect(self, func)

### Function: disconnect(self)

### Function: emit(self)

### Function: _on_call(self)

### Function: __init__(self)

### Function: _exec_run(self)

### Function: _new_run(self)

### Function: __init__(self)

### Function: _new_run(self)

### Function: get_qt_core_module()
