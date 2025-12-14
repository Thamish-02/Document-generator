## AI Summary

A file named ipapp.py.


## Class: IPAppCrashHandler

**Description:** sys.excepthook for IPython itself, leaves a detailed report on disk.

## Class: LocateIPythonApp

## Class: TerminalIPythonApp

### Function: load_default_config(ipython_dir)

**Description:** Load the default config file from the default ipython_dir.

This is useful for embedded shells.

### Function: __init__(self, app)

### Function: make_report(self, traceback)

**Description:** Return a string containing a crash report.

### Function: start(self)

### Function: _classes_default(self)

**Description:** This has to be in a method, for TerminalIPythonApp to be available.

### Function: _quick_changed(self, change)

### Function: _force_interact_changed(self, change)

### Function: _file_to_run_changed(self, change)

### Function: initialize(self, argv)

**Description:** Do actions after construct, but before starting the app.

### Function: init_shell(self)

**Description:** initialize the InteractiveShell instance

### Function: init_banner(self)

**Description:** optionally display the banner

### Function: _pylab_changed(self, name, old, new)

**Description:** Replace --pylab='inline' with --pylab='auto'

### Function: start(self)
