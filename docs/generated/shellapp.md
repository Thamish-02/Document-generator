## AI Summary

A file named shellapp.py.


## Class: MatplotlibBackendCaselessStrEnum

**Description:** An enum of Matplotlib backend strings where the case should be ignored.

Prior to Matplotlib 3.9.0 the list of valid backends is hardcoded in
pylabtools.backends. After that, Matplotlib manages backends.

The list of valid backends is determined when it is first needed to avoid
wasting unnecessary initialisation time.

## Class: InteractiveShellApp

**Description:** A Mixin for applications that start InteractiveShell instances.

Provides configurables for loading extensions and executing files
as part of configuring a Shell environment.

The following methods should be called by the :meth:`initialize` method
of the subclass:

  - :meth:`init_path`
  - :meth:`init_shell` (to be implemented by the subclass)
  - :meth:`init_gui_pylab`
  - :meth:`init_extensions`
  - :meth:`init_code`

### Function: __init__(self, default_value)

### Function: __getattribute__(self, name)

### Function: _user_ns_changed(self, change)

### Function: init_path(self)

**Description:** Add current working directory, '', to sys.path

Unlike Python's default, we insert before the first `site-packages`
or `dist-packages` directory,
so that it is after the standard library.

.. versionchanged:: 7.2
    Try to insert after the standard library, instead of first.
.. versionchanged:: 8.0
    Allow optionally not including the current directory in sys.path

### Function: init_shell(self)

### Function: init_gui_pylab(self)

**Description:** Enable GUI event loop integration, taking pylab into account.

### Function: init_extensions(self)

**Description:** Load all IPython extensions in IPythonApp.extensions.

This uses the :meth:`ExtensionManager.load_extensions` to load all
the extensions listed in ``self.extensions``.

### Function: init_code(self)

**Description:** run the pre-flight code, specified via exec_lines

### Function: _run_exec_lines(self)

**Description:** Run lines of code in IPythonApp.exec_lines in the user's namespace.

### Function: _exec_file(self, fname, shell_futures)

### Function: _run_startup_files(self)

**Description:** Run files from profile startup directory

### Function: _run_exec_files(self)

**Description:** Run files from IPythonApp.exec_files

### Function: _run_cmd_line_code(self)

**Description:** Run code or file specified at the command-line

### Function: _run_module(self)

**Description:** Run module specified at the command-line.
