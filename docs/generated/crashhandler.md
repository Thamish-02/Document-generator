## AI Summary

A file named crashhandler.py.


## Class: CrashHandler

**Description:** Customizable crash handlers for IPython applications.

Instances of this class provide a :meth:`__call__` method which can be
used as a ``sys.excepthook``.  The :meth:`__call__` signature is::

    def __call__(self, etype, evalue, etb)

### Function: crash_handler_lite(etype, evalue, tb)

**Description:** a light excepthook, adding a small message to the usual traceback

### Function: __init__(self, app, contact_name, contact_email, bug_tracker, show_crash_traceback, call_pdb)

**Description:** Create a new crash handler

Parameters
----------
app : Application
    A running :class:`Application` instance, which will be queried at
    crash time for internal information.
contact_name : str
    A string with the name of the person to contact.
contact_email : str
    A string with the email address of the contact.
bug_tracker : str
    A string with the URL for your project's bug tracker.
show_crash_traceback : bool
    If false, don't print the crash traceback on stderr, only generate
    the on-disk report
call_pdb
    Whether to call pdb on crash

Attributes
----------
These instances contain some non-argument attributes which allow for
further customization of the crash handler's behavior. Please see the
source for further details.

### Function: __call__(self, etype, evalue, etb)

**Description:** Handle an exception, call for compatible with sys.excepthook

### Function: make_report(self, traceback)

**Description:** Return a string containing a crash report.
