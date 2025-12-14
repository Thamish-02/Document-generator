## AI Summary

A file named autocall.py.


## Class: IPyAutocall

**Description:** Instances of this class are always autocalled

This happens regardless of 'autocall' variable state. Use this to
develop macro-like mechanisms.

## Class: ExitAutocall

**Description:** An autocallable object which will be added to the user namespace so that
exit, exit(), quit or quit() are all valid ways to close the shell.

## Class: ZMQExitAutocall

**Description:** Exit IPython. Autocallable, so it needn't be explicitly called.

Parameters
----------
keep_kernel : bool
  If True, leave the kernel alive. Otherwise, tell the kernel to exit too
  (default).

### Function: __init__(self, ip)

### Function: set_ip(self, ip)

**Description:** Will be used to set _ip point to current ipython instance b/f call

Override this method if you don't want this to happen.

### Function: __call__(self)

### Function: __call__(self, keep_kernel)
