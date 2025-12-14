## AI Summary

A file named local_provisioner.py.


## Class: LocalProvisioner

**Description:** :class:`LocalProvisioner` is a concrete class of ABC :py:class:`KernelProvisionerBase`
and is the out-of-box default implementation used when no kernel provisioner is
specified in the kernel specification (``kernel.json``).  It provides functional
parity to existing applications by launching the kernel locally and using
:class:`subprocess.Popen` to manage its lifecycle.

This class is intended to be subclassed for customizing local kernel environments
and serve as a reference implementation for other custom provisioners.

### Function: has_process(self)

### Function: _tolerate_no_process(os_error)

### Function: _scrub_kwargs(kwargs)

**Description:** Remove any keyword arguments that Popen does not tolerate.
