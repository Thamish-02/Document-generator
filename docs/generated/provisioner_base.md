## AI Summary

A file named provisioner_base.py.


## Class: KernelProvisionerMeta

## Class: KernelProvisionerBase

**Description:** Abstract base class defining methods for KernelProvisioner classes.

A majority of methods are abstract (requiring implementations via a subclass) while
some are optional and others provide implementations common to all instances.
Subclasses should be aware of which methods require a call to the superclass.

Many of these methods model those of :class:`subprocess.Popen` for parity with
previous versions where the kernel process was managed directly.

### Function: has_process(self)

**Description:** Returns true if this provisioner is currently managing a process.

This property is asserted to be True immediately following a call to
the provisioner's :meth:`launch_kernel` method.

### Function: get_shutdown_wait_time(self, recommended)

**Description:** Returns the time allowed for a complete shutdown. This may vary by provisioner.

This method is called from `KernelManager.finish_shutdown()` during the graceful
phase of its kernel shutdown sequence.

The recommended value will typically be what is configured in the kernel manager.

### Function: get_stable_start_time(self, recommended)

**Description:** Returns the expected upper bound for a kernel (re-)start to complete.
This may vary by provisioner.

The recommended value will typically be what is configured in the kernel restarter.

### Function: _finalize_env(self, env)

**Description:** Ensures env is appropriate prior to launch.

This method is called from `KernelProvisionerBase.pre_launch()` during the kernel's
start sequence.

NOTE: Subclasses should be sure to call super()._finalize_env(env)

### Function: __apply_env_substitutions(self, substitution_values)

**Description:** Walks entries in the kernelspec's env stanza and applies substitutions from current env.

This method is called from `KernelProvisionerBase.pre_launch()` during the kernel's
start sequence.

Returns the substituted list of env entries.

NOTE: This method is private and is not intended to be overridden by provisioners.
