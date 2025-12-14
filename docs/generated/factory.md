## AI Summary

A file named factory.py.


## Class: KernelProvisionerFactory

**Description:** :class:`KernelProvisionerFactory` is responsible for creating provisioner instances.

A singleton instance, `KernelProvisionerFactory` is also used by the :class:`KernelSpecManager`
to validate `kernel_provisioner` references found in kernel specifications to confirm their
availability (in cases where the kernel specification references a kernel provisioner that has
not been installed into the current Python environment).

It's ``default_provisioner_name`` attribute can be used to specify the default provisioner
to use when a kernel_spec is found to not reference a provisioner.  It's value defaults to
`"local-provisioner"` which identifies the local provisioner implemented by
:class:`LocalProvisioner`.

### Function: _default_provisioner_name_default(self)

**Description:** The default provisioner name.

### Function: __init__(self)

**Description:** Initialize a kernel provisioner factory.

### Function: is_provisioner_available(self, kernel_spec)

**Description:** Reads the associated ``kernel_spec`` to determine the provisioner and returns whether it
exists as an entry_point (True) or not (False).  If the referenced provisioner is not
in the current cache or cannot be loaded via entry_points, a warning message is issued
indicating it is not available.

### Function: create_provisioner_instance(self, kernel_id, kernel_spec, parent)

**Description:** Reads the associated ``kernel_spec`` to see if it has a `kernel_provisioner` stanza.
If one exists, it instantiates an instance.  If a kernel provisioner is not
specified in the kernel specification, a default provisioner stanza is fabricated
and instantiated corresponding to the current value of ``default_provisioner_name`` trait.
The instantiated instance is returned.

If the provisioner is found to not exist (not registered via entry_points),
`ModuleNotFoundError` is raised.

### Function: _check_availability(self, provisioner_name)

**Description:** Checks that the given provisioner is available.

If the given provisioner is not in the current set of loaded provisioners an attempt
is made to fetch the named entry point and, if successful, loads it into the cache.

:param provisioner_name:
:return:

### Function: _get_provisioner_config(self, kernel_spec)

**Description:** Return the kernel_provisioner stanza from the kernel_spec.

Checks the kernel_spec's metadata dictionary for a kernel_provisioner entry.
If found, it is returned, else one is created relative to the DEFAULT_PROVISIONER
and returned.

Parameters
----------
kernel_spec : Any - this is a KernelSpec type but listed as Any to avoid circular import
    The kernel specification object from which the provisioner dictionary is derived.

Returns
-------
dict
    The provisioner portion of the kernel_spec.  If one does not exist, it will contain
    the default information.  If no `config` sub-dictionary exists, an empty `config`
    dictionary will be added.

### Function: get_provisioner_entries(self)

**Description:** Returns a dictionary of provisioner entries.

The key is the provisioner name for its entry point.  The value is the colon-separated
string of the entry point's module name and object name.

### Function: _get_all_provisioners()

**Description:** Wrapper around entry_points (to fetch the set of provisioners) - primarily to facilitate testing.

### Function: _get_provisioner(self, name)

**Description:** Wrapper around entry_points (to fetch a single provisioner) - primarily to facilitate testing.
