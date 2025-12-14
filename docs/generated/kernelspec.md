## AI Summary

A file named kernelspec.py.


## Class: KernelSpec

**Description:** A kernel spec model object.

### Function: _is_valid_kernel_name(name)

**Description:** Check that a kernel name is valid.

### Function: _is_kernel_dir(path)

**Description:** Is ``path`` a kernel directory?

### Function: _list_kernels_in(dir)

**Description:** Return a mapping of kernel names to resource directories from dir.

If dir is None or does not exist, returns an empty dict.

## Class: NoSuchKernel

**Description:** An error raised when there is no kernel of a give name.

## Class: KernelSpecManager

**Description:** A manager for kernel specs.

### Function: find_kernel_specs()

**Description:** Returns a dict mapping kernel names to resource directories.

### Function: get_kernel_spec(kernel_name)

**Description:** Returns a :class:`KernelSpec` instance for the given kernel_name.

Raises KeyError if the given kernel name is not found.

### Function: install_kernel_spec(source_dir, kernel_name, user, replace, prefix)

**Description:** Install a kernel spec in a given directory.

### Function: install_native_kernel_spec(user)

**Description:** Install the native kernel spec.

### Function: from_resource_dir(cls, resource_dir)

**Description:** Create a KernelSpec object by reading kernel.json

Pass the path to the *directory* containing kernel.json.

### Function: to_dict(self)

**Description:** Convert the kernel spec to a dict.

### Function: to_json(self)

**Description:** Serialise this kernelspec to a JSON object.

Returns a string.

### Function: __init__(self, name)

**Description:** Initialize the error.

### Function: __str__(self)

### Function: _data_dir_default(self)

### Function: _user_kernel_dir_default(self)

### Function: _deprecated_trait(self, change)

**Description:** observer for deprecated traits

### Function: _kernel_dirs_default(self)

### Function: find_kernel_specs(self)

**Description:** Returns a dict mapping kernel names to resource directories.

### Function: _get_kernel_spec_by_name(self, kernel_name, resource_dir)

**Description:** Returns a :class:`KernelSpec` instance for a given kernel_name
and resource_dir.

### Function: _find_spec_directory(self, kernel_name)

**Description:** Find the resource directory of a named kernel spec

### Function: get_kernel_spec(self, kernel_name)

**Description:** Returns a :class:`KernelSpec` instance for the given kernel_name.

Raises :exc:`NoSuchKernel` if the given kernel name is not found.

### Function: get_all_specs(self)

**Description:** Returns a dict mapping kernel names to kernelspecs.

Returns a dict of the form::

    {
      'kernel_name': {
        'resource_dir': '/path/to/kernel_name',
        'spec': {"the spec itself": ...}
      },
      ...
    }

### Function: remove_kernel_spec(self, name)

**Description:** Remove a kernel spec directory by name.

Returns the path that was deleted.

### Function: _get_destination_dir(self, kernel_name, user, prefix)

### Function: install_kernel_spec(self, source_dir, kernel_name, user, replace, prefix)

**Description:** Install a kernel spec by copying its directory.

If ``kernel_name`` is not given, the basename of ``source_dir`` will
be used.

If ``user`` is False, it will attempt to install into the systemwide
kernel registry. If the process does not have appropriate permissions,
an :exc:`OSError` will be raised.

If ``prefix`` is given, the kernelspec will be installed to
PREFIX/share/jupyter/kernels/KERNEL_NAME. This can be sys.prefix
for installation inside virtual or conda envs.

### Function: install_native_kernel_spec(self, user)

**Description:** DEPRECATED: Use ipykernel.kernelspec.install
