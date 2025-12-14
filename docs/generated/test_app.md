## AI Summary

A file named test_app.py.


### Function: _create_template_dir()

### Function: _create_static_dir()

### Function: _create_schemas_dir()

**Description:** Create a temporary directory for schemas.

### Function: _create_user_settings_dir()

**Description:** Create a temporary directory for workspaces.

### Function: _create_workspaces_dir()

**Description:** Create a temporary directory for workspaces.

## Class: TestEnv

**Description:** Set Jupyter path variables to a temporary directory

Useful as a context manager or with explicit start/stop

## Class: ProcessTestApp

**Description:** A process app for running tests, includes a mock contents directory.

## Class: RootedServerApp

### Function: start(self)

### Function: stop(self)

### Function: __enter__(self)

### Function: __exit__(self)

### Function: initialize_templates(self)

### Function: initialize_settings(self)

### Function: _install_kernel(self, kernel_name, kernel_spec)

**Description:** Install a kernel spec to the data directory.

Parameters
----------
kernel_name: str
    Name of the kernel.
kernel_spec: dict
    The kernel spec for the kernel

### Function: _install_default_kernels(self)

### Function: _process_finished(self, future)

### Function: _default_root_dir(self)

**Description:** Create a temporary directory with some file structure.
