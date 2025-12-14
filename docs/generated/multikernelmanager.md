## AI Summary

A file named multikernelmanager.py.


## Class: DuplicateKernelError

### Function: kernel_method(f)

**Description:** decorator for proxying MKM.method(kernel_id) to individual KMs by ID

## Class: MultiKernelManager

**Description:** A class for managing multiple kernels.

## Class: AsyncMultiKernelManager

### Function: wrapped(self, kernel_id)

### Function: _kernel_manager_class_changed(self, change)

### Function: _kernel_manager_factory_default(self)

### Function: _create_kernel_manager_factory(self)

### Function: _starting_kernels(self)

**Description:** A shim for backwards compatibility.

### Function: _context_default(self)

### Function: __init__(self)

### Function: __del__(self)

**Description:** Handle garbage collection.  Destroy context if applicable.

### Function: list_kernel_ids(self)

**Description:** Return a list of the kernel ids of the active kernels.

### Function: __len__(self)

**Description:** Return the number of running kernels.

### Function: __contains__(self, kernel_id)

### Function: pre_start_kernel(self, kernel_name, kwargs)

### Function: update_env(self)

**Description:** Allow to update the environment of the given kernel.

Forward the update env request to the corresponding kernel.

.. version-added: 8.5

### Function: _using_pending_kernels(self)

**Description:** Returns a boolean; a clearer method for determining if
this multikernelmanager is using pending kernels or not

### Function: request_shutdown(self, kernel_id, restart)

**Description:** Ask a kernel to shut down by its kernel uuid

### Function: finish_shutdown(self, kernel_id, waittime, pollinterval)

**Description:** Wait for a kernel to finish shutting down, and kill it if it doesn't

### Function: cleanup_resources(self, kernel_id, restart)

**Description:** Clean up a kernel's resources

### Function: remove_kernel(self, kernel_id)

**Description:** remove a kernel from our mapping.

Mainly so that a kernel can be removed if it is already dead,
without having to call shutdown_kernel.

The kernel object is returned, or `None` if not found.

### Function: interrupt_kernel(self, kernel_id)

**Description:** Interrupt (SIGINT) the kernel by its uuid.

Parameters
==========
kernel_id : uuid
    The id of the kernel to interrupt.

### Function: signal_kernel(self, kernel_id, signum)

**Description:** Sends a signal to the kernel by its uuid.

Note that since only SIGTERM is supported on Windows, this function
is only useful on Unix systems.

Parameters
==========
kernel_id : uuid
    The id of the kernel to signal.
signum : int
    Signal number to send kernel.

### Function: is_alive(self, kernel_id)

**Description:** Is the kernel alive.

This calls KernelManager.is_alive() which calls Popen.poll on the
actual kernel subprocess.

Parameters
==========
kernel_id : uuid
    The id of the kernel.

### Function: _check_kernel_id(self, kernel_id)

**Description:** check that a kernel id is valid

### Function: get_kernel(self, kernel_id)

**Description:** Get the single KernelManager object for a kernel by its uuid.

Parameters
==========
kernel_id : uuid
    The id of the kernel.

### Function: add_restart_callback(self, kernel_id, callback, event)

**Description:** add a callback for the KernelRestarter

### Function: remove_restart_callback(self, kernel_id, callback, event)

**Description:** remove a callback for the KernelRestarter

### Function: get_connection_info(self, kernel_id)

**Description:** Return a dictionary of connection data for a kernel.

Parameters
==========
kernel_id : uuid
    The id of the kernel.

Returns
=======
connection_dict : dict
    A dict of the information needed to connect to a kernel.
    This includes the ip address and the integer port
    numbers of the different channels (stdin_port, iopub_port,
    shell_port, hb_port).

### Function: connect_iopub(self, kernel_id, identity)

**Description:** Return a zmq Socket connected to the iopub channel.

Parameters
==========
kernel_id : uuid
    The id of the kernel
identity : bytes (optional)
    The zmq identity of the socket

Returns
=======
stream : zmq Socket or ZMQStream

### Function: connect_shell(self, kernel_id, identity)

**Description:** Return a zmq Socket connected to the shell channel.

Parameters
==========
kernel_id : uuid
    The id of the kernel
identity : bytes (optional)
    The zmq identity of the socket

Returns
=======
stream : zmq Socket or ZMQStream

### Function: connect_control(self, kernel_id, identity)

**Description:** Return a zmq Socket connected to the control channel.

Parameters
==========
kernel_id : uuid
    The id of the kernel
identity : bytes (optional)
    The zmq identity of the socket

Returns
=======
stream : zmq Socket or ZMQStream

### Function: connect_stdin(self, kernel_id, identity)

**Description:** Return a zmq Socket connected to the stdin channel.

Parameters
==========
kernel_id : uuid
    The id of the kernel
identity : bytes (optional)
    The zmq identity of the socket

Returns
=======
stream : zmq Socket or ZMQStream

### Function: connect_hb(self, kernel_id, identity)

**Description:** Return a zmq Socket connected to the hb channel.

Parameters
==========
kernel_id : uuid
    The id of the kernel
identity : bytes (optional)
    The zmq identity of the socket

Returns
=======
stream : zmq Socket or ZMQStream

### Function: new_kernel_id(self)

**Description:** Returns the id to associate with the kernel for this request. Subclasses may override
this method to substitute other sources of kernel ids.
:param kwargs:
:return: string-ized version 4 uuid

### Function: _context_default(self)

### Function: create_kernel_manager()
