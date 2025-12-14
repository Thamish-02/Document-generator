## AI Summary

A file named managers.py.


## Class: GatewayMappingKernelManager

**Description:** Kernel manager that supports remote kernels hosted by Jupyter Kernel or Enterprise Gateway.

## Class: GatewayKernelSpecManager

**Description:** A gateway kernel spec manager.

## Class: GatewaySessionManager

**Description:** A gateway session manager.

## Class: GatewayKernelManager

**Description:** Manages a single kernel remotely via a Gateway Server.

## Class: ChannelQueue

**Description:** A queue for a named channel.

## Class: HBChannelQueue

**Description:** A queue for the heartbeat channel.

## Class: GatewayKernelClient

**Description:** Communicates with a single kernel indirectly via a websocket to a gateway server.

There are five channels associated with each kernel:

* shell: for request/reply calls to the kernel.
* iopub: for the kernel to publish results to frontends.
* hb: for monitoring the kernel's heartbeat.
* stdin: for frontends to reply to raw_input calls in the kernel.
* control: for kernel management calls to the kernel.

The messages that can be sent on these channels are exposed as methods of the
client (KernelClient.execute, complete, history, etc.). These methods only
send the message, they don't wait for a reply. To get results, use e.g.
:meth:`get_shell_msg` to fetch messages from the shell channel.

### Function: _default_kernel_manager_class(self)

### Function: _default_shared_context(self)

### Function: __init__(self)

**Description:** Initialize a gateway mapping kernel manager.

### Function: remove_kernel(self, kernel_id)

**Description:** Complete override since we want to be more tolerant of missing keys

### Function: __init__(self)

**Description:** Initialize a gateway kernel spec manager.

### Function: _get_endpoint_for_user_filter(default_endpoint)

**Description:** Get the endpoint for a user filter.

### Function: _replace_path_kernelspec_resources(self, kernel_specs)

**Description:** Helper method that replaces any gateway base_url with the server's base_url
This enables clients to properly route through jupyter_server to a gateway
for kernel resources such as logo files

### Function: _get_kernelspecs_endpoint_url(self, kernel_name)

**Description:** Builds a url for the kernels endpoint
Parameters
----------
kernel_name : kernel name (optional)

### Function: _default_cache_ports(self)

### Function: __init__(self)

**Description:** Initialize the gateway kernel manager.

### Function: has_kernel(self)

**Description:** Has a kernel been started that we are managing.

### Function: client(self)

**Description:** Create a client configured to connect to our kernel

### Function: cleanup_resources(self, restart)

**Description:** Clean up resources when the kernel is shut down

### Function: __init__(self, channel_name, channel_socket, log)

**Description:** Initialize a channel queue.

### Function: send(self, msg)

**Description:** Send a message to the queue.

### Function: serialize_datetime(dt)

**Description:** Serialize a datetime object.

### Function: start(self)

**Description:** Start the queue.

### Function: stop(self)

**Description:** Stop the queue.

### Function: is_alive(self)

**Description:** Whether the queue is alive.

### Function: is_beating(self)

**Description:** Whether the channel is beating.

### Function: __init__(self, kernel_id)

**Description:** Initialize a gateway kernel client.

### Function: stop_channels(self)

**Description:** Stops all the running channels for this kernel.

For this class, we close the websocket connection and destroy the
channel-based queues.

### Function: shell_channel(self)

**Description:** Get the shell channel object for this kernel.

### Function: iopub_channel(self)

**Description:** Get the iopub channel object for this kernel.

### Function: stdin_channel(self)

**Description:** Get the stdin channel object for this kernel.

### Function: hb_channel(self)

**Description:** Get the hb channel object for this kernel.

### Function: control_channel(self)

**Description:** Get the control channel object for this kernel.

### Function: _route_responses(self)

**Description:** Reads responses from the websocket and routes each to the appropriate channel queue based
on the message's channel.  It does this for the duration of the class's lifetime until the
channels are stopped, at which time the socket is closed (unblocking the router) and
the thread terminates.  If shutdown happens to occur while processing a response (unlikely),
termination takes place via the loop control boolean.
