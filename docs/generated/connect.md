## AI Summary

A file named connect.py.


### Function: write_connection_file(fname, shell_port, iopub_port, stdin_port, hb_port, control_port, ip, key, transport, signature_scheme, kernel_name)

**Description:** Generates a JSON config file, including the selection of random ports.

Parameters
----------

fname : unicode
    The path to the file to write

shell_port : int, optional
    The port to use for ROUTER (shell) channel.

iopub_port : int, optional
    The port to use for the SUB channel.

stdin_port : int, optional
    The port to use for the ROUTER (raw input) channel.

control_port : int, optional
    The port to use for the ROUTER (control) channel.

hb_port : int, optional
    The port to use for the heartbeat REP channel.

ip  : str, optional
    The ip address the kernel will bind to.

key : str, optional
    The Session key used for message authentication.

signature_scheme : str, optional
    The scheme used for message authentication.
    This has the form 'digest-hash', where 'digest'
    is the scheme used for digests, and 'hash' is the name of the hash function
    used by the digest scheme.
    Currently, 'hmac' is the only supported digest scheme,
    and 'sha256' is the default hash function.

kernel_name : str, optional
    The name of the kernel currently connected to.

### Function: find_connection_file(filename, path, profile)

**Description:** find a connection file, and return its absolute path.

The current working directory and optional search path
will be searched for the file if it is not given by absolute path.

If the argument does not match an existing file, it will be interpreted as a
fileglob, and the matching file in the profile's security dir with
the latest access time will be used.

Parameters
----------
filename : str
    The connection file or fileglob to search for.
path : str or list of strs[optional]
    Paths in which to search for connection files.

Returns
-------
str : The absolute path of the connection file.

### Function: tunnel_to_kernel(connection_info, sshserver, sshkey)

**Description:** tunnel connections to a kernel via ssh

This will open five SSH tunnels from localhost on this machine to the
ports associated with the kernel.  They can be either direct
localhost-localhost tunnels, or if an intermediate server is necessary,
the kernel must be listening on a public IP.

Parameters
----------
connection_info : dict or str (path)
    Either a connection dict, or the path to a JSON connection file
sshserver : str
    The ssh sever to use to tunnel to the kernel. Can be a full
    `user@server:port` string. ssh config aliases are respected.
sshkey : str [optional]
    Path to file containing ssh key to use for authentication.
    Only necessary if your ssh config does not already associate
    a keyfile with the host.

Returns
-------

(shell, iopub, stdin, hb, control) : ints
    The five ports on localhost that have been forwarded to the kernel.

## Class: ConnectionFileMixin

**Description:** Mixin for configurable classes that work with connection files

## Class: LocalPortCache

**Description:** Used to keep track of local ports in order to prevent race conditions that
can occur between port acquisition and usage by the kernel.  All locally-
provisioned kernels should use this mechanism to limit the possibility of
race conditions.  Note that this does not preclude other applications from
acquiring a cached but unused port, thereby re-introducing the issue this
class is attempting to resolve (minimize).
See: https://github.com/jupyter/jupyter_client/issues/487

### Function: _data_dir_default(self)

### Function: _ip_default(self)

### Function: _ip_changed(self, change)

### Function: ports(self)

### Function: _session_default(self)

### Function: get_connection_info(self, session)

**Description:** Return the connection info as a dict

Parameters
----------
session : bool [default: False]
    If True, return our session object will be included in the connection info.
    If False (default), the configuration parameters of our session object will be included,
    rather than the session object itself.

Returns
-------
connect_info : dict
    dictionary of connection information.

### Function: blocking_client(self)

**Description:** Make a blocking client connected to my kernel

### Function: cleanup_connection_file(self)

**Description:** Cleanup connection file *if we wrote it*

Will not raise if the connection file was already removed somehow.

### Function: cleanup_ipc_files(self)

**Description:** Cleanup ipc files if we wrote them.

### Function: _record_random_port_names(self)

**Description:** Records which of the ports are randomly assigned.

Records on first invocation, if the transport is tcp.
Does nothing on later invocations.

### Function: cleanup_random_ports(self)

**Description:** Forgets randomly assigned port numbers and cleans up the connection file.

Does nothing if no port numbers have been randomly assigned.
In particular, does nothing unless the transport is tcp.

### Function: write_connection_file(self)

**Description:** Write connection info to JSON dict in self.connection_file.

### Function: load_connection_file(self, connection_file)

**Description:** Load connection info from JSON dict in self.connection_file.

Parameters
----------
connection_file: unicode, optional
    Path to connection file to load.
    If unspecified, use self.connection_file

### Function: load_connection_info(self, info)

**Description:** Load connection info from a dict containing connection info.

Typically this data comes from a connection file
and is called by load_connection_file.

Parameters
----------
info: dict
    Dictionary containing connection_info.
    See the connection_file spec for details.

### Function: _reconcile_connection_info(self, info)

**Description:** Reconciles the connection information returned from the Provisioner.

Because some provisioners (like derivations of LocalProvisioner) may have already
written the connection file, this method needs to ensure that, if the connection
file exists, its contents match that of what was returned by the provisioner.  If
the file does exist and its contents do not match, the file will be replaced with
the provisioner information (which is considered the truth).

If the file does not exist, the connection information in 'info' is loaded into the
KernelManager and written to the file.

### Function: _equal_connections(conn1, conn2)

**Description:** Compares pertinent keys of connection info data. Returns True if equivalent, False otherwise.

### Function: _make_url(self, channel)

**Description:** Make a ZeroMQ URL for a given channel.

### Function: _create_connected_socket(self, channel, identity)

**Description:** Create a zmq Socket and connect it to the kernel.

### Function: connect_iopub(self, identity)

**Description:** return zmq Socket connected to the IOPub channel

### Function: connect_shell(self, identity)

**Description:** return zmq Socket connected to the Shell channel

### Function: connect_stdin(self, identity)

**Description:** return zmq Socket connected to the StdIn channel

### Function: connect_hb(self, identity)

**Description:** return zmq Socket connected to the Heartbeat channel

### Function: connect_control(self, identity)

**Description:** return zmq Socket connected to the Control channel

### Function: __init__(self)

### Function: find_available_port(self, ip)

### Function: return_port(self, port)
