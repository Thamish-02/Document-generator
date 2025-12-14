## AI Summary

A file named tunnel.py.


### Function: select_random_ports(n)

**Description:** Select and return n random ports that are available.

### Function: try_passwordless_ssh(server, keyfile, paramiko)

**Description:** Attempt to make an ssh connection without a password.
This is mainly used for requiring password input only once
when many tunnels may be connected to the same server.

If paramiko is None, the default for the platform is chosen.

### Function: _try_passwordless_openssh(server, keyfile)

**Description:** Try passwordless login with shell ssh command.

### Function: _try_passwordless_paramiko(server, keyfile)

**Description:** Try passwordless login with paramiko.

### Function: tunnel_connection(socket, addr, server, keyfile, password, paramiko, timeout)

**Description:** Connect a socket to an address via an ssh tunnel.

This is a wrapper for socket.connect(addr), when addr is not accessible
from the local machine.  It simply creates an ssh tunnel using the remaining args,
and calls socket.connect('tcp://localhost:lport') where lport is the randomly
selected local port of the tunnel.

### Function: open_tunnel(addr, server, keyfile, password, paramiko, timeout)

**Description:** Open a tunneled connection from a 0MQ url.

For use inside tunnel_connection.

Returns
-------

(url, tunnel) : (str, object)
    The 0MQ url that has been forwarded, and the tunnel object

### Function: openssh_tunnel(lport, rport, server, remoteip, keyfile, password, timeout)

**Description:** Create an ssh tunnel using command-line ssh that connects port lport
on this machine to localhost:rport on server.  The tunnel
will automatically close when not in use, remaining open
for a minimum of timeout seconds for an initial connection.

This creates a tunnel redirecting `localhost:lport` to `remoteip:rport`,
as seen from `server`.

keyfile and password may be specified, but ssh config is checked for defaults.

Parameters
----------

lport : int
    local port for connecting to the tunnel from this machine.
rport : int
    port on the remote machine to connect to.
server : str
    The ssh server to connect to. The full ssh server string will be parsed.
    user@server:port
remoteip : str [Default: 127.0.0.1]
    The remote ip, specifying the destination of the tunnel.
    Default is localhost, which means that the tunnel would redirect
    localhost:lport on this machine to localhost:rport on the *server*.

keyfile : str; path to public key file
    This specifies a key to be used in ssh login, default None.
    Regular default ssh keys will be used without specifying this argument.
password : str;
    Your ssh password to the ssh server. Note that if this is left None,
    you will be prompted for it if passwordless key based login is unavailable.
timeout : int [default: 60]
    The time (in seconds) after which no activity will result in the tunnel
    closing.  This prevents orphaned tunnels from running forever.

### Function: _stop_tunnel(cmd)

### Function: _split_server(server)

### Function: paramiko_tunnel(lport, rport, server, remoteip, keyfile, password, timeout)

**Description:** launch a tunner with paramiko in a subprocess. This should only be used
when shell ssh is unavailable (e.g. Windows).

This creates a tunnel redirecting `localhost:lport` to `remoteip:rport`,
as seen from `server`.

If you are familiar with ssh tunnels, this creates the tunnel:

ssh server -L localhost:lport:remoteip:rport

keyfile and password may be specified, but ssh config is checked for defaults.


Parameters
----------

lport : int
    local port for connecting to the tunnel from this machine.
rport : int
    port on the remote machine to connect to.
server : str
    The ssh server to connect to. The full ssh server string will be parsed.
    user@server:port
remoteip : str [Default: 127.0.0.1]
    The remote ip, specifying the destination of the tunnel.
    Default is localhost, which means that the tunnel would redirect
    localhost:lport on this machine to localhost:rport on the *server*.

keyfile : str; path to public key file
    This specifies a key to be used in ssh login, default None.
    Regular default ssh keys will be used without specifying this argument.
password : str;
    Your ssh password to the ssh server. Note that if this is left None,
    you will be prompted for it if passwordless key based login is unavailable.
timeout : int [default: 60]
    The time (in seconds) after which no activity will result in the tunnel
    closing.  This prevents orphaned tunnels from running forever.

### Function: _paramiko_tunnel(lport, rport, server, remoteip, keyfile, password)

**Description:** Function for actually starting a paramiko tunnel, to be passed
to multiprocessing.Process(target=this), and not called directly.

## Class: SSHException
