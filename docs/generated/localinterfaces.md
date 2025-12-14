## AI Summary

A file named localinterfaces.py.


### Function: _uniq_stable(elems)

**Description:** uniq_stable(elems) -> list

Return from an iterable, a list of all the unique elements in the input,
maintaining the order in which they first appear.

### Function: _get_output(cmd)

**Description:** Get output of a command, raising IOError if it fails

### Function: _only_once(f)

**Description:** decorator to only run a function once

### Function: _requires_ips(f)

**Description:** decorator to ensure load_ips has been run before f

## Class: NoIPAddresses

### Function: _populate_from_list(addrs)

**Description:** populate local and public IPs from flat list of all IPs

### Function: _load_ips_ifconfig()

**Description:** load ip addresses from `ifconfig` output (posix)

### Function: _load_ips_ip()

**Description:** load ip addresses from `ip addr` output (Linux)

### Function: _load_ips_ipconfig()

**Description:** load ip addresses from `ipconfig` output (Windows)

### Function: _load_ips_psutil()

**Description:** load ip addresses with netifaces

### Function: _load_ips_netifaces()

**Description:** load ip addresses with netifaces

### Function: _load_ips_gethostbyname()

**Description:** load ip addresses with socket.gethostbyname_ex

This can be slow.

### Function: _load_ips_dumb()

**Description:** Fallback in case of unexpected failure

### Function: _load_ips(suppress_exceptions)

**Description:** load the IPs that point to this machine

This function will only ever be called once.

If will use psutil to do it quickly if available.
If not, it will use netifaces to do it quickly if available.
Then it will fallback on parsing the output of ifconfig / ip addr / ipconfig, as appropriate.
Finally, it will fallback on socket.gethostbyname_ex, which can be slow.

### Function: local_ips()

**Description:** return the IP addresses that point to this machine

### Function: public_ips()

**Description:** return the IP addresses for this machine that are visible to other machines

### Function: localhost()

**Description:** return ip for localhost (almost always 127.0.0.1)

### Function: is_local_ip(ip)

**Description:** does `ip` point to this machine?

### Function: is_public_ip(ip)

**Description:** is `ip` a publicly visible address?

### Function: wrapped()

### Function: ips_loaded()
