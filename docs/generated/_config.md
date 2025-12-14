## AI Summary

A file named _config.py.


## Class: UnsetType

### Function: create_ssl_context(verify, cert, trust_env)

## Class: Timeout

**Description:** Timeout configuration.

**Usage**:

Timeout(None)               # No timeouts.
Timeout(5.0)                # 5s timeout on all operations.
Timeout(None, connect=5.0)  # 5s timeout on connect, no other timeouts.
Timeout(5.0, connect=10.0)  # 10s timeout on connect. 5s timeout elsewhere.
Timeout(5.0, pool=None)     # No timeout on acquiring connection from pool.
                            # 5s timeout elsewhere.

## Class: Limits

**Description:** Configuration for limits to various client behaviors.

**Parameters:**

* **max_connections** - The maximum number of concurrent connections that may be
        established.
* **max_keepalive_connections** - Allow the connection pool to maintain
        keep-alive connections below this point. Should be less than or equal
        to `max_connections`.
* **keepalive_expiry** - Time limit on idle keep-alive connections in seconds.

## Class: Proxy

### Function: __init__(self, timeout)

### Function: as_dict(self)

### Function: __eq__(self, other)

### Function: __repr__(self)

### Function: __init__(self)

### Function: __eq__(self, other)

### Function: __repr__(self)

### Function: __init__(self, url)

### Function: raw_auth(self)

### Function: __repr__(self)
