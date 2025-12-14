## AI Summary

A file named gateway_client.py.


## Class: GatewayTokenRenewerMeta

**Description:** The metaclass necessary for proper ABC behavior in a Configurable.

## Class: GatewayTokenRenewerBase

**Description:** Abstract base class for refreshing tokens used between this server and a Gateway
server.  Implementations requiring additional configuration can extend their class
with appropriate configuration values or convey those values via appropriate
environment variables relative to the implementation.

## Class: NoOpTokenRenewer

**Description:** NoOpTokenRenewer is the default value to the GatewayClient trait
`gateway_token_renewer` and merely returns the provided token.

## Class: GatewayClient

**Description:** This class manages the configuration.  It's its own singleton class so
that we can share these values across all objects.  It also contains some
options.
helper methods to build request arguments out of the various config

## Class: RetryableHTTPClient

**Description:** Inspired by urllib.util.Retry (https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html),
this class is initialized with desired retry characteristics, uses a recursive method `fetch()` against an instance
of `AsyncHTTPClient` which tracks the current retry count across applicable request retries.

### Function: get_token(self, auth_header_key, auth_scheme, auth_token)

**Description:** Given the current authorization header key, scheme, and token, this method returns
a (potentially renewed) token for use against the Gateway server.

### Function: get_token(self, auth_header_key, auth_scheme, auth_token)

**Description:** This implementation simply returns the current authorization token.

### Function: _default_event_logger(self)

### Function: emit(self, data)

**Description:** Emit event using the core event schema from Jupyter Server's Gateway Client.

### Function: _url_default(self)

### Function: _url_validate(self, proposal)

### Function: _ws_url_default(self)

### Function: _ws_url_validate(self, proposal)

### Function: _kernels_endpoint_default(self)

### Function: _kernelspecs_endpoint_default(self)

### Function: _kernelspecs_resource_endpoint_default(self)

### Function: _connect_timeout_default(self)

### Function: _request_timeout_default(self)

### Function: _client_key_default(self)

### Function: _client_cert_default(self)

### Function: _ca_certs_default(self)

### Function: _http_user_default(self)

### Function: _http_pwd_default(self)

### Function: _headers_default(self)

### Function: _auth_header_key_default(self)

### Function: _auth_token_default(self)

### Function: _auth_scheme_default(self)

### Function: _validate_cert_default(self)

### Function: _allowed_envs_default(self)

### Function: _gateway_retry_interval_default(self)

### Function: _gateway_retry_interval_max_default(self)

### Function: _gateway_retry_max_default(self)

### Function: _gateway_token_renewer_class_default(self)

### Function: _launch_timeout_pad_default(self)

### Function: _accept_cookies_default(self)

### Function: _deprecated_trait(self, change)

**Description:** observer for deprecated traits

### Function: gateway_enabled(self)

### Function: __init__(self)

**Description:** Initialize a gateway client.

### Function: init_connection_args(self)

**Description:** Initialize arguments used on every request.  Since these are primarily static values,
we'll perform this operation once.

### Function: load_connection_args(self)

**Description:** Merges the static args relative to the connection, with the given keyword arguments.  If static
args have yet to be initialized, we'll do that here.

### Function: update_cookies(self, cookie)

**Description:** Update cookies from existing requests for load balancers

### Function: _clear_expired_cookies(self)

**Description:** Clear expired cookies.

### Function: _update_cookie_header(self, connection_args)

**Description:** Update a cookie header.

### Function: __init__(self)

**Description:** Initialize the retryable http client.
