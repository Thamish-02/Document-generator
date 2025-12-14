## AI Summary

A file named test_auth.py.


### Function: test_auth_rest(route, a_server_url_and_token)

**Description:** Verify a REST route only provides access to an authenticated user.

### Function: test_auth_websocket(route, a_server_url_and_token)

**Description:** Verify a WebSocket does not provide access to an unauthenticated user.

### Function: a_server_url_and_token(tmp_path_factory)

**Description:** Start a temporary, isolated jupyter server.

### Function: get_unused_port()

**Description:** Get an unused port by trying to listen to any random port.

Probably could introduce race conditions if inside a tight loop.

### Function: verify_response(base_url, route, expect_code)

**Description:** Verify that a response returns the expected error.
