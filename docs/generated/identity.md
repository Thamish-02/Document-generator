## AI Summary

A file named identity.py.


## Class: User

**Description:** Object representing a User

This or a subclass should be returned from IdentityProvider.get_user

### Function: _backward_compat_user(got_user)

**Description:** Backward-compatibility for LoginHandler.get_user

Prior to 2.0, LoginHandler.get_user could return anything truthy.

Typically, this was either a simple string username,
or a simple dict.

Make some effort to allow common patterns to keep working.

## Class: IdentityProvider

**Description:** Interface for providing identity management and authentication.

Two principle methods:

- :meth:`~jupyter_server.auth.IdentityProvider.get_user` returns a :class:`~.User` object
  for successful authentication, or None for no-identity-found.
- :meth:`~jupyter_server.auth.IdentityProvider.identity_model` turns a :class:`~jupyter_server.auth.User` into a JSONable dict.
  The default is to use :py:meth:`dataclasses.asdict`,
  and usually shouldn't need override.

Additional methods can customize authentication.

.. versionadded:: 2.0

## Class: PasswordIdentityProvider

**Description:** A password identity provider.

## Class: LegacyIdentityProvider

**Description:** Legacy IdentityProvider for use with custom LoginHandlers

Login configuration has moved from LoginHandler to IdentityProvider
in Jupyter Server 2.0.

### Function: __post_init__(self)

### Function: fill_defaults(self)

**Description:** Fill out default fields in the identity model

- Ensures all values are defined
- Fills out derivative values for name fields fields
- Fills out null values for optional fields

### Function: _token_default(self)

### Function: _validate_updatable_fields(self, proposal)

**Description:** Validate that all fields in updatable_fields are valid.

### Function: get_user(self, handler)

**Description:** Get the authenticated user for a request

Must return a :class:`jupyter_server.auth.User`,
though it may be a subclass.

Return None if the request is not authenticated.

_may_ be a coroutine

### Function: update_user(self, handler, user_data)

**Description:** Update user information and persist the user model.

### Function: check_update(self, user_data)

**Description:** Raises if some fields to update are not updatable.

### Function: update_user_model(self, current_user, user_data)

**Description:** Update user information.

### Function: persist_user_model(self, handler)

**Description:** Persist the user model (i.e. a cookie).

### Function: identity_model(self, user)

**Description:** Return a User as an Identity model

### Function: get_handlers(self)

**Description:** Return list of additional handlers for this identity provider

For example, an OAuth callback handler.

### Function: user_to_cookie(self, user)

**Description:** Serialize a user to a string for storage in a cookie

If overriding in a subclass, make sure to define user_from_cookie as well.

Default is just the user's username.

### Function: user_from_cookie(self, cookie_value)

**Description:** Inverse of user_to_cookie

### Function: get_cookie_name(self, handler)

**Description:** Return the login cookie name

Uses IdentityProvider.cookie_name, if defined.
Default is to generate a string taking host into account to avoid
collisions for multiple servers on one hostname with different ports.

### Function: set_login_cookie(self, handler, user)

**Description:** Call this on handlers to set the login cookie for success

### Function: _force_clear_cookie(self, handler, name, path, domain)

**Description:** Deletes the cookie with the given name.

Tornado's cookie handling currently (Jan 2018) stores cookies in a dict
keyed by name, so it can only modify one cookie with a given name per
response. The browser can store multiple cookies with the same name
but different domains and/or paths. This method lets us clear multiple
cookies with the same name.

Due to limitations of the cookie protocol, you must pass the same
path and domain to clear a cookie as were used when that cookie
was set (but there is no way to find out on the server side
which values were used for a given cookie).

### Function: clear_login_cookie(self, handler)

**Description:** Clear the login cookie, effectively logging out the session.

### Function: get_user_cookie(self, handler)

**Description:** Get user from a cookie

Calls user_from_cookie to deserialize cookie value

### Function: get_token(self, handler)

**Description:** Get the user token from a request

Default:

- in URL parameters: ?token=<token>
- in header: Authorization: token <token>

### Function: generate_anonymous_user(self, handler)

**Description:** Generate a random anonymous user.

For use when a single shared token is used,
but does not identify a user.

### Function: should_check_origin(self, handler)

**Description:** Should the Handler check for CORS origin validation?

Origin check should be skipped for token-authenticated requests.

Returns:
- True, if Handler must check for valid CORS origin.
- False, if Handler should skip origin check since requests are token-authenticated.

### Function: is_token_authenticated(self, handler)

**Description:** Returns True if handler has been token authenticated. Otherwise, False.

Login with a token is used to signal certain things, such as:

- permit access to REST API
- xsrf protection
- skip origin-checks for scripts

### Function: validate_security(self, app, ssl_options)

**Description:** Check the application's security.

Show messages, or abort if necessary, based on the security configuration.

### Function: process_login_form(self, handler)

**Description:** Process login form data

Return authenticated User if successful, None if not.

### Function: auth_enabled(self)

**Description:** Is authentication enabled?

Should always be True, but may be False in rare, insecure cases
where requests with no auth are allowed.

Previously: LoginHandler.get_login_available

### Function: login_available(self)

**Description:** Whether a LoginHandler is needed - and therefore whether the login page should be displayed.

### Function: logout_available(self)

**Description:** Whether a LogoutHandler is needed.

### Function: _need_token_default(self)

### Function: _default_updatable_fields(self)

### Function: login_available(self)

**Description:** Whether a LoginHandler is needed - and therefore whether the login page should be displayed.

### Function: auth_enabled(self)

**Description:** Return whether any auth is enabled

### Function: update_user_model(self, current_user, user_data)

**Description:** Update user information.

### Function: persist_user_model(self, handler)

**Description:** Persist the user model to a cookie.

### Function: passwd_check(self, password)

**Description:** Check password against our stored hashed password

### Function: process_login_form(self, handler)

**Description:** Process login form data

Return authenticated User if successful, None if not.

### Function: validate_security(self, app, ssl_options)

**Description:** Handle security validation.

### Function: _default_settings(self)

### Function: _default_login_handler_class(self)

### Function: auth_enabled(self)

### Function: get_user(self, handler)

**Description:** Get the user.

### Function: login_available(self)

### Function: should_check_origin(self, handler)

**Description:** Whether we should check origin.

### Function: is_token_authenticated(self, handler)

**Description:** Whether we are token authenticated.

### Function: validate_security(self, app, ssl_options)

**Description:** Validate security.
