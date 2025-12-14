## AI Summary

A file named login.py.


## Class: LoginFormHandler

**Description:** The basic tornado login handler

accepts login form, passed to IdentityProvider.process_login_form.

## Class: LegacyLoginHandler

**Description:** Legacy LoginHandler, implementing most custom auth configuration.

Deprecated in jupyter-server 2.0.
Login configuration has moved to IdentityProvider.

### Function: _render(self, message)

**Description:** Render the login form.

### Function: _redirect_safe(self, url, default)

**Description:** Redirect if url is on our PATH

Full-domain redirects are allowed if they pass our CORS origin checks.

Otherwise use default (self.base_url if unspecified).

### Function: get(self)

**Description:** Get the login form.

### Function: post(self)

**Description:** Post a login.

### Function: hashed_password(self)

### Function: passwd_check(self, a, b)

**Description:** Check a passwd.

### Function: post(self)

**Description:** Post a login form.

### Function: set_login_cookie(cls, handler, user_id)

**Description:** Call this on handlers to set the login cookie for success

### Function: get_token(cls, handler)

**Description:** Get the user token from a request

Default:

- in URL parameters: ?token=<token>
- in header: Authorization: token <token>

### Function: should_check_origin(cls, handler)

**Description:** DEPRECATED in 2.0, use IdentityProvider API

### Function: is_token_authenticated(cls, handler)

**Description:** DEPRECATED in 2.0, use IdentityProvider API

### Function: get_user(cls, handler)

**Description:** DEPRECATED in 2.0, use IdentityProvider API

### Function: get_user_cookie(cls, handler)

**Description:** DEPRECATED in 2.0, use IdentityProvider API

### Function: get_user_token(cls, handler)

**Description:** DEPRECATED in 2.0, use IdentityProvider API

### Function: validate_security(cls, app, ssl_options)

**Description:** DEPRECATED in 2.0, use IdentityProvider API

### Function: password_from_settings(cls, settings)

**Description:** DEPRECATED in 2.0, use IdentityProvider API

### Function: get_login_available(cls, settings)

**Description:** DEPRECATED in 2.0, use IdentityProvider API
