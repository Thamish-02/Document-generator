## AI Summary

A file named _legacy.py.


### Function: hash_password(password, salt, time_cost, memory_cost, parallelism, hash_len, type)

**Description:** Legacy alias for :func:`argon2.low_level.hash_secret` with default
parameters.

.. deprecated:: 16.0.0
    Use :class:`argon2.PasswordHasher` for passwords.

### Function: hash_password_raw(password, salt, time_cost, memory_cost, parallelism, hash_len, type)

**Description:** Legacy alias for :func:`argon2.low_level.hash_secret_raw` with default
parameters.

.. deprecated:: 16.0.0
    Use :class:`argon2.PasswordHasher` for passwords.

### Function: verify_password(hash, password, type)

**Description:** Legacy alias for :func:`argon2.low_level.verify_secret` with default
parameters.

.. deprecated:: 16.0.0
    Use :class:`argon2.PasswordHasher` for passwords.
