## AI Summary

A file named low_level.py.


## Class: Type

**Description:** Enum of Argon2 variants.

Please see :doc:`parameters` on how to pick one.

### Function: hash_secret(secret, salt, time_cost, memory_cost, parallelism, hash_len, type, version)

**Description:** Hash *secret* and return an **encoded** hash.

An encoded hash can be directly passed into :func:`verify_secret` as it
contains all parameters and the salt.

Args:
    secret: Secret to hash.

    salt: A salt_. Should be random and different for each secret.

    type: Which Argon2 variant to use.

    version: Which Argon2 version to use.

For an explanation of the Argon2 parameters see
:class:`argon2.PasswordHasher`.

Returns:
    An encoded Argon2 hash.

Raises:
    argon2.exceptions.HashingError: If hashing fails.

.. versionadded:: 16.0.0

.. _salt: https://en.wikipedia.org/wiki/Salt_(cryptography)

### Function: hash_secret_raw(secret, salt, time_cost, memory_cost, parallelism, hash_len, type, version)

**Description:** Hash *password* and return a **raw** hash.

This function takes the same parameters as :func:`hash_secret`.

.. versionadded:: 16.0.0

### Function: verify_secret(hash, secret, type)

**Description:** Verify whether *secret* is correct for *hash* of *type*.

Args:
    hash:
        An encoded Argon2 hash as returned by :func:`hash_secret`.

    secret:
        The secret to verify whether it matches the one in *hash*.

    type: Type for *hash*.

Raises:
    argon2.exceptions.VerifyMismatchError:
        If verification fails because *hash* is not valid for *secret* of
        *type*.

    argon2.exceptions.VerificationError:
        If verification fails for other reasons.

Returns:
    ``True`` on success, raise :exc:`~argon2.exceptions.VerificationError`
    otherwise.

.. versionadded:: 16.0.0
.. versionchanged:: 16.1.0
    Raise :exc:`~argon2.exceptions.VerifyMismatchError` on mismatches
    instead of its more generic superclass.

### Function: core(context, type)

**Description:** Direct binding to the ``argon2_ctx`` function.

.. warning::
    This is a strictly advanced function working on raw C data structures.
    Both Argon2's and *argon2-cffi*'s higher-level bindings do a lot of
    sanity checks and housekeeping work that *you* are now responsible for
    (e.g. clearing buffers). The structure of the *context* object can,
    has, and will change with *any* release!

    Use at your own peril; *argon2-cffi* does *not* use this binding
    itself.

Args:
    context:
        A CFFI Argon2 context object (i.e. an ``struct Argon2_Context`` /
        ``argon2_context``).

    type:
        Which Argon2 variant to use.  You can use the ``value`` field of
        :class:`Type`'s fields.

Returns:
    An Argon2 error code.  Can be transformed into a string using
    :func:`error_to_str`.

.. versionadded:: 16.0.0

### Function: error_to_str(error)

**Description:** Convert an Argon2 error code into a native string.

Args:
    error: An Argon2 error code as returned by :func:`core`.

Returns:
    A human-readable string describing the error.

.. versionadded:: 16.0.0
