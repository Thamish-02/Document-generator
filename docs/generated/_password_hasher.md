## AI Summary

A file named _password_hasher.py.


### Function: _ensure_bytes(s, encoding)

**Description:** Ensure *s* is a bytes string.  Encode using *encoding* if it isn't.

## Class: PasswordHasher

**Description:** High level class to hash passwords with sensible defaults.

Uses Argon2\ **id** by default and uses a random salt_ for hashing. But it
can verify any type of Argon2 as long as the hash is correctly encoded.

The reason for this being a class is both for convenience to carry
parameters and to verify the parameters only *once*.  Any unnecessary
slowdown when hashing is a tangible advantage for a brute-force attacker.

Args:
    time_cost:
        Defines the amount of computation realized and therefore the
        execution time, given in number of iterations.

    memory_cost: Defines the memory usage, given in kibibytes_.

    parallelism:
        Defines the number of parallel threads (*changes* the resulting
        hash value).

    hash_len: Length of the hash in bytes.

    salt_len: Length of random salt to be generated for each password.

    encoding:
        The Argon2 C library expects bytes.  So if :meth:`hash` or
        :meth:`verify` are passed a ``str``, it will be encoded using this
        encoding.

    type:
        Argon2 type to use.  Only change for interoperability with legacy
        systems.

.. versionadded:: 16.0.0
.. versionchanged:: 18.2.0
   Switch from Argon2i to Argon2id based on the recommendation by the
   current RFC draft. See also :doc:`parameters`.
.. versionchanged:: 18.2.0
   Changed default *memory_cost* to 100 MiB and default *parallelism* to 8.
.. versionchanged:: 18.2.0 ``verify`` now will determine the type of hash.
.. versionchanged:: 18.3.0 The Argon2 type is configurable now.
.. versionadded:: 21.2.0 :meth:`from_parameters`
.. versionchanged:: 21.2.0
   Changed defaults to :data:`argon2.profiles.RFC_9106_LOW_MEMORY`.

.. _salt: https://en.wikipedia.org/wiki/Salt_(cryptography)
.. _kibibytes: https://en.wikipedia.org/wiki/Binary_prefix#kibi

### Function: __init__(self, time_cost, memory_cost, parallelism, hash_len, salt_len, encoding, type)

### Function: from_parameters(cls, params)

**Description:** Construct a `PasswordHasher` from *params*.

Returns:
    A `PasswordHasher` instance with the parameters from *params*.

.. versionadded:: 21.2.0

### Function: time_cost(self)

### Function: memory_cost(self)

### Function: parallelism(self)

### Function: hash_len(self)

### Function: salt_len(self)

### Function: type(self)

### Function: hash(self, password)

**Description:** Hash *password* and return an encoded hash.

Args:
    password: Password to hash.

    salt:
        If None, a random salt is securely created.

        .. danger::

            You should **not** pass a salt unless you really know what
            you are doing.

Raises:
    argon2.exceptions.HashingError: If hashing fails.

Returns:
    Hashed *password*.

.. versionadded:: 23.1.0 *salt* parameter

### Function: verify(self, hash, password)

**Description:** Verify that *password* matches *hash*.

.. warning::

    It is assumed that the caller is in full control of the hash.  No
    other parsing than the determination of the hash type is done by
    *argon2-cffi*.

Args:
    hash: An encoded hash as returned from :meth:`PasswordHasher.hash`.

    password: The password to verify.

Raises:
    argon2.exceptions.VerifyMismatchError:
        If verification fails because *hash* is not valid for
        *password*.

    argon2.exceptions.VerificationError:
        If verification fails for other reasons.

    argon2.exceptions.InvalidHashError:
        If *hash* is so clearly invalid, that it couldn't be passed to
        Argon2.

Returns:
    ``True`` on success, otherwise an exception is raised.

.. versionchanged:: 16.1.0
    Raise :exc:`~argon2.exceptions.VerifyMismatchError` on mismatches
    instead of its more generic superclass.
.. versionadded:: 18.2.0 Hash type agility.

### Function: check_needs_rehash(self, hash)

**Description:** Check whether *hash* was created using the instance's parameters.

Whenever your Argon2 parameters -- or *argon2-cffi*'s defaults! --
change, you should rehash your passwords at the next opportunity.  The
common approach is to do that whenever a user logs in, since that
should be the only time when you have access to the cleartext
password.

Therefore it's best practice to check -- and if necessary rehash --
passwords after each successful authentication.

Args:
    hash: An encoded Argon2 password hash.

Returns:
    Whether *hash* was created using the instance's parameters.

.. versionadded:: 18.2.0
.. versionchanged:: 24.1.0 Accepts bytes for *hash*.
