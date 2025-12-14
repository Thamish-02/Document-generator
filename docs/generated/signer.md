## AI Summary

A file named signer.py.


## Class: SigningAlgorithm

**Description:** Subclasses must implement :meth:`get_signature` to provide
signature generation functionality.

## Class: NoneAlgorithm

**Description:** Provides an algorithm that does not perform any signing and
returns an empty signature.

### Function: _lazy_sha1(string)

**Description:** Don't access ``hashlib.sha1`` until runtime. FIPS builds may not include
SHA-1, in which case the import and use as a default would fail before the
developer can configure something else.

## Class: HMACAlgorithm

**Description:** Provides signature generation using HMACs.

### Function: _make_keys_list(secret_key)

## Class: Signer

**Description:** A signer securely signs bytes, then unsigns them to verify that
the value hasn't been changed.

The secret key should be a random string of ``bytes`` and should not
be saved to code or version control. Different salts should be used
to distinguish signing in different contexts. See :doc:`/concepts`
for information about the security of the secret key and salt.

:param secret_key: The secret key to sign and verify with. Can be a
    list of keys, oldest to newest, to support key rotation.
:param salt: Extra key to combine with ``secret_key`` to distinguish
    signatures in different contexts.
:param sep: Separator between the signature and value.
:param key_derivation: How to derive the signing key from the secret
    key and salt. Possible values are ``concat``, ``django-concat``,
    or ``hmac``. Defaults to :attr:`default_key_derivation`, which
    defaults to ``django-concat``.
:param digest_method: Hash function to use when generating the HMAC
    signature. Defaults to :attr:`default_digest_method`, which
    defaults to :func:`hashlib.sha1`. Note that the security of the
    hash alone doesn't apply when used intermediately in HMAC.
:param algorithm: A :class:`SigningAlgorithm` instance to use
    instead of building a default :class:`HMACAlgorithm` with the
    ``digest_method``.

.. versionchanged:: 2.0
    Added support for key rotation by passing a list to
    ``secret_key``.

.. versionchanged:: 0.18
    ``algorithm`` was added as an argument to the class constructor.

.. versionchanged:: 0.14
    ``key_derivation`` and ``digest_method`` were added as arguments
    to the class constructor.

### Function: get_signature(self, key, value)

**Description:** Returns the signature for the given key and value.

### Function: verify_signature(self, key, value, sig)

**Description:** Verifies the given signature matches the expected
signature.

### Function: get_signature(self, key, value)

### Function: __init__(self, digest_method)

### Function: get_signature(self, key, value)

### Function: __init__(self, secret_key, salt, sep, key_derivation, digest_method, algorithm)

### Function: secret_key(self)

**Description:** The newest (last) entry in the :attr:`secret_keys` list. This
is for compatibility from before key rotation support was added.

### Function: derive_key(self, secret_key)

**Description:** This method is called to derive the key. The default key
derivation choices can be overridden here. Key derivation is not
intended to be used as a security method to make a complex key
out of a short password. Instead you should use large random
secret keys.

:param secret_key: A specific secret key to derive from.
    Defaults to the last item in :attr:`secret_keys`.

.. versionchanged:: 2.0
    Added the ``secret_key`` parameter.

### Function: get_signature(self, value)

**Description:** Returns the signature for the given value.

### Function: sign(self, value)

**Description:** Signs the given string.

### Function: verify_signature(self, value, sig)

**Description:** Verifies the signature for the given value.

### Function: unsign(self, signed_value)

**Description:** Unsigns the given string.

### Function: validate(self, signed_value)

**Description:** Only validates the given signed value. Returns ``True`` if
the signature exists and is valid.
