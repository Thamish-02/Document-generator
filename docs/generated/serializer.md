## AI Summary

A file named serializer.py.


## Class: _PDataSerializer

### Function: is_text_serializer(serializer)

**Description:** Checks whether a serializer generates text or binary.

## Class: Serializer

**Description:** A serializer wraps a :class:`~itsdangerous.signer.Signer` to
enable serializing and securely signing data other than bytes. It
can unsign to verify that the data hasn't been changed.

The serializer provides :meth:`dumps` and :meth:`loads`, similar to
:mod:`json`, and by default uses :mod:`json` internally to serialize
the data to bytes.

The secret key should be a random string of ``bytes`` and should not
be saved to code or version control. Different salts should be used
to distinguish signing in different contexts. See :doc:`/concepts`
for information about the security of the secret key and salt.

:param secret_key: The secret key to sign and verify with. Can be a
    list of keys, oldest to newest, to support key rotation.
:param salt: Extra key to combine with ``secret_key`` to distinguish
    signatures in different contexts.
:param serializer: An object that provides ``dumps`` and ``loads``
    methods for serializing data to a string. Defaults to
    :attr:`default_serializer`, which defaults to :mod:`json`.
:param serializer_kwargs: Keyword arguments to pass when calling
    ``serializer.dumps``.
:param signer: A ``Signer`` class to instantiate when signing data.
    Defaults to :attr:`default_signer`, which defaults to
    :class:`~itsdangerous.signer.Signer`.
:param signer_kwargs: Keyword arguments to pass when instantiating
    the ``Signer`` class.
:param fallback_signers: List of signer parameters to try when
    unsigning with the default signer fails. Each item can be a dict
    of ``signer_kwargs``, a ``Signer`` class, or a tuple of
    ``(signer, signer_kwargs)``. Defaults to
    :attr:`default_fallback_signers`.

.. versionchanged:: 2.0
    Added support for key rotation by passing a list to
    ``secret_key``.

.. versionchanged:: 2.0
    Removed the default SHA-512 fallback signer from
    ``default_fallback_signers``.

.. versionchanged:: 1.1
    Added support for ``fallback_signers`` and configured a default
    SHA-512 fallback. This fallback is for users who used the yanked
    1.0.0 release which defaulted to SHA-512.

.. versionchanged:: 0.14
    The ``signer`` and ``signer_kwargs`` parameters were added to
    the constructor.

### Function: loads()

### Function: dumps()

### Function: __init__(self, secret_key, salt, serializer, serializer_kwargs, signer, signer_kwargs, fallback_signers)

### Function: __init__(self, secret_key, salt, serializer, serializer_kwargs, signer, signer_kwargs, fallback_signers)

### Function: __init__(self, secret_key, salt)

### Function: __init__(self, secret_key, salt, serializer, serializer_kwargs, signer, signer_kwargs, fallback_signers)

### Function: __init__(self, secret_key, salt)

### Function: __init__(self, secret_key, salt, serializer, serializer_kwargs, signer, signer_kwargs, fallback_signers)

### Function: secret_key(self)

**Description:** The newest (last) entry in the :attr:`secret_keys` list. This
is for compatibility from before key rotation support was added.

### Function: load_payload(self, payload, serializer)

**Description:** Loads the encoded object. This function raises
:class:`.BadPayload` if the payload is not valid. The
``serializer`` parameter can be used to override the serializer
stored on the class. The encoded ``payload`` should always be
bytes.

### Function: dump_payload(self, obj)

**Description:** Dumps the encoded object. The return value is always bytes.
If the internal serializer returns text, the value will be
encoded as UTF-8.

### Function: make_signer(self, salt)

**Description:** Creates a new instance of the signer to be used. The default
implementation uses the :class:`.Signer` base class.

### Function: iter_unsigners(self, salt)

**Description:** Iterates over all signers to be tried for unsigning. Starts
with the configured signer, then constructs each signer
specified in ``fallback_signers``.

### Function: dumps(self, obj, salt)

**Description:** Returns a signed string serialized with the internal
serializer. The return value can be either a byte or unicode
string depending on the format of the internal serializer.

### Function: dump(self, obj, f, salt)

**Description:** Like :meth:`dumps` but dumps into a file. The file handle has
to be compatible with what the internal serializer expects.

### Function: loads(self, s, salt)

**Description:** Reverse of :meth:`dumps`. Raises :exc:`.BadSignature` if the
signature validation fails.

### Function: load(self, f, salt)

**Description:** Like :meth:`loads` but loads from a file.

### Function: loads_unsafe(self, s, salt)

**Description:** Like :meth:`loads` but without verifying the signature. This
is potentially very dangerous to use depending on how your
serializer works. The return value is ``(signature_valid,
payload)`` instead of just the payload. The first item will be a
boolean that indicates if the signature is valid. This function
never fails.

Use it for debugging only and if you know that your serializer
module is not exploitable (for example, do not use it with a
pickle serializer).

.. versionadded:: 0.15

### Function: _loads_unsafe_impl(self, s, salt, load_kwargs, load_payload_kwargs)

**Description:** Low level helper function to implement :meth:`loads_unsafe`
in serializer subclasses.

### Function: load_unsafe(self, f, salt)

**Description:** Like :meth:`loads_unsafe` but loads from a file.

.. versionadded:: 0.15
