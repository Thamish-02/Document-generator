## AI Summary

A file named timed.py.


## Class: TimestampSigner

**Description:** Works like the regular :class:`.Signer` but also records the time
of the signing and can be used to expire signatures. The
:meth:`unsign` method can raise :exc:`.SignatureExpired` if the
unsigning failed because the signature is expired.

## Class: TimedSerializer

**Description:** Uses :class:`TimestampSigner` instead of the default
:class:`.Signer`.

### Function: get_timestamp(self)

**Description:** Returns the current timestamp. The function must return an
integer.

### Function: timestamp_to_datetime(self, ts)

**Description:** Convert the timestamp from :meth:`get_timestamp` into an
aware :class`datetime.datetime` in UTC.

.. versionchanged:: 2.0
    The timestamp is returned as a timezone-aware ``datetime``
    in UTC rather than a naive ``datetime`` assumed to be UTC.

### Function: sign(self, value)

**Description:** Signs the given string and also attaches time information.

### Function: unsign(self, signed_value, max_age, return_timestamp)

### Function: unsign(self, signed_value, max_age, return_timestamp)

### Function: unsign(self, signed_value, max_age, return_timestamp)

**Description:** Works like the regular :meth:`.Signer.unsign` but can also
validate the time. See the base docstring of the class for
the general behavior. If ``return_timestamp`` is ``True`` the
timestamp of the signature will be returned as an aware
:class:`datetime.datetime` object in UTC.

.. versionchanged:: 2.0
    The timestamp is returned as a timezone-aware ``datetime``
    in UTC rather than a naive ``datetime`` assumed to be UTC.

### Function: validate(self, signed_value, max_age)

**Description:** Only validates the given signed value. Returns ``True`` if
the signature exists and is valid.

### Function: iter_unsigners(self, salt)

### Function: loads(self, s, max_age, return_timestamp, salt)

**Description:** Reverse of :meth:`dumps`, raises :exc:`.BadSignature` if the
signature validation fails. If a ``max_age`` is provided it will
ensure the signature is not older than that time in seconds. In
case the signature is outdated, :exc:`.SignatureExpired` is
raised. All arguments are forwarded to the signer's
:meth:`~TimestampSigner.unsign` method.

### Function: loads_unsafe(self, s, max_age, salt)
