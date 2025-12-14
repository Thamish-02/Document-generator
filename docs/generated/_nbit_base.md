## AI Summary

A file named _nbit_base.py.


## Class: NBitBase

**Description:** A type representing `numpy.number` precision during static type checking.

Used exclusively for the purpose static type checking, `NBitBase`
represents the base of a hierarchical set of subclasses.
Each subsequent subclass is herein used for representing a lower level
of precision, *e.g.* ``64Bit > 32Bit > 16Bit``.

.. versionadded:: 1.20

Examples
--------
Below is a typical usage example: `NBitBase` is herein used for annotating
a function that takes a float and integer of arbitrary precision
as arguments and returns a new float of whichever precision is largest
(*e.g.* ``np.float16 + np.int64 -> np.float64``).

.. code-block:: python

    >>> from __future__ import annotations
    >>> from typing import TypeVar, TYPE_CHECKING
    >>> import numpy as np
    >>> import numpy.typing as npt

    >>> S = TypeVar("S", bound=npt.NBitBase)
    >>> T = TypeVar("T", bound=npt.NBitBase)

    >>> def add(a: np.floating[S], b: np.integer[T]) -> np.floating[S | T]:
    ...     return a + b

    >>> a = np.float16()
    >>> b = np.int64()
    >>> out = add(a, b)

    >>> if TYPE_CHECKING:
    ...     reveal_locals()
    ...     # note: Revealed local types are:
    ...     # note:     a: numpy.floating[numpy.typing._16Bit*]
    ...     # note:     b: numpy.signedinteger[numpy.typing._64Bit*]
    ...     # note:     out: numpy.floating[numpy.typing._64Bit*]

## Class: _256Bit

## Class: _128Bit

## Class: _96Bit

## Class: _80Bit

## Class: _64Bit

## Class: _32Bit

## Class: _16Bit

## Class: _8Bit

### Function: __init_subclass__(cls)
