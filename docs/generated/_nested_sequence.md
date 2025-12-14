## AI Summary

A file named _nested_sequence.py.


## Class: _NestedSequence

**Description:** A protocol for representing nested sequences.

Warning
-------
`_NestedSequence` currently does not work in combination with typevars,
*e.g.* ``def func(a: _NestedSequnce[T]) -> T: ...``.

See Also
--------
collections.abc.Sequence
    ABCs for read-only and mutable :term:`sequences`.

Examples
--------
.. code-block:: python

    >>> from __future__ import annotations

    >>> from typing import TYPE_CHECKING
    >>> import numpy as np
    >>> from numpy._typing import _NestedSequence

    >>> def get_dtype(seq: _NestedSequence[float]) -> np.dtype[np.float64]:
    ...     return np.asarray(seq).dtype

    >>> a = get_dtype([1.0])
    >>> b = get_dtype([[1.0]])
    >>> c = get_dtype([[[1.0]]])
    >>> d = get_dtype([[[[1.0]]]])

    >>> if TYPE_CHECKING:
    ...     reveal_locals()
    ...     # note: Revealed local types are:
    ...     # note:     a: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
    ...     # note:     b: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
    ...     # note:     c: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
    ...     # note:     d: numpy.dtype[numpy.floating[numpy._typing._64Bit]]

### Function: __len__()

**Description:** Implement ``len(self)``.

### Function: __getitem__()

**Description:** Implement ``self[x]``.

### Function: __contains__()

**Description:** Implement ``x in self``.

### Function: __iter__()

**Description:** Implement ``iter(self)``.

### Function: __reversed__()

**Description:** Implement ``reversed(self)``.

### Function: count()

**Description:** Return the number of occurrences of `value`.

### Function: index()

**Description:** Return the first index of `value`.
