## AI Summary

A file named _signals.py.


### Function: open_signal_receiver()

**Description:** Start receiving operating system signals.

:param signals: signals to receive (e.g. ``signal.SIGINT``)
:return: an asynchronous context manager for an asynchronous iterator which yields
    signal numbers

.. warning:: Windows does not support signals natively so it is best to avoid
    relying on this in cross-platform applications.

.. warning:: On asyncio, this permanently replaces any previous signal handler for
    the given signals, as set via :meth:`~asyncio.loop.add_signal_handler`.
