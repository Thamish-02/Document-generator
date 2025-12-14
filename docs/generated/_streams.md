## AI Summary

A file named _streams.py.


## Class: create_memory_object_stream

**Description:** Create a memory object stream.

The stream's item type can be annotated like
:func:`create_memory_object_stream[T_Item]`.

:param max_buffer_size: number of items held in the buffer until ``send()`` starts
    blocking
:param item_type: old way of marking the streams with the right generic type for
    static typing (does nothing on AnyIO 4)

    .. deprecated:: 4.0
      Use ``create_memory_object_stream[YourItemType](...)`` instead.
:return: a tuple of (send stream, receive stream)

### Function: __new__(cls, max_buffer_size, item_type)
