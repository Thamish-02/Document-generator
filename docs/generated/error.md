## AI Summary

A file named error.py.


## Class: IPythonCoreError

## Class: TryNext

**Description:** Try next hook exception.

Raise this in your hook function to indicate that the next hook handler
should be used to handle the operation.

## Class: UsageError

**Description:** Error in magic function arguments, etc.

Something that probably won't warrant a full traceback, but should
nevertheless interrupt a macro / batch file.

## Class: StdinNotImplementedError

**Description:** raw_input was requested in a context where it is not supported

For use in IPython kernels, where only some frontends may support
stdin requests.

## Class: InputRejected

**Description:** Input rejected by ast transformer.

Raise this in your NodeTransformer to indicate that InteractiveShell should
not execute the supplied input.
