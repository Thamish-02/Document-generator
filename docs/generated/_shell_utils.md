## AI Summary

A file named _shell_utils.py.


## Class: CommandLineParser

**Description:** An object that knows how to split and join command-line arguments.

It must be true that ``argv == split(join(argv))`` for all ``argv``.
The reverse neednt be true - `join(split(cmd))` may result in the addition
or removal of unnecessary escaping.

## Class: WindowsParser

**Description:** The parsing behavior used by `subprocess.call("string")` on Windows, which
matches the Microsoft C/C++ runtime.

Note that this is _not_ the behavior of cmd.

## Class: PosixParser

**Description:** The parsing behavior used by `subprocess.call("string", shell=True)` on Posix.

### Function: join(argv)

**Description:** Join a list of arguments into a command line string 

### Function: split(cmd)

**Description:** Split a command line string into a list of arguments 

### Function: join(argv)

### Function: split(cmd)

### Function: join(argv)

### Function: split(cmd)
