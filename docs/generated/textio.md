## AI Summary

A file named textio.py.


## Class: HexInput

**Description:** Static functions for user input parsing.
The counterparts for each method are in the L{HexOutput} class.

## Class: HexOutput

**Description:** Static functions for user output parsing.
The counterparts for each method are in the L{HexInput} class.

@type integer_size: int
@cvar integer_size: Default size in characters of an outputted integer.
    This value is platform dependent.

@type address_size: int
@cvar address_size: Default Number of bits of the target architecture.
    This value is platform dependent.

## Class: HexDump

**Description:** Static functions for hexadecimal dumps.

@type integer_size: int
@cvar integer_size: Size in characters of an outputted integer.
    This value is platform dependent.

@type address_size: int
@cvar address_size: Size in characters of an outputted address.
    This value is platform dependent.

## Class: Color

**Description:** Colored console output.

## Class: Table

**Description:** Text based table. The number of columns and the width of each column
is automatically calculated.

## Class: CrashDump

**Description:** Static functions for crash dumps.

@type reg_template: str
@cvar reg_template: Template for the L{dump_registers} method.

## Class: DebugLog

**Description:** Static functions for debug logging.

## Class: Logger

**Description:** Logs text to standard output and/or a text file.

@type logfile: str or None
@ivar logfile: Append messages to this text file.

@type verbose: bool
@ivar verbose: C{True} to print messages to standard output.

@type fd: file
@ivar fd: File object where log messages are printed to.
    C{None} if no log file is used.

### Function: integer(token)

**Description:** Convert numeric strings into integers.

@type  token: str
@param token: String to parse.

@rtype:  int
@return: Parsed integer value.

### Function: address(token)

**Description:** Convert numeric strings into memory addresses.

@type  token: str
@param token: String to parse.

@rtype:  int
@return: Parsed integer value.

### Function: hexadecimal(token)

**Description:** Convert a strip of hexadecimal numbers into binary data.

@type  token: str
@param token: String to parse.

@rtype:  str
@return: Parsed string value.

### Function: pattern(token)

**Description:** Convert an hexadecimal search pattern into a POSIX regular expression.

For example, the following pattern::

    "B8 0? ?0 ?? ??"

Would match the following data::

    "B8 0D F0 AD BA"    # mov eax, 0xBAADF00D

@type  token: str
@param token: String to parse.

@rtype:  str
@return: Parsed string value.

### Function: is_pattern(token)

**Description:** Determine if the given argument is a valid hexadecimal pattern to be
used with L{pattern}.

@type  token: str
@param token: String to parse.

@rtype:  bool
@return:
    C{True} if it's a valid hexadecimal pattern, C{False} otherwise.

### Function: integer_list_file(cls, filename)

**Description:** Read a list of integers from a file.

The file format is:

 - # anywhere in the line begins a comment
 - leading and trailing spaces are ignored
 - empty lines are ignored
 - integers can be specified as:
    - decimal numbers ("100" is 100)
    - hexadecimal numbers ("0x100" is 256)
    - binary numbers ("0b100" is 4)
    - octal numbers ("0100" is 64)

@type  filename: str
@param filename: Name of the file to read.

@rtype:  list( int )
@return: List of integers read from the file.

### Function: string_list_file(cls, filename)

**Description:** Read a list of string values from a file.

The file format is:

 - # anywhere in the line begins a comment
 - leading and trailing spaces are ignored
 - empty lines are ignored
 - strings cannot span over a single line

@type  filename: str
@param filename: Name of the file to read.

@rtype:  list
@return: List of integers and strings read from the file.

### Function: mixed_list_file(cls, filename)

**Description:** Read a list of mixed values from a file.

The file format is:

 - # anywhere in the line begins a comment
 - leading and trailing spaces are ignored
 - empty lines are ignored
 - strings cannot span over a single line
 - integers can be specified as:
    - decimal numbers ("100" is 100)
    - hexadecimal numbers ("0x100" is 256)
    - binary numbers ("0b100" is 4)
    - octal numbers ("0100" is 64)

@type  filename: str
@param filename: Name of the file to read.

@rtype:  list
@return: List of integers and strings read from the file.

### Function: integer(cls, integer, bits)

**Description:** @type  integer: int
@param integer: Integer.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexOutput.integer_size}

@rtype:  str
@return: Text output.

### Function: address(cls, address, bits)

**Description:** @type  address: int
@param address: Memory address.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexOutput.address_size}

@rtype:  str
@return: Text output.

### Function: hexadecimal(data)

**Description:** Convert binary data to a string of hexadecimal numbers.

@type  data: str
@param data: Binary data.

@rtype:  str
@return: Hexadecimal representation.

### Function: integer_list_file(cls, filename, values, bits)

**Description:** Write a list of integers to a file.
If a file of the same name exists, it's contents are replaced.

See L{HexInput.integer_list_file} for a description of the file format.

@type  filename: str
@param filename: Name of the file to write.

@type  values: list( int )
@param values: List of integers to write to the file.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexOutput.integer_size}

### Function: string_list_file(cls, filename, values)

**Description:** Write a list of strings to a file.
If a file of the same name exists, it's contents are replaced.

See L{HexInput.string_list_file} for a description of the file format.

@type  filename: str
@param filename: Name of the file to write.

@type  values: list( int )
@param values: List of strings to write to the file.

### Function: mixed_list_file(cls, filename, values, bits)

**Description:** Write a list of mixed values to a file.
If a file of the same name exists, it's contents are replaced.

See L{HexInput.mixed_list_file} for a description of the file format.

@type  filename: str
@param filename: Name of the file to write.

@type  values: list( int )
@param values: List of mixed values to write to the file.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexOutput.integer_size}

### Function: integer(cls, integer, bits)

**Description:** @type  integer: int
@param integer: Integer.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.integer_size}

@rtype:  str
@return: Text output.

### Function: address(cls, address, bits)

**Description:** @type  address: int
@param address: Memory address.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.address_size}

@rtype:  str
@return: Text output.

### Function: printable(data)

**Description:** Replace unprintable characters with dots.

@type  data: str
@param data: Binary data.

@rtype:  str
@return: Printable text.

### Function: hexadecimal(data, separator)

**Description:** Convert binary data to a string of hexadecimal numbers.

@type  data: str
@param data: Binary data.

@type  separator: str
@param separator:
    Separator between the hexadecimal representation of each character.

@rtype:  str
@return: Hexadecimal representation.

### Function: hexa_word(data, separator)

**Description:** Convert binary data to a string of hexadecimal WORDs.

@type  data: str
@param data: Binary data.

@type  separator: str
@param separator:
    Separator between the hexadecimal representation of each WORD.

@rtype:  str
@return: Hexadecimal representation.

### Function: hexa_dword(data, separator)

**Description:** Convert binary data to a string of hexadecimal DWORDs.

@type  data: str
@param data: Binary data.

@type  separator: str
@param separator:
    Separator between the hexadecimal representation of each DWORD.

@rtype:  str
@return: Hexadecimal representation.

### Function: hexa_qword(data, separator)

**Description:** Convert binary data to a string of hexadecimal QWORDs.

@type  data: str
@param data: Binary data.

@type  separator: str
@param separator:
    Separator between the hexadecimal representation of each QWORD.

@rtype:  str
@return: Hexadecimal representation.

### Function: hexline(cls, data, separator, width)

**Description:** Dump a line of hexadecimal numbers from binary data.

@type  data: str
@param data: Binary data.

@type  separator: str
@param separator:
    Separator between the hexadecimal representation of each character.

@type  width: int
@param width:
    (Optional) Maximum number of characters to convert per text line.
    This value is also used for padding.

@rtype:  str
@return: Multiline output text.

### Function: hexblock(cls, data, address, bits, separator, width)

**Description:** Dump a block of hexadecimal numbers from binary data.
Also show a printable text version of the data.

@type  data: str
@param data: Binary data.

@type  address: str
@param address: Memory address where the data was read from.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.address_size}

@type  separator: str
@param separator:
    Separator between the hexadecimal representation of each character.

@type  width: int
@param width:
    (Optional) Maximum number of characters to convert per text line.

@rtype:  str
@return: Multiline output text.

### Function: hexblock_cb(cls, callback, data, address, bits, width, cb_args, cb_kwargs)

**Description:** Dump a block of binary data using a callback function to convert each
line of text.

@type  callback: function
@param callback: Callback function to convert each line of data.

@type  data: str
@param data: Binary data.

@type  address: str
@param address:
    (Optional) Memory address where the data was read from.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.address_size}

@type  cb_args: str
@param cb_args:
    (Optional) Arguments to pass to the callback function.

@type  cb_kwargs: str
@param cb_kwargs:
    (Optional) Keyword arguments to pass to the callback function.

@type  width: int
@param width:
    (Optional) Maximum number of bytes to convert per text line.

@rtype:  str
@return: Multiline output text.

### Function: hexblock_byte(cls, data, address, bits, separator, width)

**Description:** Dump a block of hexadecimal BYTEs from binary data.

@type  data: str
@param data: Binary data.

@type  address: str
@param address: Memory address where the data was read from.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.address_size}

@type  separator: str
@param separator:
    Separator between the hexadecimal representation of each BYTE.

@type  width: int
@param width:
    (Optional) Maximum number of BYTEs to convert per text line.

@rtype:  str
@return: Multiline output text.

### Function: hexblock_word(cls, data, address, bits, separator, width)

**Description:** Dump a block of hexadecimal WORDs from binary data.

@type  data: str
@param data: Binary data.

@type  address: str
@param address: Memory address where the data was read from.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.address_size}

@type  separator: str
@param separator:
    Separator between the hexadecimal representation of each WORD.

@type  width: int
@param width:
    (Optional) Maximum number of WORDs to convert per text line.

@rtype:  str
@return: Multiline output text.

### Function: hexblock_dword(cls, data, address, bits, separator, width)

**Description:** Dump a block of hexadecimal DWORDs from binary data.

@type  data: str
@param data: Binary data.

@type  address: str
@param address: Memory address where the data was read from.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.address_size}

@type  separator: str
@param separator:
    Separator between the hexadecimal representation of each DWORD.

@type  width: int
@param width:
    (Optional) Maximum number of DWORDs to convert per text line.

@rtype:  str
@return: Multiline output text.

### Function: hexblock_qword(cls, data, address, bits, separator, width)

**Description:** Dump a block of hexadecimal QWORDs from binary data.

@type  data: str
@param data: Binary data.

@type  address: str
@param address: Memory address where the data was read from.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.address_size}

@type  separator: str
@param separator:
    Separator between the hexadecimal representation of each QWORD.

@type  width: int
@param width:
    (Optional) Maximum number of QWORDs to convert per text line.

@rtype:  str
@return: Multiline output text.

### Function: _get_text_attributes()

### Function: _set_text_attributes(wAttributes)

### Function: can_use_colors(cls)

**Description:** Determine if we can use colors.

Colored output only works when the output is a real console, and fails
when redirected to a file or pipe. Call this method before issuing a
call to any other method of this class to make sure it's actually
possible to use colors.

@rtype:  bool
@return: C{True} if it's possible to output text with color,
    C{False} otherwise.

### Function: reset(cls)

**Description:** Reset the colors to the default values.

### Function: default(cls)

**Description:** Make the current foreground color the default.

### Function: light(cls)

**Description:** Make the current foreground color light.

### Function: dark(cls)

**Description:** Make the current foreground color dark.

### Function: black(cls)

**Description:** Make the text foreground color black.

### Function: white(cls)

**Description:** Make the text foreground color white.

### Function: red(cls)

**Description:** Make the text foreground color red.

### Function: green(cls)

**Description:** Make the text foreground color green.

### Function: blue(cls)

**Description:** Make the text foreground color blue.

### Function: cyan(cls)

**Description:** Make the text foreground color cyan.

### Function: magenta(cls)

**Description:** Make the text foreground color magenta.

### Function: yellow(cls)

**Description:** Make the text foreground color yellow.

### Function: bk_default(cls)

**Description:** Make the current background color the default.

### Function: bk_light(cls)

**Description:** Make the current background color light.

### Function: bk_dark(cls)

**Description:** Make the current background color dark.

### Function: bk_black(cls)

**Description:** Make the text background color black.

### Function: bk_white(cls)

**Description:** Make the text background color white.

### Function: bk_red(cls)

**Description:** Make the text background color red.

### Function: bk_green(cls)

**Description:** Make the text background color green.

### Function: bk_blue(cls)

**Description:** Make the text background color blue.

### Function: bk_cyan(cls)

**Description:** Make the text background color cyan.

### Function: bk_magenta(cls)

**Description:** Make the text background color magenta.

### Function: bk_yellow(cls)

**Description:** Make the text background color yellow.

### Function: __init__(self, sep)

**Description:** @type  sep: str
@param sep: Separator between cells in each row.

### Function: addRow(self)

**Description:** Add a row to the table. All items are converted to strings.

@type    row: tuple
@keyword row: Each argument is a cell in the table.

### Function: justify(self, column, direction)

**Description:** Make the text in a column left or right justified.

@type  column: int
@param column: Index of the column.

@type  direction: int
@param direction:
    C{-1} to justify left,
    C{1} to justify right.

@raise IndexError: Bad column index.
@raise ValueError: Bad direction value.

### Function: getWidth(self)

**Description:** Get the width of the text output for the table.

@rtype:  int
@return: Width in characters for the text output,
    including the newline character.

### Function: getOutput(self)

**Description:** Get the text output for the table.

@rtype:  str
@return: Text output.

### Function: yieldOutput(self)

**Description:** Generate the text output for the table.

@rtype:  generator of str
@return: Text output.

### Function: show(self)

**Description:** Print the text output for the table.

### Function: dump_flags(efl)

**Description:** Dump the x86 processor flags.
The output mimics that of the WinDBG debugger.
Used by L{dump_registers}.

@type  efl: int
@param efl: Value of the eFlags register.

@rtype:  str
@return: Text suitable for logging.

### Function: dump_registers(cls, registers, arch)

**Description:** Dump the x86/x64 processor register values.
The output mimics that of the WinDBG debugger.

@type  registers: dict( str S{->} int )
@param registers: Dictionary mapping register names to their values.

@type  arch: str
@param arch: Architecture of the machine whose registers were dumped.
    Defaults to the current architecture.
    Currently only the following architectures are supported:
     - L{win32.ARCH_I386}
     - L{win32.ARCH_AMD64}

@rtype:  str
@return: Text suitable for logging.

### Function: dump_registers_peek(registers, data, separator, width)

**Description:** Dump data pointed to by the given registers, if any.

@type  registers: dict( str S{->} int )
@param registers: Dictionary mapping register names to their values.
    This value is returned by L{Thread.get_context}.

@type  data: dict( str S{->} str )
@param data: Dictionary mapping register names to the data they point to.
    This value is returned by L{Thread.peek_pointers_in_registers}.

@rtype:  str
@return: Text suitable for logging.

### Function: dump_data_peek(data, base, separator, width, bits)

**Description:** Dump data from pointers guessed within the given binary data.

@type  data: str
@param data: Dictionary mapping offsets to the data they point to.

@type  base: int
@param base: Base offset.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.address_size}

@rtype:  str
@return: Text suitable for logging.

### Function: dump_stack_peek(data, separator, width, arch)

**Description:** Dump data from pointers guessed within the given stack dump.

@type  data: str
@param data: Dictionary mapping stack offsets to the data they point to.

@type  separator: str
@param separator:
    Separator between the hexadecimal representation of each character.

@type  width: int
@param width:
    (Optional) Maximum number of characters to convert per text line.
    This value is also used for padding.

@type  arch: str
@param arch: Architecture of the machine whose registers were dumped.
    Defaults to the current architecture.

@rtype:  str
@return: Text suitable for logging.

### Function: dump_stack_trace(stack_trace, bits)

**Description:** Dump a stack trace, as returned by L{Thread.get_stack_trace} with the
C{bUseLabels} parameter set to C{False}.

@type  stack_trace: list( int, int, str )
@param stack_trace: Stack trace as a list of tuples of
    ( return address, frame pointer, module filename )

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.address_size}

@rtype:  str
@return: Text suitable for logging.

### Function: dump_stack_trace_with_labels(stack_trace, bits)

**Description:** Dump a stack trace,
as returned by L{Thread.get_stack_trace_with_labels}.

@type  stack_trace: list( int, int, str )
@param stack_trace: Stack trace as a list of tuples of
    ( return address, frame pointer, module filename )

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.address_size}

@rtype:  str
@return: Text suitable for logging.

### Function: dump_code(disassembly, pc, bLowercase, bits)

**Description:** Dump a disassembly. Optionally mark where the program counter is.

@type  disassembly: list of tuple( int, int, str, str )
@param disassembly: Disassembly dump as returned by
    L{Process.disassemble} or L{Thread.disassemble_around_pc}.

@type  pc: int
@param pc: (Optional) Program counter.

@type  bLowercase: bool
@param bLowercase: (Optional) If C{True} convert the code to lowercase.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.address_size}

@rtype:  str
@return: Text suitable for logging.

### Function: dump_code_line(disassembly_line, bShowAddress, bShowDump, bLowercase, dwDumpWidth, dwCodeWidth, bits)

**Description:** Dump a single line of code. To dump a block of code use L{dump_code}.

@type  disassembly_line: tuple( int, int, str, str )
@param disassembly_line: Single item of the list returned by
    L{Process.disassemble} or L{Thread.disassemble_around_pc}.

@type  bShowAddress: bool
@param bShowAddress: (Optional) If C{True} show the memory address.

@type  bShowDump: bool
@param bShowDump: (Optional) If C{True} show the hexadecimal dump.

@type  bLowercase: bool
@param bLowercase: (Optional) If C{True} convert the code to lowercase.

@type  dwDumpWidth: int or None
@param dwDumpWidth: (Optional) Width in characters of the hex dump.

@type  dwCodeWidth: int or None
@param dwCodeWidth: (Optional) Width in characters of the code.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.address_size}

@rtype:  str
@return: Text suitable for logging.

### Function: dump_memory_map(memoryMap, mappedFilenames, bits)

**Description:** Dump the memory map of a process. Optionally show the filenames for
memory mapped files as well.

@type  memoryMap: list( L{win32.MemoryBasicInformation} )
@param memoryMap: Memory map returned by L{Process.get_memory_map}.

@type  mappedFilenames: dict( int S{->} str )
@param mappedFilenames: (Optional) Memory mapped filenames
    returned by L{Process.get_mapped_filenames}.

@type  bits: int
@param bits:
    (Optional) Number of bits of the target architecture.
    The default is platform dependent. See: L{HexDump.address_size}

@rtype:  str
@return: Text suitable for logging.

### Function: log_text(text)

**Description:** Log lines of text, inserting a timestamp.

@type  text: str
@param text: Text to log.

@rtype:  str
@return: Log line.

### Function: log_event(cls, event, text)

**Description:** Log lines of text associated with a debug event.

@type  event: L{Event}
@param event: Event object.

@type  text: str
@param text: (Optional) Text to log. If no text is provided the default
    is to show a description of the event itself.

@rtype:  str
@return: Log line.

### Function: __init__(self, logfile, verbose)

**Description:** @type  logfile: str or None
@param logfile: Append messages to this text file.

@type  verbose: bool
@param verbose: C{True} to print messages to standard output.

### Function: __logfile_error(self, e)

**Description:** Shows an error message to standard error
if the log file can't be written to.

Used internally.

@type  e: Exception
@param e: Exception raised when trying to write to the log file.

### Function: __do_log(self, text)

**Description:** Writes the given text verbatim into the log file (if any)
and/or standard input (if the verbose flag is turned on).

Used internally.

@type  text: str
@param text: Text to print.

### Function: log_text(self, text)

**Description:** Log lines of text, inserting a timestamp.

@type  text: str
@param text: Text to log.

### Function: log_event(self, event, text)

**Description:** Log lines of text associated with a debug event.

@type  event: L{Event}
@param event: Event object.

@type  text: str
@param text: (Optional) Text to log. If no text is provided the default
    is to show a description of the event itself.

### Function: log_exc(self)

**Description:** Log lines of text associated with the last Python exception.

### Function: is_enabled(self)

**Description:** Determines if the logger will actually print anything when the log_*
methods are called.

This may save some processing if the log text requires a lengthy
calculation to prepare. If no log file is set and stdout logging
is disabled, there's no point in preparing a log text that won't
be shown to anyone.

@rtype:  bool
@return: C{True} if a log file was set and/or standard output logging
    is enabled, or C{False} otherwise.
