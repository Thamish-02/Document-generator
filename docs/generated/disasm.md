## AI Summary

A file named disasm.py.


## Class: Engine

**Description:** Base class for disassembly engine adaptors.

@type name: str
@cvar name: Engine name to use with the L{Disassembler} class.

@type desc: str
@cvar desc: User friendly name of the disassembler engine.

@type url: str
@cvar url: Download URL.

@type supported: set(str)
@cvar supported: Set of supported processor architectures.
    For more details see L{win32.version._get_arch}.

@type arch: str
@ivar arch: Name of the processor architecture.

## Class: BeaEngine

**Description:** Integration with the BeaEngine disassembler by Beatrix.

@see: U{https://sourceforge.net/projects/winappdbg/files/additional%20packages/BeaEngine/}

## Class: DistormEngine

**Description:** Integration with the diStorm disassembler by Gil Dabah.

@see: U{https://code.google.com/p/distorm3}

## Class: PyDasmEngine

**Description:** Integration with PyDasm: Python bindings to libdasm.

@see: U{https://code.google.com/p/libdasm/}

## Class: LibdisassembleEngine

**Description:** Integration with Immunity libdisassemble.

@see: U{http://www.immunitysec.com/resources-freesoftware.shtml}

## Class: CapstoneEngine

**Description:** Integration with the Capstone disassembler by Nguyen Anh Quynh.

@see: U{http://www.capstone-engine.org/}

## Class: Disassembler

**Description:** Generic disassembler. Uses a set of adapters to decide which library to
load for which supported platform.

@type engines: tuple( L{Engine} )
@cvar engines: Set of supported engines. If you implement your own adapter
    you can add its class here to make it available to L{Disassembler}.
    Supported disassemblers are:

### Function: __init__(self, arch)

**Description:** @type  arch: str
@param arch: Name of the processor architecture.
    If not provided the current processor architecture is assumed.
    For more details see L{win32.version._get_arch}.

@raise NotImplementedError: This disassembler doesn't support the
    requested processor architecture.

### Function: _validate_arch(self, arch)

**Description:** @type  arch: str
@param arch: Name of the processor architecture.
    If not provided the current processor architecture is assumed.
    For more details see L{win32.version._get_arch}.

@rtype:  str
@return: Name of the processor architecture.
    If not provided the current processor architecture is assumed.
    For more details see L{win32.version._get_arch}.

@raise NotImplementedError: This disassembler doesn't support the
    requested processor architecture.

### Function: _import_dependencies(self)

**Description:** Loads the dependencies for this disassembler.

@raise ImportError: This disassembler cannot find or load the
    necessary dependencies to make it work.

### Function: decode(self, address, code)

**Description:** @type  address: int
@param address: Memory address where the code was read from.

@type  code: str
@param code: Machine code to disassemble.

@rtype:  list of tuple( long, int, str, str )
@return: List of tuples. Each tuple represents an assembly instruction
    and contains:
     - Memory address of instruction.
     - Size of instruction in bytes.
     - Disassembly line of instruction.
     - Hexadecimal dump of instruction.

@raise NotImplementedError: This disassembler could not be loaded.
    This may be due to missing dependencies.

### Function: _import_dependencies(self)

### Function: decode(self, address, code)

### Function: _import_dependencies(self)

### Function: decode(self, address, code)

### Function: _import_dependencies(self)

### Function: decode(self, address, code)

### Function: _import_dependencies(self)

### Function: decode(self, address, code)

### Function: _import_dependencies(self)

### Function: decode(self, address, code)

### Function: __new__(cls, arch, engine)

**Description:** Factory class. You can't really instance a L{Disassembler} object,
instead one of the adapter L{Engine} subclasses is returned.

@type  arch: str
@param arch: (Optional) Name of the processor architecture.
    If not provided the current processor architecture is assumed.
    For more details see L{win32.version._get_arch}.

@type  engine: str
@param engine: (Optional) Name of the disassembler engine.
    If not provided a compatible one is loaded automatically.
    See: L{Engine.name}

@raise NotImplementedError: No compatible disassembler was found that
    could decode machine code for the requested architecture. This may
    be due to missing dependencies.

@raise ValueError: An unknown engine name was supplied.
