## AI Summary

A file named crash.py.


## Class: Marshaller

**Description:** Custom pickler for L{Crash} objects. Optimizes the pickled data when using
the standard C{pickle} (or C{cPickle}) module. The pickled data is then
compressed using zlib.

## Class: CrashWarning

**Description:** An error occurred while gathering crash data.
Some data may be incomplete or missing.

## Class: Crash

**Description:** Represents a crash, bug, or another interesting event in the debugee.

@group Basic information:
    timeStamp, signature, eventCode, eventName, pid, tid, arch, os, bits,
    registers, labelPC, pc, sp, fp

@group Optional information:
    debugString,
    modFileName,
    lpBaseOfDll,
    exceptionCode,
    exceptionName,
    exceptionDescription,
    exceptionAddress,
    exceptionLabel,
    firstChance,
    faultType,
    faultAddress,
    faultLabel,
    isOurBreakpoint,
    isSystemBreakpoint,
    stackTrace,
    stackTracePC,
    stackTraceLabels,
    stackTracePretty

@group Extra information:
    commandLine,
    environment,
    environmentData,
    registersPeek,
    stackRange,
    stackFrame,
    stackPeek,
    faultCode,
    faultMem,
    faultPeek,
    faultDisasm,
    memoryMap

@group Report:
    briefReport, fullReport, notesReport, environmentReport, isExploitable

@group Notes:
    addNote, getNotes, iterNotes, hasNotes, clearNotes, notes

@group Miscellaneous:
    fetch_extra_data

@type timeStamp: float
@ivar timeStamp: Timestamp as returned by time.time().

@type signature: object
@ivar signature: Approximately unique signature for the Crash object.

    This signature can be used as an heuristic to determine if two crashes
    were caused by the same software error. Ideally it should be treated as
    as opaque serializable object that can be tested for equality.

@type notes: list( str )
@ivar notes: List of strings, each string is a note.

@type eventCode: int
@ivar eventCode: Event code as defined by the Win32 API.

@type eventName: str
@ivar eventName: Event code user-friendly name.

@type pid: int
@ivar pid: Process global ID.

@type tid: int
@ivar tid: Thread global ID.

@type arch: str
@ivar arch: Processor architecture.

@type os: str
@ivar os: Operating system version.

    May indicate a 64 bit version even if L{arch} and L{bits} indicate 32
    bits. This means the crash occurred inside a WOW64 process.

@type bits: int
@ivar bits: C{32} or C{64} bits.

@type commandLine: None or str
@ivar commandLine: Command line for the target process.

    C{None} if unapplicable or unable to retrieve.

@type environmentData: None or list of str
@ivar environmentData: Environment data for the target process.

    C{None} if unapplicable or unable to retrieve.

@type environment: None or dict( str S{->} str )
@ivar environment: Environment variables for the target process.

    C{None} if unapplicable or unable to retrieve.

@type registers: dict( str S{->} int )
@ivar registers: Dictionary mapping register names to their values.

@type registersPeek: None or dict( str S{->} str )
@ivar registersPeek: Dictionary mapping register names to the data they point to.

    C{None} if unapplicable or unable to retrieve.

@type labelPC: None or str
@ivar labelPC: Label pointing to the program counter.

    C{None} or invalid if unapplicable or unable to retrieve.

@type debugString: None or str
@ivar debugString: Debug string sent by the debugee.

    C{None} if unapplicable or unable to retrieve.

@type exceptionCode: None or int
@ivar exceptionCode: Exception code as defined by the Win32 API.

    C{None} if unapplicable or unable to retrieve.

@type exceptionName: None or str
@ivar exceptionName: Exception code user-friendly name.

    C{None} if unapplicable or unable to retrieve.

@type exceptionDescription: None or str
@ivar exceptionDescription: Exception description.

    C{None} if unapplicable or unable to retrieve.

@type exceptionAddress: None or int
@ivar exceptionAddress: Memory address where the exception occured.

    C{None} if unapplicable or unable to retrieve.

@type exceptionLabel: None or str
@ivar exceptionLabel: Label pointing to the exception address.

    C{None} or invalid if unapplicable or unable to retrieve.

@type faultType: None or int
@ivar faultType: Access violation type.
    Only applicable to memory faults.
    Should be one of the following constants:

     - L{win32.ACCESS_VIOLATION_TYPE_READ}
     - L{win32.ACCESS_VIOLATION_TYPE_WRITE}
     - L{win32.ACCESS_VIOLATION_TYPE_DEP}

    C{None} if unapplicable or unable to retrieve.

@type faultAddress: None or int
@ivar faultAddress: Access violation memory address.
    Only applicable to memory faults.

    C{None} if unapplicable or unable to retrieve.

@type faultLabel: None or str
@ivar faultLabel: Label pointing to the access violation memory address.
    Only applicable to memory faults.

    C{None} if unapplicable or unable to retrieve.

@type firstChance: None or bool
@ivar firstChance:
    C{True} for first chance exceptions, C{False} for second chance.

    C{None} if unapplicable or unable to retrieve.

@type isOurBreakpoint: bool
@ivar isOurBreakpoint:
    C{True} for breakpoints defined by the L{Debug} class,
    C{False} otherwise.

    C{None} if unapplicable.

@type isSystemBreakpoint: bool
@ivar isSystemBreakpoint:
    C{True} for known system-defined breakpoints,
    C{False} otherwise.

    C{None} if unapplicable.

@type modFileName: None or str
@ivar modFileName: File name of module where the program counter points to.

    C{None} or invalid if unapplicable or unable to retrieve.

@type lpBaseOfDll: None or int
@ivar lpBaseOfDll: Base of module where the program counter points to.

    C{None} if unapplicable or unable to retrieve.

@type stackTrace: None or tuple of tuple( int, int, str )
@ivar stackTrace:
    Stack trace of the current thread as a tuple of
    ( frame pointer, return address, module filename ).

    C{None} or empty if unapplicable or unable to retrieve.

@type stackTracePretty: None or tuple of tuple( int, str )
@ivar stackTracePretty:
    Stack trace of the current thread as a tuple of
    ( frame pointer, return location ).

    C{None} or empty if unapplicable or unable to retrieve.

@type stackTracePC: None or tuple( int... )
@ivar stackTracePC: Tuple of return addresses in the stack trace.

    C{None} or empty if unapplicable or unable to retrieve.

@type stackTraceLabels: None or tuple( str... )
@ivar stackTraceLabels:
    Tuple of labels pointing to the return addresses in the stack trace.

    C{None} or empty if unapplicable or unable to retrieve.

@type stackRange: tuple( int, int )
@ivar stackRange:
    Stack beginning and end pointers, in memory addresses order.

    C{None} if unapplicable or unable to retrieve.

@type stackFrame: None or str
@ivar stackFrame: Data pointed to by the stack pointer.

    C{None} or empty if unapplicable or unable to retrieve.

@type stackPeek: None or dict( int S{->} str )
@ivar stackPeek: Dictionary mapping stack offsets to the data they point to.

    C{None} or empty if unapplicable or unable to retrieve.

@type faultCode: None or str
@ivar faultCode: Data pointed to by the program counter.

    C{None} or empty if unapplicable or unable to retrieve.

@type faultMem: None or str
@ivar faultMem: Data pointed to by the exception address.

    C{None} or empty if unapplicable or unable to retrieve.

@type faultPeek: None or dict( intS{->} str )
@ivar faultPeek: Dictionary mapping guessed pointers at L{faultMem} to the data they point to.

    C{None} or empty if unapplicable or unable to retrieve.

@type faultDisasm: None or tuple of tuple( long, int, str, str )
@ivar faultDisasm: Dissassembly around the program counter.

    C{None} or empty if unapplicable or unable to retrieve.

@type memoryMap: None or list of L{win32.MemoryBasicInformation} objects.
@ivar memoryMap: Memory snapshot of the program. May contain the actual
    data from the entire process memory if requested.
    See L{fetch_extra_data} for more details.

    C{None} or empty if unapplicable or unable to retrieve.

@type _rowid: int
@ivar _rowid: Row ID in the database. Internally used by the DAO layer.
    Only present in crash dumps retrieved from the database. Do not rely
    on this property to be present in future versions of WinAppDbg.

## Class: CrashContainer

**Description:** Old crash dump persistencer using a DBM database.
Doesn't support duplicate crashes.

@warning:
    DBM database support is provided for backwards compatibility with older
    versions of WinAppDbg. New applications should not use this class.
    Also, DBM databases in Python suffer from multiple problems that can
    easily be avoided by switching to a SQL database.

@see: If you really must use a DBM database, try the standard C{shelve}
    module instead: U{http://docs.python.org/library/shelve.html}

@group Marshalling configuration:
    optimizeKeys, optimizeValues, compressKeys, compressValues, escapeKeys,
    escapeValues, binaryKeys, binaryValues

@type optimizeKeys: bool
@cvar optimizeKeys: Ignored by the current implementation.

    Up to WinAppDbg 1.4 this setting caused the database keys to be
    optimized when pickled with the standard C{pickle} module.

    But with a DBM database backend that causes inconsistencies, since the
    same key can be serialized into multiple optimized pickles, thus losing
    uniqueness.

@type optimizeValues: bool
@cvar optimizeValues: C{True} to optimize the marshalling of keys, C{False}
    otherwise. Only used with the C{pickle} module, ignored when using the
    more secure C{cerealizer} module.

@type compressKeys: bool
@cvar compressKeys: C{True} to compress keys when marshalling, C{False}
    to leave them uncompressed.

@type compressValues: bool
@cvar compressValues: C{True} to compress values when marshalling, C{False}
    to leave them uncompressed.

@type escapeKeys: bool
@cvar escapeKeys: C{True} to escape keys when marshalling, C{False}
    to leave them uncompressed.

@type escapeValues: bool
@cvar escapeValues: C{True} to escape values when marshalling, C{False}
    to leave them uncompressed.

@type binaryKeys: bool
@cvar binaryKeys: C{True} to marshall keys to binary format (the Python
    C{buffer} type), C{False} to use text marshalled keys (C{str} type).

@type binaryValues: bool
@cvar binaryValues: C{True} to marshall values to binary format (the Python
    C{buffer} type), C{False} to use text marshalled values (C{str} type).

## Class: CrashDictionary

**Description:** Dictionary-like persistence interface for L{Crash} objects.

Currently the only implementation is through L{sql.CrashDAO}.

## Class: CrashTable

**Description:** Old crash dump persistencer using a SQLite database.

@warning:
    Superceded by L{CrashDictionary} since WinAppDbg 1.5.
    New applications should not use this class.

## Class: CrashTableMSSQL

**Description:** Old crash dump persistencer using a Microsoft SQL Server database.

@warning:
    Superceded by L{CrashDictionary} since WinAppDbg 1.5.
    New applications should not use this class.

## Class: VolatileCrashContainer

**Description:** Old in-memory crash dump storage.

@warning:
    Superceded by L{CrashDictionary} since WinAppDbg 1.5.
    New applications should not use this class.

## Class: DummyCrashContainer

**Description:** Fakes a database of volatile Crash objects,
trying to mimic part of it's interface, but
doesn't actually store anything.

Normally applications don't need to use this.

@see: L{CrashDictionary}

### Function: optimize(picklestring)

### Function: dumps(obj, protocol)

### Function: loads(data)

### Function: __init__(self, event)

**Description:** @type  event: L{Event}
@param event: Event object for crash.

### Function: fetch_extra_data(self, event, takeMemorySnapshot)

**Description:** Fetch extra data from the L{Event} object.

@note: Since this method may take a little longer to run, it's best to
    call it only after you've determined the crash is interesting and
    you want to save it.

@type  event: L{Event}
@param event: Event object for crash.

@type  takeMemorySnapshot: int
@param takeMemorySnapshot:
    Memory snapshot behavior:
     - C{0} to take no memory information (default).
     - C{1} to take only the memory map.
       See L{Process.get_memory_map}.
     - C{2} to take a full memory snapshot.
       See L{Process.take_memory_snapshot}.
     - C{3} to take a live memory snapshot.
       See L{Process.generate_memory_snapshot}.

### Function: pc(self)

**Description:** Value of the program counter register.

@rtype:  int

### Function: sp(self)

**Description:** Value of the stack pointer register.

@rtype:  int

### Function: fp(self)

**Description:** Value of the frame pointer register.

@rtype:  int

### Function: __str__(self)

### Function: key(self)

**Description:** Alias of L{signature}. Deprecated since WinAppDbg 1.5.

### Function: signature(self)

### Function: isExploitable(self)

**Description:** Guess how likely is it that the bug causing the crash can be leveraged
into an exploitable vulnerability.

@note: Don't take this as an equivalent of a real exploitability
    analysis, that can only be done by a human being! This is only
    a guideline, useful for example to sort crashes - placing the most
    interesting ones at the top.

@see: The heuristics are similar to those of the B{!exploitable}
    extension for I{WinDBG}, which can be downloaded from here:

    U{http://www.codeplex.com/msecdbg}

@rtype: tuple( str, str, str )
@return: The first element of the tuple is the result of the analysis,
    being one of the following:

     - Not an exception
     - Not exploitable
     - Not likely exploitable
     - Unknown
     - Probably exploitable
     - Exploitable

    The second element of the tuple is a code to identify the matched
    heuristic rule.

    The third element of the tuple is a description string of the
    reason behind the result.

### Function: __is_control_flow(self)

**Description:** Private method to tell if the instruction pointed to by the program
counter is a control flow instruction.

Currently only works for x86 and amd64 architectures.

### Function: __is_block_data_move(self)

**Description:** Private method to tell if the instruction pointed to by the program
counter is a block data move instruction.

Currently only works for x86 and amd64 architectures.

### Function: briefReport(self)

**Description:** @rtype:  str
@return: Short description of the event.

### Function: fullReport(self, bShowNotes)

**Description:** @type  bShowNotes: bool
@param bShowNotes: C{True} to show the user notes, C{False} otherwise.

@rtype:  str
@return: Long description of the event.

### Function: environmentReport(self)

**Description:** @rtype: str
@return: The process environment variables,
    merged and formatted for a report.

### Function: notesReport(self)

**Description:** @rtype:  str
@return: All notes, merged and formatted for a report.

### Function: addNote(self, msg)

**Description:** Add a note to the crash event.

@type msg:  str
@param msg: Note text.

### Function: clearNotes(self)

**Description:** Clear the notes of this crash event.

### Function: getNotes(self)

**Description:** Get the list of notes of this crash event.

@rtype:  list( str )
@return: List of notes.

### Function: iterNotes(self)

**Description:** Iterate the notes of this crash event.

@rtype:  listiterator
@return: Iterator of the list of notes.

### Function: hasNotes(self)

**Description:** @rtype:  bool
@return: C{True} if there are notes for this crash event.

### Function: __init__(self, filename, allowRepeatedKeys)

**Description:** @type  filename: str
@param filename: (Optional) File name for crash database.
    If no filename is specified, the container is volatile.

    Volatile containers are stored only in memory and
    destroyed when they go out of scope.

@type  allowRepeatedKeys: bool
@param allowRepeatedKeys:
    Currently not supported, always use C{False}.

### Function: remove_key(self, key)

**Description:** Removes the given key from the set of known keys.

@type  key: L{Crash} key.
@param key: Key to remove.

### Function: marshall_key(self, key)

**Description:** Marshalls a Crash key to be used in the database.

@see: L{__init__}

@type  key: L{Crash} key.
@param key: Key to convert.

@rtype:  str or buffer
@return: Converted key.

### Function: unmarshall_key(self, key)

**Description:** Unmarshalls a Crash key read from the database.

@type  key: str or buffer
@param key: Key to convert.

@rtype:  L{Crash} key.
@return: Converted key.

### Function: marshall_value(self, value, storeMemoryMap)

**Description:** Marshalls a Crash object to be used in the database.
By default the C{memoryMap} member is B{NOT} stored here.

@warning: Setting the C{storeMemoryMap} argument to C{True} can lead to
    a severe performance penalty!

@type  value: L{Crash}
@param value: Object to convert.

@type  storeMemoryMap: bool
@param storeMemoryMap: C{True} to store the memory map, C{False}
    otherwise.

@rtype:  str
@return: Converted object.

### Function: unmarshall_value(self, value)

**Description:** Unmarshalls a Crash object read from the database.

@type  value: str
@param value: Object to convert.

@rtype:  L{Crash}
@return: Converted object.

### Function: __len__(self)

**Description:** @rtype:  int
@return: Count of known keys.

### Function: __bool__(self)

**Description:** @rtype:  bool
@return: C{False} if there are no known keys.

### Function: __contains__(self, crash)

**Description:** @type  crash: L{Crash}
@param crash: Crash object.

@rtype:  bool
@return:
    C{True} if a Crash object with the same key is in the container.

### Function: has_key(self, key)

**Description:** @type  key: L{Crash} key.
@param key: Key to find.

@rtype:  bool
@return: C{True} if the key is present in the set of known keys.

### Function: iterkeys(self)

**Description:** @rtype:  iterator
@return: Iterator of known L{Crash} keys.

## Class: __CrashContainerIterator

**Description:** Iterator of Crash objects. Returned by L{CrashContainer.__iter__}.

### Function: __del__(self)

**Description:** Class destructor. Closes the database when this object is destroyed.

### Function: __iter__(self)

**Description:** @see:    L{itervalues}
@rtype:  iterator
@return: Iterator of the contained L{Crash} objects.

### Function: itervalues(self)

**Description:** @rtype:  iterator
@return: Iterator of the contained L{Crash} objects.

@warning: A B{copy} of each object is returned,
    so any changes made to them will be lost.

    To preserve changes do the following:
        1. Keep a reference to the object.
        2. Delete the object from the set.
        3. Modify the object and add it again.

### Function: add(self, crash)

**Description:** Adds a new crash to the container.
If the crash appears to be already known, it's ignored.

@see: L{Crash.key}

@type  crash: L{Crash}
@param crash: Crash object to add.

### Function: __delitem__(self, key)

**Description:** Removes a crash from the container.

@type  key: L{Crash} unique key.
@param key: Key of the crash to get.

### Function: remove(self, crash)

**Description:** Removes a crash from the container.

@type  crash: L{Crash}
@param crash: Crash object to remove.

### Function: get(self, key)

**Description:** Retrieves a crash from the container.

@type  key: L{Crash} unique key.
@param key: Key of the crash to get.

@rtype:  L{Crash} object.
@return: Crash matching the given key.

@see:     L{iterkeys}
@warning: A B{copy} of each object is returned,
    so any changes made to them will be lost.

    To preserve changes do the following:
        1. Keep a reference to the object.
        2. Delete the object from the set.
        3. Modify the object and add it again.

### Function: __getitem__(self, key)

**Description:** Retrieves a crash from the container.

@type  key: L{Crash} unique key.
@param key: Key of the crash to get.

@rtype:  L{Crash} object.
@return: Crash matching the given key.

@see:     L{iterkeys}
@warning: A B{copy} of each object is returned,
    so any changes made to them will be lost.

    To preserve changes do the following:
        1. Keep a reference to the object.
        2. Delete the object from the set.
        3. Modify the object and add it again.

### Function: __init__(self, url, creator, allowRepeatedKeys)

**Description:** @type  url: str
@param url: Connection URL of the crash database.
    See L{sql.CrashDAO.__init__} for more details.

@type  creator: callable
@param creator: (Optional) Callback function that creates the SQL
    database connection.

    Normally it's not necessary to use this argument. However in some
    odd cases you may need to customize the database connection, for
    example when using the integrated authentication in MSSQL.

@type  allowRepeatedKeys: bool
@param allowRepeatedKeys:
    If C{True} all L{Crash} objects are stored.

    If C{False} any L{Crash} object with the same signature as a
    previously existing object will be ignored.

### Function: add(self, crash)

**Description:** Adds a new crash to the container.

@note:
    When the C{allowRepeatedKeys} parameter of the constructor
    is set to C{False}, duplicated crashes are ignored.

@see: L{Crash.key}

@type  crash: L{Crash}
@param crash: Crash object to add.

### Function: get(self, key)

**Description:** Retrieves a crash from the container.

@type  key: L{Crash} signature.
@param key: Heuristic signature of the crash to get.

@rtype:  L{Crash} object.
@return: Crash matching the given signature. If more than one is found,
    retrieve the newest one.

@see:     L{iterkeys}
@warning: A B{copy} of each object is returned,
    so any changes made to them will be lost.

    To preserve changes do the following:
        1. Keep a reference to the object.
        2. Delete the object from the set.
        3. Modify the object and add it again.

### Function: __iter__(self)

**Description:** @rtype:  iterator
@return: Iterator of the contained L{Crash} objects.

### Function: itervalues(self)

**Description:** @rtype:  iterator
@return: Iterator of the contained L{Crash} objects.

### Function: iterkeys(self)

**Description:** @rtype:  iterator
@return: Iterator of the contained L{Crash} heuristic signatures.

### Function: __contains__(self, crash)

**Description:** @type  crash: L{Crash}
@param crash: Crash object.

@rtype:  bool
@return: C{True} if the Crash object is in the container.

### Function: has_key(self, key)

**Description:** @type  key: L{Crash} signature.
@param key: Heuristic signature of the crash to get.

@rtype:  bool
@return: C{True} if a matching L{Crash} object is in the container.

### Function: __len__(self)

**Description:** @rtype:  int
@return: Count of L{Crash} elements in the container.

### Function: __bool__(self)

**Description:** @rtype:  bool
@return: C{False} if the container is empty.

### Function: __init__(self, location, allowRepeatedKeys)

**Description:** @type  location: str
@param location: (Optional) Location of the crash database.
    If the location is a filename, it's an SQLite database file.

    If no location is specified, the container is volatile.
    Volatile containers are stored only in memory and
    destroyed when they go out of scope.

@type  allowRepeatedKeys: bool
@param allowRepeatedKeys:
    If C{True} all L{Crash} objects are stored.

    If C{False} any L{Crash} object with the same signature as a
    previously existing object will be ignored.

### Function: __init__(self, location, allowRepeatedKeys)

**Description:** @type  location: str
@param location: Location of the crash database.
    It must be an ODBC connection string.

@type  allowRepeatedKeys: bool
@param allowRepeatedKeys:
    If C{True} all L{Crash} objects are stored.

    If C{False} any L{Crash} object with the same signature as a
    previously existing object will be ignored.

### Function: __init__(self, allowRepeatedKeys)

**Description:** Volatile containers are stored only in memory and
destroyed when they go out of scope.

@type  allowRepeatedKeys: bool
@param allowRepeatedKeys:
    If C{True} all L{Crash} objects are stored.

    If C{False} any L{Crash} object with the same key as a
    previously existing object will be ignored.

### Function: __init__(self, allowRepeatedKeys)

**Description:** Fake containers don't store L{Crash} objects, but they implement the
interface properly.

@type  allowRepeatedKeys: bool
@param allowRepeatedKeys:
    Mimics the duplicate filter behavior found in real containers.

### Function: __contains__(self, crash)

**Description:** @type  crash: L{Crash}
@param crash: Crash object.

@rtype:  bool
@return: C{True} if the Crash object is in the container.

### Function: __len__(self)

**Description:** @rtype:  int
@return: Count of L{Crash} elements in the container.

### Function: __bool__(self)

**Description:** @rtype:  bool
@return: C{False} if the container is empty.

### Function: add(self, crash)

**Description:** Adds a new crash to the container.

@note:
    When the C{allowRepeatedKeys} parameter of the constructor
    is set to C{False}, duplicated crashes are ignored.

@see: L{Crash.key}

@type  crash: L{Crash}
@param crash: Crash object to add.

### Function: get(self, key)

**Description:** This method is not supported.

### Function: has_key(self, key)

**Description:** @type  key: L{Crash} signature.
@param key: Heuristic signature of the crash to get.

@rtype:  bool
@return: C{True} if a matching L{Crash} object is in the container.

### Function: iterkeys(self)

**Description:** @rtype:  iterator
@return: Iterator of the contained L{Crash} object keys.

@see:     L{get}
@warning: A B{copy} of each object is returned,
    so any changes made to them will be lost.

    To preserve changes do the following:
        1. Keep a reference to the object.
        2. Delete the object from the set.
        3. Modify the object and add it again.

### Function: __init__(self, container)

**Description:** @type  container: L{CrashContainer}
@param container: Crash set to iterate.

### Function: next(self)

**Description:** @rtype:  L{Crash}
@return: A B{copy} of a Crash object in the L{CrashContainer}.
@raise StopIteration: No more items left.

### Function: optimize(picklestring)
