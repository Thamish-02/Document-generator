## AI Summary

A file named breakpoint.py.


## Class: BreakpointWarning

**Description:** This warning is issued when a non-fatal error occurs that's related to
breakpoints.

## Class: BreakpointCallbackWarning

**Description:** This warning is issued when an uncaught exception was raised by a
breakpoint's user-defined callback.

## Class: Breakpoint

**Description:** Base class for breakpoints.
Here's the breakpoints state machine.

@see: L{CodeBreakpoint}, L{PageBreakpoint}, L{HardwareBreakpoint}

@group Breakpoint states:
    DISABLED, ENABLED, ONESHOT, RUNNING
@group State machine:
    hit, disable, enable, one_shot, running,
    is_disabled, is_enabled, is_one_shot, is_running,
    get_state, get_state_name
@group Information:
    get_address, get_size, get_span, is_here
@group Conditional breakpoints:
    is_conditional, is_unconditional,
    get_condition, set_condition, eval_condition
@group Automatic breakpoints:
    is_automatic, is_interactive,
    get_action, set_action, run_action

@cvar DISABLED: I{Disabled} S{->} Enabled, OneShot
@cvar ENABLED:  I{Enabled}  S{->} I{Running}, Disabled
@cvar ONESHOT:  I{OneShot}  S{->} I{Disabled}
@cvar RUNNING:  I{Running}  S{->} I{Enabled}, Disabled

@type DISABLED: int
@type ENABLED:  int
@type ONESHOT:  int
@type RUNNING:  int

@type stateNames: dict E{lb} int S{->} str E{rb}
@cvar stateNames: User-friendly names for each breakpoint state.

@type typeName: str
@cvar typeName: User friendly breakpoint type string.

## Class: CodeBreakpoint

**Description:** Code execution breakpoints (using an int3 opcode).

@see: L{Debug.break_at}

@type bpInstruction: str
@cvar bpInstruction: Breakpoint instruction for the current processor.

## Class: PageBreakpoint

**Description:** Page access breakpoint (using guard pages).

@see: L{Debug.watch_buffer}

@group Information:
    get_size_in_pages

## Class: HardwareBreakpoint

**Description:** Hardware breakpoint (using debug registers).

@see: L{Debug.watch_variable}

@group Information:
    get_slot, get_trigger, get_watch

@group Trigger flags:
    BREAK_ON_EXECUTION, BREAK_ON_WRITE, BREAK_ON_ACCESS

@group Watch size flags:
    WATCH_BYTE, WATCH_WORD, WATCH_DWORD, WATCH_QWORD

@type BREAK_ON_EXECUTION: int
@cvar BREAK_ON_EXECUTION: Break on execution.

@type BREAK_ON_WRITE: int
@cvar BREAK_ON_WRITE: Break on write.

@type BREAK_ON_ACCESS: int
@cvar BREAK_ON_ACCESS: Break on read or write.

@type WATCH_BYTE: int
@cvar WATCH_BYTE: Watch a byte.

@type WATCH_WORD: int
@cvar WATCH_WORD: Watch a word (2 bytes).

@type WATCH_DWORD: int
@cvar WATCH_DWORD: Watch a double word (4 bytes).

@type WATCH_QWORD: int
@cvar WATCH_QWORD: Watch one quad word (8 bytes).

@type validTriggers: tuple
@cvar validTriggers: Valid trigger flag values.

@type validWatchSizes: tuple
@cvar validWatchSizes: Valid watch flag values.

## Class: Hook

**Description:** Factory class to produce hook objects. Used by L{Debug.hook_function} and
L{Debug.stalk_function}.

When you try to instance this class, one of the architecture specific
implementations is returned instead.

Instances act as an action callback for code breakpoints set at the
beginning of a function. It automatically retrieves the parameters from
the stack, sets a breakpoint at the return address and retrieves the
return value from the function call.

@see: L{_Hook_i386}, L{_Hook_amd64}

@type useHardwareBreakpoints: bool
@cvar useHardwareBreakpoints: C{True} to try to use hardware breakpoints,
    C{False} otherwise.

## Class: _Hook_i386

**Description:** Implementation details for L{Hook} on the L{win32.ARCH_I386} architecture.

## Class: _Hook_amd64

**Description:** Implementation details for L{Hook} on the L{win32.ARCH_AMD64} architecture.

## Class: ApiHook

**Description:** Used by L{EventHandler}.

This class acts as an action callback for code breakpoints set at the
beginning of a function. It automatically retrieves the parameters from
the stack, sets a breakpoint at the return address and retrieves the
return value from the function call.

@see: L{EventHandler.apiHooks}

@type modName: str
@ivar modName: Module name.

@type procName: str
@ivar procName: Procedure name.

## Class: BufferWatch

**Description:** Returned by L{Debug.watch_buffer}.

This object uniquely references a buffer being watched, even if there are
multiple watches set on the exact memory region.

@type pid: int
@ivar pid: Process ID.

@type start: int
@ivar start: Memory address of the start of the buffer.

@type end: int
@ivar end: Memory address of the end of the buffer.

@type action: callable
@ivar action: Action callback.

@type oneshot: bool
@ivar oneshot: C{True} for one shot breakpoints, C{False} otherwise.

## Class: _BufferWatchCondition

**Description:** Used by L{Debug.watch_buffer}.

This class acts as a condition callback for page breakpoints.
It emulates page breakpoints that can overlap and/or take up less
than a page's size.

## Class: _BreakpointContainer

**Description:** Encapsulates the capability to contain Breakpoint objects.

@group Breakpoints:
    break_at, watch_variable, watch_buffer, hook_function,
    dont_break_at, dont_watch_variable, dont_watch_buffer,
    dont_hook_function, unhook_function,
    break_on_error, dont_break_on_error

@group Stalking:
    stalk_at, stalk_variable, stalk_buffer, stalk_function,
    dont_stalk_at, dont_stalk_variable, dont_stalk_buffer,
    dont_stalk_function

@group Tracing:
    is_tracing, get_traced_tids,
    start_tracing, stop_tracing,
    start_tracing_process, stop_tracing_process,
    start_tracing_all, stop_tracing_all

@group Symbols:
    resolve_label, resolve_exported_function

@group Advanced breakpoint use:
    define_code_breakpoint,
    define_page_breakpoint,
    define_hardware_breakpoint,
    has_code_breakpoint,
    has_page_breakpoint,
    has_hardware_breakpoint,
    get_code_breakpoint,
    get_page_breakpoint,
    get_hardware_breakpoint,
    erase_code_breakpoint,
    erase_page_breakpoint,
    erase_hardware_breakpoint,
    enable_code_breakpoint,
    enable_page_breakpoint,
    enable_hardware_breakpoint,
    enable_one_shot_code_breakpoint,
    enable_one_shot_page_breakpoint,
    enable_one_shot_hardware_breakpoint,
    disable_code_breakpoint,
    disable_page_breakpoint,
    disable_hardware_breakpoint

@group Listing breakpoints:
    get_all_breakpoints,
    get_all_code_breakpoints,
    get_all_page_breakpoints,
    get_all_hardware_breakpoints,
    get_process_breakpoints,
    get_process_code_breakpoints,
    get_process_page_breakpoints,
    get_process_hardware_breakpoints,
    get_thread_hardware_breakpoints,
    get_all_deferred_code_breakpoints,
    get_process_deferred_code_breakpoints

@group Batch operations on breakpoints:
    enable_all_breakpoints,
    enable_one_shot_all_breakpoints,
    disable_all_breakpoints,
    erase_all_breakpoints,
    enable_process_breakpoints,
    enable_one_shot_process_breakpoints,
    disable_process_breakpoints,
    erase_process_breakpoints

@group Breakpoint types:
    BP_TYPE_ANY, BP_TYPE_CODE, BP_TYPE_PAGE, BP_TYPE_HARDWARE
@group Breakpoint states:
    BP_STATE_DISABLED, BP_STATE_ENABLED, BP_STATE_ONESHOT, BP_STATE_RUNNING
@group Memory breakpoint trigger flags:
    BP_BREAK_ON_EXECUTION, BP_BREAK_ON_WRITE, BP_BREAK_ON_ACCESS
@group Memory breakpoint size flags:
    BP_WATCH_BYTE, BP_WATCH_WORD, BP_WATCH_DWORD, BP_WATCH_QWORD

@type BP_TYPE_ANY: int
@cvar BP_TYPE_ANY: To get all breakpoints
@type BP_TYPE_CODE: int
@cvar BP_TYPE_CODE: To get code breakpoints only
@type BP_TYPE_PAGE: int
@cvar BP_TYPE_PAGE: To get page breakpoints only
@type BP_TYPE_HARDWARE: int
@cvar BP_TYPE_HARDWARE: To get hardware breakpoints only

@type BP_STATE_DISABLED: int
@cvar BP_STATE_DISABLED: Breakpoint is disabled.
@type BP_STATE_ENABLED: int
@cvar BP_STATE_ENABLED: Breakpoint is enabled.
@type BP_STATE_ONESHOT: int
@cvar BP_STATE_ONESHOT: Breakpoint is enabled for one shot.
@type BP_STATE_RUNNING: int
@cvar BP_STATE_RUNNING: Breakpoint is running (recently hit).

@type BP_BREAK_ON_EXECUTION: int
@cvar BP_BREAK_ON_EXECUTION: Break on code execution.
@type BP_BREAK_ON_WRITE: int
@cvar BP_BREAK_ON_WRITE: Break on memory write.
@type BP_BREAK_ON_ACCESS: int
@cvar BP_BREAK_ON_ACCESS: Break on memory read or write.

### Function: __init__(self, address, size, condition, action)

**Description:** Breakpoint object.

@type  address: int
@param address: Memory address for breakpoint.

@type  size: int
@param size: Size of breakpoint in bytes (defaults to 1).

@type  condition: function
@param condition: (Optional) Condition callback function.

    The callback signature is::

        def condition_callback(event):
            return True     # returns True or False

    Where B{event} is an L{Event} object,
    and the return value is a boolean
    (C{True} to dispatch the event, C{False} otherwise).

@type  action: function
@param action: (Optional) Action callback function.
    If specified, the event is handled by this callback instead of
    being dispatched normally.

    The callback signature is::

        def action_callback(event):
            pass        # no return value

    Where B{event} is an L{Event} object.

### Function: __repr__(self)

### Function: is_disabled(self)

**Description:** @rtype:  bool
@return: C{True} if the breakpoint is in L{DISABLED} state.

### Function: is_enabled(self)

**Description:** @rtype:  bool
@return: C{True} if the breakpoint is in L{ENABLED} state.

### Function: is_one_shot(self)

**Description:** @rtype:  bool
@return: C{True} if the breakpoint is in L{ONESHOT} state.

### Function: is_running(self)

**Description:** @rtype:  bool
@return: C{True} if the breakpoint is in L{RUNNING} state.

### Function: is_here(self, address)

**Description:** @rtype:  bool
@return: C{True} if the address is within the range of the breakpoint.

### Function: get_address(self)

**Description:** @rtype:  int
@return: The target memory address for the breakpoint.

### Function: get_size(self)

**Description:** @rtype:  int
@return: The size in bytes of the breakpoint.

### Function: get_span(self)

**Description:** @rtype:  tuple( int, int )
@return:
    Starting and ending address of the memory range
    covered by the breakpoint.

### Function: get_state(self)

**Description:** @rtype:  int
@return: The current state of the breakpoint
    (L{DISABLED}, L{ENABLED}, L{ONESHOT}, L{RUNNING}).

### Function: get_state_name(self)

**Description:** @rtype:  str
@return: The name of the current state of the breakpoint.

### Function: is_conditional(self)

**Description:** @see: L{__init__}
@rtype:  bool
@return: C{True} if the breakpoint has a condition callback defined.

### Function: is_unconditional(self)

**Description:** @rtype:  bool
@return: C{True} if the breakpoint doesn't have a condition callback defined.

### Function: get_condition(self)

**Description:** @rtype:  bool, function
@return: Returns the condition callback for conditional breakpoints.
    Returns C{True} for unconditional breakpoints.

### Function: set_condition(self, condition)

**Description:** Sets a new condition callback for the breakpoint.

@see: L{__init__}

@type  condition: function
@param condition: (Optional) Condition callback function.

### Function: eval_condition(self, event)

**Description:** Evaluates the breakpoint condition, if any was set.

@type  event: L{Event}
@param event: Debug event triggered by the breakpoint.

@rtype:  bool
@return: C{True} to dispatch the event, C{False} otherwise.

### Function: is_automatic(self)

**Description:** @rtype:  bool
@return: C{True} if the breakpoint has an action callback defined.

### Function: is_interactive(self)

**Description:** @rtype:  bool
@return:
    C{True} if the breakpoint doesn't have an action callback defined.

### Function: get_action(self)

**Description:** @rtype:  bool, function
@return: Returns the action callback for automatic breakpoints.
    Returns C{None} for interactive breakpoints.

### Function: set_action(self, action)

**Description:** Sets a new action callback for the breakpoint.

@type  action: function
@param action: (Optional) Action callback function.

### Function: run_action(self, event)

**Description:** Executes the breakpoint action callback, if any was set.

@type  event: L{Event}
@param event: Debug event triggered by the breakpoint.

### Function: __bad_transition(self, state)

**Description:** Raises an C{AssertionError} exception for an invalid state transition.

@see: L{stateNames}

@type  state: int
@param state: Intended breakpoint state.

@raise Exception: Always.

### Function: disable(self, aProcess, aThread)

**Description:** Transition to L{DISABLED} state.
  - When hit: OneShot S{->} Disabled
  - Forced by user: Enabled, OneShot, Running S{->} Disabled
  - Transition from running state may require special handling
    by the breakpoint implementation class.

@type  aProcess: L{Process}
@param aProcess: Process object.

@type  aThread: L{Thread}
@param aThread: Thread object.

### Function: enable(self, aProcess, aThread)

**Description:** Transition to L{ENABLED} state.
  - When hit: Running S{->} Enabled
  - Forced by user: Disabled, Running S{->} Enabled
  - Transition from running state may require special handling
    by the breakpoint implementation class.

@type  aProcess: L{Process}
@param aProcess: Process object.

@type  aThread: L{Thread}
@param aThread: Thread object.

### Function: one_shot(self, aProcess, aThread)

**Description:** Transition to L{ONESHOT} state.
  - Forced by user: Disabled S{->} OneShot

@type  aProcess: L{Process}
@param aProcess: Process object.

@type  aThread: L{Thread}
@param aThread: Thread object.

### Function: running(self, aProcess, aThread)

**Description:** Transition to L{RUNNING} state.
  - When hit: Enabled S{->} Running

@type  aProcess: L{Process}
@param aProcess: Process object.

@type  aThread: L{Thread}
@param aThread: Thread object.

### Function: hit(self, event)

**Description:** Notify a breakpoint that it's been hit.

This triggers the corresponding state transition and sets the
C{breakpoint} property of the given L{Event} object.

@see: L{disable}, L{enable}, L{one_shot}, L{running}

@type  event: L{Event}
@param event: Debug event to handle (depends on the breakpoint type).

@raise AssertionError: Disabled breakpoints can't be hit.

### Function: __init__(self, address, condition, action)

**Description:** Code breakpoint object.

@see: L{Breakpoint.__init__}

@type  address: int
@param address: Memory address for breakpoint.

@type  condition: function
@param condition: (Optional) Condition callback function.

@type  action: function
@param action: (Optional) Action callback function.

### Function: __set_bp(self, aProcess)

**Description:** Writes a breakpoint instruction at the target address.

@type  aProcess: L{Process}
@param aProcess: Process object.

### Function: __clear_bp(self, aProcess)

**Description:** Restores the original byte at the target address.

@type  aProcess: L{Process}
@param aProcess: Process object.

### Function: disable(self, aProcess, aThread)

### Function: enable(self, aProcess, aThread)

### Function: one_shot(self, aProcess, aThread)

### Function: running(self, aProcess, aThread)

### Function: __init__(self, address, pages, condition, action)

**Description:** Page breakpoint object.

@see: L{Breakpoint.__init__}

@type  address: int
@param address: Memory address for breakpoint.

@type  pages: int
@param address: Size of breakpoint in pages.

@type  condition: function
@param condition: (Optional) Condition callback function.

@type  action: function
@param action: (Optional) Action callback function.

### Function: get_size_in_pages(self)

**Description:** @rtype:  int
@return: The size in pages of the breakpoint.

### Function: __set_bp(self, aProcess)

**Description:** Sets the target pages as guard pages.

@type  aProcess: L{Process}
@param aProcess: Process object.

### Function: __clear_bp(self, aProcess)

**Description:** Restores the original permissions of the target pages.

@type  aProcess: L{Process}
@param aProcess: Process object.

### Function: disable(self, aProcess, aThread)

### Function: enable(self, aProcess, aThread)

### Function: one_shot(self, aProcess, aThread)

### Function: running(self, aProcess, aThread)

### Function: __init__(self, address, triggerFlag, sizeFlag, condition, action)

**Description:** Hardware breakpoint object.

@see: L{Breakpoint.__init__}

@type  address: int
@param address: Memory address for breakpoint.

@type  triggerFlag: int
@param triggerFlag: Trigger of breakpoint. Must be one of the following:

     - L{BREAK_ON_EXECUTION}

       Break on code execution.

     - L{BREAK_ON_WRITE}

       Break on memory read or write.

     - L{BREAK_ON_ACCESS}

       Break on memory write.

@type  sizeFlag: int
@param sizeFlag: Size of breakpoint. Must be one of the following:

     - L{WATCH_BYTE}

       One (1) byte in size.

     - L{WATCH_WORD}

       Two (2) bytes in size.

     - L{WATCH_DWORD}

       Four (4) bytes in size.

     - L{WATCH_QWORD}

       Eight (8) bytes in size.

@type  condition: function
@param condition: (Optional) Condition callback function.

@type  action: function
@param action: (Optional) Action callback function.

### Function: __clear_bp(self, aThread)

**Description:** Clears this breakpoint from the debug registers.

@type  aThread: L{Thread}
@param aThread: Thread object.

### Function: __set_bp(self, aThread)

**Description:** Sets this breakpoint in the debug registers.

@type  aThread: L{Thread}
@param aThread: Thread object.

### Function: get_slot(self)

**Description:** @rtype:  int
@return: The debug register number used by this breakpoint,
    or C{None} if the breakpoint is not active.

### Function: get_trigger(self)

**Description:** @see: L{validTriggers}
@rtype:  int
@return: The breakpoint trigger flag.

### Function: get_watch(self)

**Description:** @see: L{validWatchSizes}
@rtype:  int
@return: The breakpoint watch flag.

### Function: disable(self, aProcess, aThread)

### Function: enable(self, aProcess, aThread)

### Function: one_shot(self, aProcess, aThread)

### Function: running(self, aProcess, aThread)

### Function: __new__(cls)

### Function: __init__(self, preCB, postCB, paramCount, signature, arch)

**Description:** @type  preCB: function
@param preCB: (Optional) Callback triggered on function entry.

    The signature for the callback should be something like this::

        def pre_LoadLibraryEx(event, ra, lpFilename, hFile, dwFlags):

            # return address
            ra = params[0]

            # function arguments start from here...
            szFilename = event.get_process().peek_string(lpFilename)

            # (...)

    Note that all pointer types are treated like void pointers, so your
    callback won't get the string or structure pointed to by it, but
    the remote memory address instead. This is so to prevent the ctypes
    library from being "too helpful" and trying to dereference the
    pointer. To get the actual data being pointed to, use one of the
    L{Process.read} methods.

@type  postCB: function
@param postCB: (Optional) Callback triggered on function exit.

    The signature for the callback should be something like this::

        def post_LoadLibraryEx(event, return_value):

            # (...)

@type  paramCount: int
@param paramCount:
    (Optional) Number of parameters for the C{preCB} callback,
    not counting the return address. Parameters are read from
    the stack and assumed to be DWORDs in 32 bits and QWORDs in 64.

    This is a faster way to pull stack parameters in 32 bits, but in 64
    bits (or with some odd APIs in 32 bits) it won't be useful, since
    not all arguments to the hooked function will be of the same size.

    For a more reliable and cross-platform way of hooking use the
    C{signature} argument instead.

@type  signature: tuple
@param signature:
    (Optional) Tuple of C{ctypes} data types that constitute the
    hooked function signature. When the function is called, this will
    be used to parse the arguments from the stack. Overrides the
    C{paramCount} argument.

@type  arch: str
@param arch: (Optional) Target architecture. Defaults to the current
    architecture. See: L{win32.arch}

### Function: _cast_signature_pointers_to_void(self, signature)

### Function: _calc_signature(self, signature)

### Function: _get_return_address(self, aProcess, aThread)

### Function: _get_function_arguments(self, aProcess, aThread)

### Function: _get_return_value(self, aThread)

### Function: __call__(self, event)

**Description:** Handles the breakpoint event on entry of the function.

@type  event: L{ExceptionEvent}
@param event: Breakpoint hit event.

@raise WindowsError: An error occured.

### Function: __postCallAction_hwbp(self, event)

**Description:** Handles hardware breakpoint events on return from the function.

@type  event: L{ExceptionEvent}
@param event: Single step event.

### Function: __postCallAction_codebp(self, event)

**Description:** Handles code breakpoint events on return from the function.

@type  event: L{ExceptionEvent}
@param event: Breakpoint hit event.

### Function: __postCallAction(self, event)

**Description:** Calls the "post" callback.

@type  event: L{ExceptionEvent}
@param event: Breakpoint hit event.

### Function: __callHandler(self, callback, event)

**Description:** Calls a "pre" or "post" handler, if set.

@type  callback: function
@param callback: Callback function to call.

@type  event: L{ExceptionEvent}
@param event: Breakpoint hit event.

@type  params: tuple
@param params: Parameters for the callback function.

### Function: __push_params(self, tid, params)

**Description:** Remembers the arguments tuple for the last call to the hooked function
from this thread.

@type  tid: int
@param tid: Thread global ID.

@type  params: tuple( arg, arg, arg... )
@param params: Tuple of arguments.

### Function: __pop_params(self, tid)

**Description:** Forgets the arguments tuple for the last call to the hooked function
from this thread.

@type  tid: int
@param tid: Thread global ID.

### Function: get_params(self, tid)

**Description:** Returns the parameters found in the stack when the hooked function
was last called by this thread.

@type  tid: int
@param tid: Thread global ID.

@rtype:  tuple( arg, arg, arg... )
@return: Tuple of arguments.

### Function: get_params_stack(self, tid)

**Description:** Returns the parameters found in the stack each time the hooked function
was called by this thread and hasn't returned yet.

@type  tid: int
@param tid: Thread global ID.

@rtype:  list of tuple( arg, arg, arg... )
@return: List of argument tuples.

### Function: hook(self, debug, pid, address)

**Description:** Installs the function hook at a given process and address.

@see: L{unhook}

@warning: Do not call from an function hook callback.

@type  debug: L{Debug}
@param debug: Debug object.

@type  pid: int
@param pid: Process ID.

@type  address: int
@param address: Function address.

### Function: unhook(self, debug, pid, address)

**Description:** Removes the function hook at a given process and address.

@see: L{hook}

@warning: Do not call from an function hook callback.

@type  debug: L{Debug}
@param debug: Debug object.

@type  pid: int
@param pid: Process ID.

@type  address: int
@param address: Function address.

### Function: _calc_signature(self, signature)

### Function: _get_return_address(self, aProcess, aThread)

### Function: _get_function_arguments(self, aProcess, aThread)

### Function: _get_return_value(self, aThread)

### Function: _calc_signature(self, signature)

### Function: _get_return_address(self, aProcess, aThread)

### Function: _get_function_arguments(self, aProcess, aThread)

### Function: _get_arguments_from_buffer(self, buffer, structure)

### Function: _get_return_value(self, aThread)

### Function: __init__(self, eventHandler, modName, procName, paramCount, signature)

**Description:** @type  eventHandler: L{EventHandler}
@param eventHandler: Event handler instance. This is where the hook
    callbacks are to be defined (see below).

@type  modName: str
@param modName: Module name.

@type  procName: str
@param procName: Procedure name.
    The pre and post callbacks will be deduced from it.

    For example, if the procedure is "LoadLibraryEx" the callback
    routines will be "pre_LoadLibraryEx" and "post_LoadLibraryEx".

    The signature for the callbacks should be something like this::

        def pre_LoadLibraryEx(self, event, ra, lpFilename, hFile, dwFlags):

            # return address
            ra = params[0]

            # function arguments start from here...
            szFilename = event.get_process().peek_string(lpFilename)

            # (...)

        def post_LoadLibraryEx(self, event, return_value):

            # (...)

    Note that all pointer types are treated like void pointers, so your
    callback won't get the string or structure pointed to by it, but
    the remote memory address instead. This is so to prevent the ctypes
    library from being "too helpful" and trying to dereference the
    pointer. To get the actual data being pointed to, use one of the
    L{Process.read} methods.

@type  paramCount: int
@param paramCount:
    (Optional) Number of parameters for the C{preCB} callback,
    not counting the return address. Parameters are read from
    the stack and assumed to be DWORDs in 32 bits and QWORDs in 64.

    This is a faster way to pull stack parameters in 32 bits, but in 64
    bits (or with some odd APIs in 32 bits) it won't be useful, since
    not all arguments to the hooked function will be of the same size.

    For a more reliable and cross-platform way of hooking use the
    C{signature} argument instead.

@type  signature: tuple
@param signature:
    (Optional) Tuple of C{ctypes} data types that constitute the
    hooked function signature. When the function is called, this will
    be used to parse the arguments from the stack. Overrides the
    C{paramCount} argument.

### Function: __call__(self, event)

**Description:** Handles the breakpoint event on entry of the function.

@type  event: L{ExceptionEvent}
@param event: Breakpoint hit event.

@raise WindowsError: An error occured.

### Function: modName(self)

### Function: procName(self)

### Function: hook(self, debug, pid)

**Description:** Installs the API hook on a given process and module.

@warning: Do not call from an API hook callback.

@type  debug: L{Debug}
@param debug: Debug object.

@type  pid: int
@param pid: Process ID.

### Function: unhook(self, debug, pid)

**Description:** Removes the API hook from the given process and module.

@warning: Do not call from an API hook callback.

@type  debug: L{Debug}
@param debug: Debug object.

@type  pid: int
@param pid: Process ID.

### Function: __init__(self, pid, start, end, action, oneshot)

### Function: pid(self)

### Function: start(self)

### Function: end(self)

### Function: action(self)

### Function: oneshot(self)

### Function: match(self, address)

**Description:** Determine if the given memory address lies within the watched buffer.

@rtype: bool
@return: C{True} if the given memory address lies within the watched
    buffer, C{False} otherwise.

### Function: __init__(self)

### Function: add(self, bw)

**Description:** Adds a buffer watch identifier.

@type  bw: L{BufferWatch}
@param bw:
    Buffer watch identifier.

### Function: remove(self, bw)

**Description:** Removes a buffer watch identifier.

@type  bw: L{BufferWatch}
@param bw:
    Buffer watch identifier.

@raise KeyError: The buffer watch identifier was already removed.

### Function: remove_last_match(self, address, size)

**Description:** Removes the last buffer from the watch object
to match the given address and size.

@type  address: int
@param address: Memory address of buffer to stop watching.

@type  size: int
@param size: Size in bytes of buffer to stop watching.

@rtype:  int
@return: Number of matching elements found. Only the last one to be
    added is actually deleted upon calling this method.

    This counter allows you to know if there are more matching elements
    and how many.

### Function: count(self)

**Description:** @rtype:  int
@return: Number of buffers being watched.

### Function: __call__(self, event)

**Description:** Breakpoint condition callback.

This method will also call the action callbacks for each
buffer being watched.

@type  event: L{ExceptionEvent}
@param event: Guard page exception event.

@rtype:  bool
@return: C{True} if the address being accessed belongs
    to at least one of the buffers that was being watched
    and had no action callback.

### Function: __init__(self)

### Function: __get_running_bp_set(self, tid)

**Description:** Auxiliary method.

### Function: __add_running_bp(self, tid, bp)

**Description:** Auxiliary method.

### Function: __del_running_bp(self, tid, bp)

**Description:** Auxiliary method.

### Function: __del_running_bp_from_all_threads(self, bp)

**Description:** Auxiliary method.

### Function: __cleanup_breakpoint(self, event, bp)

**Description:** Auxiliary method.

### Function: __cleanup_thread(self, event)

**Description:** Auxiliary method for L{_notify_exit_thread}
and L{_notify_exit_process}.

### Function: __cleanup_process(self, event)

**Description:** Auxiliary method for L{_notify_exit_process}.

### Function: __cleanup_module(self, event)

**Description:** Auxiliary method for L{_notify_unload_dll}.

### Function: define_code_breakpoint(self, dwProcessId, address, condition, action)

**Description:** Creates a disabled code breakpoint at the given address.

@see:
    L{has_code_breakpoint},
    L{get_code_breakpoint},
    L{enable_code_breakpoint},
    L{enable_one_shot_code_breakpoint},
    L{disable_code_breakpoint},
    L{erase_code_breakpoint}

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address of the code instruction to break at.

@type  condition: function
@param condition: (Optional) Condition callback function.

    The callback signature is::

        def condition_callback(event):
            return True     # returns True or False

    Where B{event} is an L{Event} object,
    and the return value is a boolean
    (C{True} to dispatch the event, C{False} otherwise).

@type  action: function
@param action: (Optional) Action callback function.
    If specified, the event is handled by this callback instead of
    being dispatched normally.

    The callback signature is::

        def action_callback(event):
            pass        # no return value

    Where B{event} is an L{Event} object,
    and the return value is a boolean
    (C{True} to dispatch the event, C{False} otherwise).

@rtype:  L{CodeBreakpoint}
@return: The code breakpoint object.

### Function: define_page_breakpoint(self, dwProcessId, address, pages, condition, action)

**Description:** Creates a disabled page breakpoint at the given address.

@see:
    L{has_page_breakpoint},
    L{get_page_breakpoint},
    L{enable_page_breakpoint},
    L{enable_one_shot_page_breakpoint},
    L{disable_page_breakpoint},
    L{erase_page_breakpoint}

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address of the first page to watch.

@type  pages: int
@param pages: Number of pages to watch.

@type  condition: function
@param condition: (Optional) Condition callback function.

    The callback signature is::

        def condition_callback(event):
            return True     # returns True or False

    Where B{event} is an L{Event} object,
    and the return value is a boolean
    (C{True} to dispatch the event, C{False} otherwise).

@type  action: function
@param action: (Optional) Action callback function.
    If specified, the event is handled by this callback instead of
    being dispatched normally.

    The callback signature is::

        def action_callback(event):
            pass        # no return value

    Where B{event} is an L{Event} object,
    and the return value is a boolean
    (C{True} to dispatch the event, C{False} otherwise).

@rtype:  L{PageBreakpoint}
@return: The page breakpoint object.

### Function: define_hardware_breakpoint(self, dwThreadId, address, triggerFlag, sizeFlag, condition, action)

**Description:** Creates a disabled hardware breakpoint at the given address.

@see:
    L{has_hardware_breakpoint},
    L{get_hardware_breakpoint},
    L{enable_hardware_breakpoint},
    L{enable_one_shot_hardware_breakpoint},
    L{disable_hardware_breakpoint},
    L{erase_hardware_breakpoint}

@note:
    Hardware breakpoints do not seem to work properly on VirtualBox.
    See U{http://www.virtualbox.org/ticket/477}.

@type  dwThreadId: int
@param dwThreadId: Thread global ID.

@type  address: int
@param address: Memory address to watch.

@type  triggerFlag: int
@param triggerFlag: Trigger of breakpoint. Must be one of the following:

     - L{BP_BREAK_ON_EXECUTION}

       Break on code execution.

     - L{BP_BREAK_ON_WRITE}

       Break on memory read or write.

     - L{BP_BREAK_ON_ACCESS}

       Break on memory write.

@type  sizeFlag: int
@param sizeFlag: Size of breakpoint. Must be one of the following:

     - L{BP_WATCH_BYTE}

       One (1) byte in size.

     - L{BP_WATCH_WORD}

       Two (2) bytes in size.

     - L{BP_WATCH_DWORD}

       Four (4) bytes in size.

     - L{BP_WATCH_QWORD}

       Eight (8) bytes in size.

@type  condition: function
@param condition: (Optional) Condition callback function.

    The callback signature is::

        def condition_callback(event):
            return True     # returns True or False

    Where B{event} is an L{Event} object,
    and the return value is a boolean
    (C{True} to dispatch the event, C{False} otherwise).

@type  action: function
@param action: (Optional) Action callback function.
    If specified, the event is handled by this callback instead of
    being dispatched normally.

    The callback signature is::

        def action_callback(event):
            pass        # no return value

    Where B{event} is an L{Event} object,
    and the return value is a boolean
    (C{True} to dispatch the event, C{False} otherwise).

@rtype:  L{HardwareBreakpoint}
@return: The hardware breakpoint object.

### Function: has_code_breakpoint(self, dwProcessId, address)

**Description:** Checks if a code breakpoint is defined at the given address.

@see:
    L{define_code_breakpoint},
    L{get_code_breakpoint},
    L{erase_code_breakpoint},
    L{enable_code_breakpoint},
    L{enable_one_shot_code_breakpoint},
    L{disable_code_breakpoint}

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address of breakpoint.

@rtype:  bool
@return: C{True} if the breakpoint is defined, C{False} otherwise.

### Function: has_page_breakpoint(self, dwProcessId, address)

**Description:** Checks if a page breakpoint is defined at the given address.

@see:
    L{define_page_breakpoint},
    L{get_page_breakpoint},
    L{erase_page_breakpoint},
    L{enable_page_breakpoint},
    L{enable_one_shot_page_breakpoint},
    L{disable_page_breakpoint}

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address of breakpoint.

@rtype:  bool
@return: C{True} if the breakpoint is defined, C{False} otherwise.

### Function: has_hardware_breakpoint(self, dwThreadId, address)

**Description:** Checks if a hardware breakpoint is defined at the given address.

@see:
    L{define_hardware_breakpoint},
    L{get_hardware_breakpoint},
    L{erase_hardware_breakpoint},
    L{enable_hardware_breakpoint},
    L{enable_one_shot_hardware_breakpoint},
    L{disable_hardware_breakpoint}

@type  dwThreadId: int
@param dwThreadId: Thread global ID.

@type  address: int
@param address: Memory address of breakpoint.

@rtype:  bool
@return: C{True} if the breakpoint is defined, C{False} otherwise.

### Function: get_code_breakpoint(self, dwProcessId, address)

**Description:** Returns the internally used breakpoint object,
for the code breakpoint defined at the given address.

@warning: It's usually best to call the L{Debug} methods
    instead of accessing the breakpoint objects directly.

@see:
    L{define_code_breakpoint},
    L{has_code_breakpoint},
    L{enable_code_breakpoint},
    L{enable_one_shot_code_breakpoint},
    L{disable_code_breakpoint},
    L{erase_code_breakpoint}

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address where the breakpoint is defined.

@rtype:  L{CodeBreakpoint}
@return: The code breakpoint object.

### Function: get_page_breakpoint(self, dwProcessId, address)

**Description:** Returns the internally used breakpoint object,
for the page breakpoint defined at the given address.

@warning: It's usually best to call the L{Debug} methods
    instead of accessing the breakpoint objects directly.

@see:
    L{define_page_breakpoint},
    L{has_page_breakpoint},
    L{enable_page_breakpoint},
    L{enable_one_shot_page_breakpoint},
    L{disable_page_breakpoint},
    L{erase_page_breakpoint}

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address where the breakpoint is defined.

@rtype:  L{PageBreakpoint}
@return: The page breakpoint object.

### Function: get_hardware_breakpoint(self, dwThreadId, address)

**Description:** Returns the internally used breakpoint object,
for the code breakpoint defined at the given address.

@warning: It's usually best to call the L{Debug} methods
    instead of accessing the breakpoint objects directly.

@see:
    L{define_hardware_breakpoint},
    L{has_hardware_breakpoint},
    L{get_code_breakpoint},
    L{enable_hardware_breakpoint},
    L{enable_one_shot_hardware_breakpoint},
    L{disable_hardware_breakpoint},
    L{erase_hardware_breakpoint}

@type  dwThreadId: int
@param dwThreadId: Thread global ID.

@type  address: int
@param address: Memory address where the breakpoint is defined.

@rtype:  L{HardwareBreakpoint}
@return: The hardware breakpoint object.

### Function: enable_code_breakpoint(self, dwProcessId, address)

**Description:** Enables the code breakpoint at the given address.

@see:
    L{define_code_breakpoint},
    L{has_code_breakpoint},
    L{enable_one_shot_code_breakpoint},
    L{disable_code_breakpoint}
    L{erase_code_breakpoint},

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address of breakpoint.

### Function: enable_page_breakpoint(self, dwProcessId, address)

**Description:** Enables the page breakpoint at the given address.

@see:
    L{define_page_breakpoint},
    L{has_page_breakpoint},
    L{get_page_breakpoint},
    L{enable_one_shot_page_breakpoint},
    L{disable_page_breakpoint}
    L{erase_page_breakpoint},

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address of breakpoint.

### Function: enable_hardware_breakpoint(self, dwThreadId, address)

**Description:** Enables the hardware breakpoint at the given address.

@see:
    L{define_hardware_breakpoint},
    L{has_hardware_breakpoint},
    L{get_hardware_breakpoint},
    L{enable_one_shot_hardware_breakpoint},
    L{disable_hardware_breakpoint}
    L{erase_hardware_breakpoint},

@note: Do not set hardware breakpoints while processing the system
    breakpoint event.

@type  dwThreadId: int
@param dwThreadId: Thread global ID.

@type  address: int
@param address: Memory address of breakpoint.

### Function: enable_one_shot_code_breakpoint(self, dwProcessId, address)

**Description:** Enables the code breakpoint at the given address for only one shot.

@see:
    L{define_code_breakpoint},
    L{has_code_breakpoint},
    L{get_code_breakpoint},
    L{enable_code_breakpoint},
    L{disable_code_breakpoint}
    L{erase_code_breakpoint},

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address of breakpoint.

### Function: enable_one_shot_page_breakpoint(self, dwProcessId, address)

**Description:** Enables the page breakpoint at the given address for only one shot.

@see:
    L{define_page_breakpoint},
    L{has_page_breakpoint},
    L{get_page_breakpoint},
    L{enable_page_breakpoint},
    L{disable_page_breakpoint}
    L{erase_page_breakpoint},

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address of breakpoint.

### Function: enable_one_shot_hardware_breakpoint(self, dwThreadId, address)

**Description:** Enables the hardware breakpoint at the given address for only one shot.

@see:
    L{define_hardware_breakpoint},
    L{has_hardware_breakpoint},
    L{get_hardware_breakpoint},
    L{enable_hardware_breakpoint},
    L{disable_hardware_breakpoint}
    L{erase_hardware_breakpoint},

@type  dwThreadId: int
@param dwThreadId: Thread global ID.

@type  address: int
@param address: Memory address of breakpoint.

### Function: disable_code_breakpoint(self, dwProcessId, address)

**Description:** Disables the code breakpoint at the given address.

@see:
    L{define_code_breakpoint},
    L{has_code_breakpoint},
    L{get_code_breakpoint},
    L{enable_code_breakpoint}
    L{enable_one_shot_code_breakpoint},
    L{erase_code_breakpoint},

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address of breakpoint.

### Function: disable_page_breakpoint(self, dwProcessId, address)

**Description:** Disables the page breakpoint at the given address.

@see:
    L{define_page_breakpoint},
    L{has_page_breakpoint},
    L{get_page_breakpoint},
    L{enable_page_breakpoint}
    L{enable_one_shot_page_breakpoint},
    L{erase_page_breakpoint},

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address of breakpoint.

### Function: disable_hardware_breakpoint(self, dwThreadId, address)

**Description:** Disables the hardware breakpoint at the given address.

@see:
    L{define_hardware_breakpoint},
    L{has_hardware_breakpoint},
    L{get_hardware_breakpoint},
    L{enable_hardware_breakpoint}
    L{enable_one_shot_hardware_breakpoint},
    L{erase_hardware_breakpoint},

@type  dwThreadId: int
@param dwThreadId: Thread global ID.

@type  address: int
@param address: Memory address of breakpoint.

### Function: erase_code_breakpoint(self, dwProcessId, address)

**Description:** Erases the code breakpoint at the given address.

@see:
    L{define_code_breakpoint},
    L{has_code_breakpoint},
    L{get_code_breakpoint},
    L{enable_code_breakpoint},
    L{enable_one_shot_code_breakpoint},
    L{disable_code_breakpoint}

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address of breakpoint.

### Function: erase_page_breakpoint(self, dwProcessId, address)

**Description:** Erases the page breakpoint at the given address.

@see:
    L{define_page_breakpoint},
    L{has_page_breakpoint},
    L{get_page_breakpoint},
    L{enable_page_breakpoint},
    L{enable_one_shot_page_breakpoint},
    L{disable_page_breakpoint}

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@type  address: int
@param address: Memory address of breakpoint.

### Function: erase_hardware_breakpoint(self, dwThreadId, address)

**Description:** Erases the hardware breakpoint at the given address.

@see:
    L{define_hardware_breakpoint},
    L{has_hardware_breakpoint},
    L{get_hardware_breakpoint},
    L{enable_hardware_breakpoint},
    L{enable_one_shot_hardware_breakpoint},
    L{disable_hardware_breakpoint}

@type  dwThreadId: int
@param dwThreadId: Thread global ID.

@type  address: int
@param address: Memory address of breakpoint.

### Function: get_all_breakpoints(self)

**Description:** Returns all breakpoint objects as a list of tuples.

Each tuple contains:
 - Process global ID to which the breakpoint applies.
 - Thread global ID to which the breakpoint applies, or C{None}.
 - The L{Breakpoint} object itself.

@note: If you're only interested in a specific breakpoint type, or in
    breakpoints for a specific process or thread, it's probably faster
    to call one of the following methods:
     - L{get_all_code_breakpoints}
     - L{get_all_page_breakpoints}
     - L{get_all_hardware_breakpoints}
     - L{get_process_code_breakpoints}
     - L{get_process_page_breakpoints}
     - L{get_process_hardware_breakpoints}
     - L{get_thread_hardware_breakpoints}

@rtype:  list of tuple( pid, tid, bp )
@return: List of all breakpoints.

### Function: get_all_code_breakpoints(self)

**Description:** @rtype:  list of tuple( int, L{CodeBreakpoint} )
@return: All code breakpoints as a list of tuples (pid, bp).

### Function: get_all_page_breakpoints(self)

**Description:** @rtype:  list of tuple( int, L{PageBreakpoint} )
@return: All page breakpoints as a list of tuples (pid, bp).

### Function: get_all_hardware_breakpoints(self)

**Description:** @rtype:  list of tuple( int, L{HardwareBreakpoint} )
@return: All hardware breakpoints as a list of tuples (tid, bp).

### Function: get_process_breakpoints(self, dwProcessId)

**Description:** Returns all breakpoint objects for the given process as a list of tuples.

Each tuple contains:
 - Process global ID to which the breakpoint applies.
 - Thread global ID to which the breakpoint applies, or C{None}.
 - The L{Breakpoint} object itself.

@note: If you're only interested in a specific breakpoint type, or in
    breakpoints for a specific process or thread, it's probably faster
    to call one of the following methods:
     - L{get_all_code_breakpoints}
     - L{get_all_page_breakpoints}
     - L{get_all_hardware_breakpoints}
     - L{get_process_code_breakpoints}
     - L{get_process_page_breakpoints}
     - L{get_process_hardware_breakpoints}
     - L{get_thread_hardware_breakpoints}

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@rtype:  list of tuple( pid, tid, bp )
@return: List of all breakpoints for the given process.

### Function: get_process_code_breakpoints(self, dwProcessId)

**Description:** @type  dwProcessId: int
@param dwProcessId: Process global ID.

@rtype:  list of L{CodeBreakpoint}
@return: All code breakpoints for the given process.

### Function: get_process_page_breakpoints(self, dwProcessId)

**Description:** @type  dwProcessId: int
@param dwProcessId: Process global ID.

@rtype:  list of L{PageBreakpoint}
@return: All page breakpoints for the given process.

### Function: get_thread_hardware_breakpoints(self, dwThreadId)

**Description:** @see: L{get_process_hardware_breakpoints}

@type  dwThreadId: int
@param dwThreadId: Thread global ID.

@rtype:  list of L{HardwareBreakpoint}
@return: All hardware breakpoints for the given thread.

### Function: get_process_hardware_breakpoints(self, dwProcessId)

**Description:** @see: L{get_thread_hardware_breakpoints}

@type  dwProcessId: int
@param dwProcessId: Process global ID.

@rtype:  list of tuple( int, L{HardwareBreakpoint} )
@return: All hardware breakpoints for each thread in the given process
    as a list of tuples (tid, bp).

### Function: enable_all_breakpoints(self)

**Description:** Enables all disabled breakpoints in all processes.

@see:
    enable_code_breakpoint,
    enable_page_breakpoint,
    enable_hardware_breakpoint

### Function: enable_one_shot_all_breakpoints(self)

**Description:** Enables for one shot all disabled breakpoints in all processes.

@see:
    enable_one_shot_code_breakpoint,
    enable_one_shot_page_breakpoint,
    enable_one_shot_hardware_breakpoint

### Function: disable_all_breakpoints(self)

**Description:** Disables all breakpoints in all processes.

@see:
    disable_code_breakpoint,
    disable_page_breakpoint,
    disable_hardware_breakpoint

### Function: erase_all_breakpoints(self)

**Description:** Erases all breakpoints in all processes.

@see:
    erase_code_breakpoint,
    erase_page_breakpoint,
    erase_hardware_breakpoint

### Function: enable_process_breakpoints(self, dwProcessId)

**Description:** Enables all disabled breakpoints for the given process.

@type  dwProcessId: int
@param dwProcessId: Process global ID.

### Function: enable_one_shot_process_breakpoints(self, dwProcessId)

**Description:** Enables for one shot all disabled breakpoints for the given process.

@type  dwProcessId: int
@param dwProcessId: Process global ID.

### Function: disable_process_breakpoints(self, dwProcessId)

**Description:** Disables all breakpoints for the given process.

@type  dwProcessId: int
@param dwProcessId: Process global ID.

### Function: erase_process_breakpoints(self, dwProcessId)

**Description:** Erases all breakpoints for the given process.

@type  dwProcessId: int
@param dwProcessId: Process global ID.

### Function: _notify_guard_page(self, event)

**Description:** Notify breakpoints of a guard page exception event.

@type  event: L{ExceptionEvent}
@param event: Guard page exception event.

@rtype:  bool
@return: C{True} to call the user-defined handle, C{False} otherwise.

### Function: _notify_breakpoint(self, event)

**Description:** Notify breakpoints of a breakpoint exception event.

@type  event: L{ExceptionEvent}
@param event: Breakpoint exception event.

@rtype:  bool
@return: C{True} to call the user-defined handle, C{False} otherwise.

### Function: _notify_single_step(self, event)

**Description:** Notify breakpoints of a single step exception event.

@type  event: L{ExceptionEvent}
@param event: Single step exception event.

@rtype:  bool
@return: C{True} to call the user-defined handle, C{False} otherwise.

### Function: _notify_load_dll(self, event)

**Description:** Notify the loading of a DLL.

@type  event: L{LoadDLLEvent}
@param event: Load DLL event.

@rtype:  bool
@return: C{True} to call the user-defined handler, C{False} otherwise.

### Function: _notify_unload_dll(self, event)

**Description:** Notify the unloading of a DLL.

@type  event: L{UnloadDLLEvent}
@param event: Unload DLL event.

@rtype:  bool
@return: C{True} to call the user-defined handler, C{False} otherwise.

### Function: _notify_exit_thread(self, event)

**Description:** Notify the termination of a thread.

@type  event: L{ExitThreadEvent}
@param event: Exit thread event.

@rtype:  bool
@return: C{True} to call the user-defined handler, C{False} otherwise.

### Function: _notify_exit_process(self, event)

**Description:** Notify the termination of a process.

@type  event: L{ExitProcessEvent}
@param event: Exit process event.

@rtype:  bool
@return: C{True} to call the user-defined handler, C{False} otherwise.

### Function: __set_break(self, pid, address, action, oneshot)

**Description:** Used by L{break_at} and L{stalk_at}.

@type  pid: int
@param pid: Process global ID.

@type  address: int or str
@param address:
    Memory address of code instruction to break at. It can be an
    integer value for the actual address or a string with a label
    to be resolved.

@type  action: function
@param action: (Optional) Action callback function.

    See L{define_code_breakpoint} for more details.

@type  oneshot: bool
@param oneshot: C{True} for one-shot breakpoints, C{False} otherwise.

@rtype:  L{Breakpoint}
@return: Returns the new L{Breakpoint} object, or C{None} if the label
    couldn't be resolved and the breakpoint was deferred. Deferred
    breakpoints are set when the DLL they point to is loaded.

### Function: __clear_break(self, pid, address)

**Description:** Used by L{dont_break_at} and L{dont_stalk_at}.

@type  pid: int
@param pid: Process global ID.

@type  address: int or str
@param address:
    Memory address of code instruction to break at. It can be an
    integer value for the actual address or a string with a label
    to be resolved.

### Function: __set_deferred_breakpoints(self, event)

**Description:** Used internally. Sets all deferred breakpoints for a DLL when it's
loaded.

@type  event: L{LoadDLLEvent}
@param event: Load DLL event.

### Function: get_all_deferred_code_breakpoints(self)

**Description:** Returns a list of deferred code breakpoints.

@rtype:  tuple of (int, str, callable, bool)
@return: Tuple containing the following elements:
     - Process ID where to set the breakpoint.
     - Label pointing to the address where to set the breakpoint.
     - Action callback for the breakpoint.
     - C{True} of the breakpoint is one-shot, C{False} otherwise.

### Function: get_process_deferred_code_breakpoints(self, dwProcessId)

**Description:** Returns a list of deferred code breakpoints.

@type  dwProcessId: int
@param dwProcessId: Process ID.

@rtype:  tuple of (int, str, callable, bool)
@return: Tuple containing the following elements:
     - Label pointing to the address where to set the breakpoint.
     - Action callback for the breakpoint.
     - C{True} of the breakpoint is one-shot, C{False} otherwise.

### Function: stalk_at(self, pid, address, action)

**Description:** Sets a one shot code breakpoint at the given process and address.

If instead of an address you pass a label, the breakpoint may be
deferred until the DLL it points to is loaded.

@see: L{break_at}, L{dont_stalk_at}

@type  pid: int
@param pid: Process global ID.

@type  address: int or str
@param address:
    Memory address of code instruction to break at. It can be an
    integer value for the actual address or a string with a label
    to be resolved.

@type  action: function
@param action: (Optional) Action callback function.

    See L{define_code_breakpoint} for more details.

@rtype:  bool
@return: C{True} if the breakpoint was set immediately, or C{False} if
    it was deferred.

### Function: break_at(self, pid, address, action)

**Description:** Sets a code breakpoint at the given process and address.

If instead of an address you pass a label, the breakpoint may be
deferred until the DLL it points to is loaded.

@see: L{stalk_at}, L{dont_break_at}

@type  pid: int
@param pid: Process global ID.

@type  address: int or str
@param address:
    Memory address of code instruction to break at. It can be an
    integer value for the actual address or a string with a label
    to be resolved.

@type  action: function
@param action: (Optional) Action callback function.

    See L{define_code_breakpoint} for more details.

@rtype:  bool
@return: C{True} if the breakpoint was set immediately, or C{False} if
    it was deferred.

### Function: dont_break_at(self, pid, address)

**Description:** Clears a code breakpoint set by L{break_at}.

@type  pid: int
@param pid: Process global ID.

@type  address: int or str
@param address:
    Memory address of code instruction to break at. It can be an
    integer value for the actual address or a string with a label
    to be resolved.

### Function: dont_stalk_at(self, pid, address)

**Description:** Clears a code breakpoint set by L{stalk_at}.

@type  pid: int
@param pid: Process global ID.

@type  address: int or str
@param address:
    Memory address of code instruction to break at. It can be an
    integer value for the actual address or a string with a label
    to be resolved.

### Function: hook_function(self, pid, address, preCB, postCB, paramCount, signature)

**Description:** Sets a function hook at the given address.

If instead of an address you pass a label, the hook may be
deferred until the DLL it points to is loaded.

@type  pid: int
@param pid: Process global ID.

@type  address: int or str
@param address:
    Memory address of code instruction to break at. It can be an
    integer value for the actual address or a string with a label
    to be resolved.

@type  preCB: function
@param preCB: (Optional) Callback triggered on function entry.

    The signature for the callback should be something like this::

        def pre_LoadLibraryEx(event, ra, lpFilename, hFile, dwFlags):

            # return address
            ra = params[0]

            # function arguments start from here...
            szFilename = event.get_process().peek_string(lpFilename)

            # (...)

    Note that all pointer types are treated like void pointers, so your
    callback won't get the string or structure pointed to by it, but
    the remote memory address instead. This is so to prevent the ctypes
    library from being "too helpful" and trying to dereference the
    pointer. To get the actual data being pointed to, use one of the
    L{Process.read} methods.

@type  postCB: function
@param postCB: (Optional) Callback triggered on function exit.

    The signature for the callback should be something like this::

        def post_LoadLibraryEx(event, return_value):

            # (...)

@type  paramCount: int
@param paramCount:
    (Optional) Number of parameters for the C{preCB} callback,
    not counting the return address. Parameters are read from
    the stack and assumed to be DWORDs in 32 bits and QWORDs in 64.

    This is a faster way to pull stack parameters in 32 bits, but in 64
    bits (or with some odd APIs in 32 bits) it won't be useful, since
    not all arguments to the hooked function will be of the same size.

    For a more reliable and cross-platform way of hooking use the
    C{signature} argument instead.

@type  signature: tuple
@param signature:
    (Optional) Tuple of C{ctypes} data types that constitute the
    hooked function signature. When the function is called, this will
    be used to parse the arguments from the stack. Overrides the
    C{paramCount} argument.

@rtype:  bool
@return: C{True} if the hook was set immediately, or C{False} if
    it was deferred.

### Function: stalk_function(self, pid, address, preCB, postCB, paramCount, signature)

**Description:** Sets a one-shot function hook at the given address.

If instead of an address you pass a label, the hook may be
deferred until the DLL it points to is loaded.

@type  pid: int
@param pid: Process global ID.

@type  address: int or str
@param address:
    Memory address of code instruction to break at. It can be an
    integer value for the actual address or a string with a label
    to be resolved.

@type  preCB: function
@param preCB: (Optional) Callback triggered on function entry.

    The signature for the callback should be something like this::

        def pre_LoadLibraryEx(event, ra, lpFilename, hFile, dwFlags):

            # return address
            ra = params[0]

            # function arguments start from here...
            szFilename = event.get_process().peek_string(lpFilename)

            # (...)

    Note that all pointer types are treated like void pointers, so your
    callback won't get the string or structure pointed to by it, but
    the remote memory address instead. This is so to prevent the ctypes
    library from being "too helpful" and trying to dereference the
    pointer. To get the actual data being pointed to, use one of the
    L{Process.read} methods.

@type  postCB: function
@param postCB: (Optional) Callback triggered on function exit.

    The signature for the callback should be something like this::

        def post_LoadLibraryEx(event, return_value):

            # (...)

@type  paramCount: int
@param paramCount:
    (Optional) Number of parameters for the C{preCB} callback,
    not counting the return address. Parameters are read from
    the stack and assumed to be DWORDs in 32 bits and QWORDs in 64.

    This is a faster way to pull stack parameters in 32 bits, but in 64
    bits (or with some odd APIs in 32 bits) it won't be useful, since
    not all arguments to the hooked function will be of the same size.

    For a more reliable and cross-platform way of hooking use the
    C{signature} argument instead.

@type  signature: tuple
@param signature:
    (Optional) Tuple of C{ctypes} data types that constitute the
    hooked function signature. When the function is called, this will
    be used to parse the arguments from the stack. Overrides the
    C{paramCount} argument.

@rtype:  bool
@return: C{True} if the breakpoint was set immediately, or C{False} if
    it was deferred.

### Function: dont_hook_function(self, pid, address)

**Description:** Removes a function hook set by L{hook_function}.

@type  pid: int
@param pid: Process global ID.

@type  address: int or str
@param address:
    Memory address of code instruction to break at. It can be an
    integer value for the actual address or a string with a label
    to be resolved.

### Function: dont_stalk_function(self, pid, address)

**Description:** Removes a function hook set by L{stalk_function}.

@type  pid: int
@param pid: Process global ID.

@type  address: int or str
@param address:
    Memory address of code instruction to break at. It can be an
    integer value for the actual address or a string with a label
    to be resolved.

### Function: __set_variable_watch(self, tid, address, size, action)

**Description:** Used by L{watch_variable} and L{stalk_variable}.

@type  tid: int
@param tid: Thread global ID.

@type  address: int
@param address: Memory address of variable to watch.

@type  size: int
@param size: Size of variable to watch. The only supported sizes are:
    byte (1), word (2), dword (4) and qword (8).

@type  action: function
@param action: (Optional) Action callback function.

    See L{define_hardware_breakpoint} for more details.

@rtype:  L{HardwareBreakpoint}
@return: Hardware breakpoint at the requested address.

### Function: __clear_variable_watch(self, tid, address)

**Description:** Used by L{dont_watch_variable} and L{dont_stalk_variable}.

@type  tid: int
@param tid: Thread global ID.

@type  address: int
@param address: Memory address of variable to stop watching.

### Function: watch_variable(self, tid, address, size, action)

**Description:** Sets a hardware breakpoint at the given thread, address and size.

@see: L{dont_watch_variable}

@type  tid: int
@param tid: Thread global ID.

@type  address: int
@param address: Memory address of variable to watch.

@type  size: int
@param size: Size of variable to watch. The only supported sizes are:
    byte (1), word (2), dword (4) and qword (8).

@type  action: function
@param action: (Optional) Action callback function.

    See L{define_hardware_breakpoint} for more details.

### Function: stalk_variable(self, tid, address, size, action)

**Description:** Sets a one-shot hardware breakpoint at the given thread,
address and size.

@see: L{dont_watch_variable}

@type  tid: int
@param tid: Thread global ID.

@type  address: int
@param address: Memory address of variable to watch.

@type  size: int
@param size: Size of variable to watch. The only supported sizes are:
    byte (1), word (2), dword (4) and qword (8).

@type  action: function
@param action: (Optional) Action callback function.

    See L{define_hardware_breakpoint} for more details.

### Function: dont_watch_variable(self, tid, address)

**Description:** Clears a hardware breakpoint set by L{watch_variable}.

@type  tid: int
@param tid: Thread global ID.

@type  address: int
@param address: Memory address of variable to stop watching.

### Function: dont_stalk_variable(self, tid, address)

**Description:** Clears a hardware breakpoint set by L{stalk_variable}.

@type  tid: int
@param tid: Thread global ID.

@type  address: int
@param address: Memory address of variable to stop watching.

### Function: __set_buffer_watch(self, pid, address, size, action, bOneShot)

**Description:** Used by L{watch_buffer} and L{stalk_buffer}.

@type  pid: int
@param pid: Process global ID.

@type  address: int
@param address: Memory address of buffer to watch.

@type  size: int
@param size: Size in bytes of buffer to watch.

@type  action: function
@param action: (Optional) Action callback function.

    See L{define_page_breakpoint} for more details.

@type  bOneShot: bool
@param bOneShot:
    C{True} to set a one-shot breakpoint,
    C{False} to set a normal breakpoint.

### Function: __clear_buffer_watch_old_method(self, pid, address, size)

**Description:** Used by L{dont_watch_buffer} and L{dont_stalk_buffer}.

@warn: Deprecated since WinAppDbg 1.5.

@type  pid: int
@param pid: Process global ID.

@type  address: int
@param address: Memory address of buffer to stop watching.

@type  size: int
@param size: Size in bytes of buffer to stop watching.

### Function: __clear_buffer_watch(self, bw)

**Description:** Used by L{dont_watch_buffer} and L{dont_stalk_buffer}.

@type  bw: L{BufferWatch}
@param bw: Buffer watch identifier.

### Function: watch_buffer(self, pid, address, size, action)

**Description:** Sets a page breakpoint and notifies when the given buffer is accessed.

@see: L{dont_watch_variable}

@type  pid: int
@param pid: Process global ID.

@type  address: int
@param address: Memory address of buffer to watch.

@type  size: int
@param size: Size in bytes of buffer to watch.

@type  action: function
@param action: (Optional) Action callback function.

    See L{define_page_breakpoint} for more details.

@rtype:  L{BufferWatch}
@return: Buffer watch identifier.

### Function: stalk_buffer(self, pid, address, size, action)

**Description:** Sets a one-shot page breakpoint and notifies
when the given buffer is accessed.

@see: L{dont_watch_variable}

@type  pid: int
@param pid: Process global ID.

@type  address: int
@param address: Memory address of buffer to watch.

@type  size: int
@param size: Size in bytes of buffer to watch.

@type  action: function
@param action: (Optional) Action callback function.

    See L{define_page_breakpoint} for more details.

@rtype:  L{BufferWatch}
@return: Buffer watch identifier.

### Function: dont_watch_buffer(self, bw)

**Description:** Clears a page breakpoint set by L{watch_buffer}.

@type  bw: L{BufferWatch}
@param bw:
    Buffer watch identifier returned by L{watch_buffer}.

### Function: dont_stalk_buffer(self, bw)

**Description:** Clears a page breakpoint set by L{stalk_buffer}.

@type  bw: L{BufferWatch}
@param bw:
    Buffer watch identifier returned by L{stalk_buffer}.

### Function: __start_tracing(self, thread)

**Description:** @type  thread: L{Thread}
@param thread: Thread to start tracing.

### Function: __stop_tracing(self, thread)

**Description:** @type  thread: L{Thread}
@param thread: Thread to stop tracing.

### Function: is_tracing(self, tid)

**Description:** @type  tid: int
@param tid: Thread global ID.

@rtype:  bool
@return: C{True} if the thread is being traced, C{False} otherwise.

### Function: get_traced_tids(self)

**Description:** Retrieves the list of global IDs of all threads being traced.

@rtype:  list( int... )
@return: List of thread global IDs.

### Function: start_tracing(self, tid)

**Description:** Start tracing mode in the given thread.

@type  tid: int
@param tid: Global ID of thread to start tracing.

### Function: stop_tracing(self, tid)

**Description:** Stop tracing mode in the given thread.

@type  tid: int
@param tid: Global ID of thread to stop tracing.

### Function: start_tracing_process(self, pid)

**Description:** Start tracing mode for all threads in the given process.

@type  pid: int
@param pid: Global ID of process to start tracing.

### Function: stop_tracing_process(self, pid)

**Description:** Stop tracing mode for all threads in the given process.

@type  pid: int
@param pid: Global ID of process to stop tracing.

### Function: start_tracing_all(self)

**Description:** Start tracing mode for all threads in all debugees.

### Function: stop_tracing_all(self)

**Description:** Stop tracing mode for all threads in all debugees.

### Function: break_on_error(self, pid, errorCode)

**Description:** Sets or clears the system breakpoint for a given Win32 error code.

Use L{Process.is_system_defined_breakpoint} to tell if a breakpoint
exception was caused by a system breakpoint or by the application
itself (for example because of a failed assertion in the code).

@note: This functionality is only available since Windows Server 2003.
    In 2003 it only breaks on error values set externally to the
    kernel32.dll library, but this was fixed in Windows Vista.

@warn: This method will fail if the debug symbols for ntdll (kernel32
    in Windows 2003) are not present. For more information see:
    L{System.fix_symbol_store_path}.

@see: U{http://www.nynaeve.net/?p=147}

@type  pid: int
@param pid: Process ID.

@type  errorCode: int
@param errorCode: Win32 error code to stop on. Set to C{0} or
    C{ERROR_SUCCESS} to clear the breakpoint instead.

@raise NotImplementedError:
    The functionality is not supported in this system.

@raise WindowsError:
    An error occurred while processing this request.

### Function: dont_break_on_error(self, pid)

**Description:** Alias to L{break_on_error}C{(pid, ERROR_SUCCESS)}.

@type  pid: int
@param pid: Process ID.

@raise NotImplementedError:
    The functionality is not supported in this system.

@raise WindowsError:
    An error occurred while processing this request.

### Function: resolve_exported_function(self, pid, modName, procName)

**Description:** Resolves the exported DLL function for the given process.

@type  pid: int
@param pid: Process global ID.

@type  modName: str
@param modName: Name of the module that exports the function.

@type  procName: str
@param procName: Name of the exported function to resolve.

@rtype:  int, None
@return: On success, the address of the exported function.
    On failure, returns C{None}.

### Function: resolve_label(self, pid, label)

**Description:** Resolves a label for the given process.

@type  pid: int
@param pid: Process global ID.

@type  label: str
@param label: Label to resolve.

@rtype:  int
@return: Memory address pointed to by the label.

@raise ValueError: The label is malformed or impossible to resolve.
@raise RuntimeError: Cannot resolve the module or function.

## Class: Arguments

## Class: RegisterArguments

## Class: FloatArguments

## Class: StackArguments
