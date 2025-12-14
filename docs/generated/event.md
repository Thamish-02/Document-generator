## AI Summary

A file named event.py.


## Class: EventCallbackWarning

**Description:** This warning is issued when an uncaught exception was raised by a
user-defined event handler.

## Class: Event

**Description:** Event object.

@type eventMethod: str
@cvar eventMethod:
    Method name to call when using L{EventHandler} subclasses.
    Used internally.

@type eventName: str
@cvar eventName:
    User-friendly name of the event.

@type eventDescription: str
@cvar eventDescription:
    User-friendly description of the event.

@type debug: L{Debug}
@ivar debug:
    Debug object that received the event.

@type raw: L{DEBUG_EVENT}
@ivar raw:
    Raw DEBUG_EVENT structure as used by the Win32 API.

@type continueStatus: int
@ivar continueStatus:
    Continue status to pass to L{win32.ContinueDebugEvent}.

## Class: NoEvent

**Description:** No event.

Dummy L{Event} object that can be used as a placeholder when no debug
event has occured yet. It's never returned by the L{EventFactory}.

## Class: ExceptionEvent

**Description:** Exception event.

@type exceptionName: dict( int S{->} str )
@cvar exceptionName:
    Mapping of exception constants to their names.

@type exceptionDescription: dict( int S{->} str )
@cvar exceptionDescription:
    Mapping of exception constants to user-friendly strings.

@type breakpoint: L{Breakpoint}
@ivar breakpoint:
    If the exception was caused by one of our breakpoints, this member
    contains a reference to the breakpoint object. Otherwise it's not
    defined. It should only be used from the condition or action callback
    routines, instead of the event handler.

@type hook: L{Hook}
@ivar hook:
    If the exception was caused by a function hook, this member contains a
    reference to the hook object. Otherwise it's not defined. It should
    only be used from the hook callback routines, instead of the event
    handler.

## Class: CreateThreadEvent

**Description:** Thread creation event.

## Class: CreateProcessEvent

**Description:** Process creation event.

## Class: ExitThreadEvent

**Description:** Thread termination event.

## Class: ExitProcessEvent

**Description:** Process termination event.

## Class: LoadDLLEvent

**Description:** Module load event.

## Class: UnloadDLLEvent

**Description:** Module unload event.

## Class: OutputDebugStringEvent

**Description:** Debug string output event.

## Class: RIPEvent

**Description:** RIP event.

## Class: EventFactory

**Description:** Factory of L{Event} objects.

@type baseEvent: L{Event}
@cvar baseEvent:
    Base class for Event objects.
    It's used for unknown event codes.

@type eventClasses: dict( int S{->} L{Event} )
@cvar eventClasses:
    Dictionary that maps event codes to L{Event} subclasses.

## Class: EventHandler

**Description:** Base class for debug event handlers.

Your program should subclass it to implement it's own event handling.

The constructor can be overriden as long as you call the superclass
constructor. The special method L{__call__} B{MUST NOT} be overriden.

The signature for event handlers is the following::

    def event_handler(self, event):

Where B{event} is an L{Event} object.

Each event handler is named after the event they handle.
This is the list of all valid event handler names:

 - I{event}

   Receives an L{Event} object or an object of any of it's subclasses,
   and handles any event for which no handler was defined.

 - I{unknown_event}

   Receives an L{Event} object or an object of any of it's subclasses,
   and handles any event unknown to the debugging engine. (This is not
   likely to happen unless the Win32 debugging API is changed in future
   versions of Windows).

 - I{exception}

   Receives an L{ExceptionEvent} object and handles any exception for
   which no handler was defined. See above for exception handlers.

 - I{unknown_exception}

   Receives an L{ExceptionEvent} object and handles any exception unknown
   to the debugging engine. This usually happens for C++ exceptions, which
   are not standardized and may change from one compiler to the next.

   Currently we have partial support for C++ exceptions thrown by Microsoft
   compilers.

   Also see: U{RaiseException()
   <http://msdn.microsoft.com/en-us/library/ms680552(VS.85).aspx>}

 - I{create_thread}

   Receives a L{CreateThreadEvent} object.

 - I{create_process}

   Receives a L{CreateProcessEvent} object.

 - I{exit_thread}

   Receives a L{ExitThreadEvent} object.

 - I{exit_process}

   Receives a L{ExitProcessEvent} object.

 - I{load_dll}

   Receives a L{LoadDLLEvent} object.

 - I{unload_dll}

   Receives an L{UnloadDLLEvent} object.

 - I{output_string}

   Receives an L{OutputDebugStringEvent} object.

 - I{rip}

   Receives a L{RIPEvent} object.

This is the list of all valid exception handler names
(they all receive an L{ExceptionEvent} object):

 - I{access_violation}
 - I{array_bounds_exceeded}
 - I{breakpoint}
 - I{control_c_exit}
 - I{datatype_misalignment}
 - I{debug_control_c}
 - I{float_denormal_operand}
 - I{float_divide_by_zero}
 - I{float_inexact_result}
 - I{float_invalid_operation}
 - I{float_overflow}
 - I{float_stack_check}
 - I{float_underflow}
 - I{guard_page}
 - I{illegal_instruction}
 - I{in_page_error}
 - I{integer_divide_by_zero}
 - I{integer_overflow}
 - I{invalid_disposition}
 - I{invalid_handle}
 - I{ms_vc_exception}
 - I{noncontinuable_exception}
 - I{possible_deadlock}
 - I{privileged_instruction}
 - I{single_step}
 - I{stack_overflow}
 - I{wow64_breakpoint}



@type apiHooks: dict( str S{->} list( tuple( str, int ) ) )
@cvar apiHooks:
    Dictionary that maps module names to lists of
    tuples of ( procedure name, parameter count ).

    All procedures listed here will be hooked for calls from the debugee.
    When this happens, the corresponding event handler can be notified both
    when the procedure is entered and when it's left by the debugee.

    For example, let's hook the LoadLibraryEx() API call.
    This would be the declaration of apiHooks::

        from winappdbg import EventHandler
        from winappdbg.win32 import *

        # (...)

        class MyEventHandler (EventHandler):

            apiHook = {

                "kernel32.dll" : (

                    #   Procedure name      Signature
                    (   "LoadLibraryEx",    (PVOID, HANDLE, DWORD) ),

                    # (more procedures can go here...)
                ),

                # (more libraries can go here...)
            }

            # (your method definitions go here...)

    Note that all pointer types are treated like void pointers, so your
    callback won't get the string or structure pointed to by it, but the
    remote memory address instead. This is so to prevent the ctypes library
    from being "too helpful" and trying to dereference the pointer. To get
    the actual data being pointed to, use one of the L{Process.read}
    methods.

    Now, to intercept calls to LoadLibraryEx define a method like this in
    your event handler class::

        def pre_LoadLibraryEx(self, event, ra, lpFilename, hFile, dwFlags):
            szFilename = event.get_process().peek_string(lpFilename)

            # (...)

    Note that the first parameter is always the L{Event} object, and the
    second parameter is the return address. The third parameter and above
    are the values passed to the hooked function.

    Finally, to intercept returns from calls to LoadLibraryEx define a
    method like this::

        def post_LoadLibraryEx(self, event, retval):
            # (...)

    The first parameter is the L{Event} object and the second is the
    return value from the hooked function.

## Class: EventSift

**Description:** Event handler that allows you to use customized event handlers for each
process you're attached to.

This makes coding the event handlers much easier, because each instance
will only "know" about one process. So you can code your event handler as
if only one process was being debugged, but your debugger can attach to
multiple processes.

Example::
    from winappdbg import Debug, EventHandler, EventSift

    # This class was written assuming only one process is attached.
    # If you used it directly it would break when attaching to another
    # process, or when a child process is spawned.
    class MyEventHandler (EventHandler):

        def create_process(self, event):
            self.first = True
            self.name = event.get_process().get_filename()
            print "Attached to %s" % self.name

        def breakpoint(self, event):
            if self.first:
                self.first = False
                print "First breakpoint reached at %s" % self.name

        def exit_process(self, event):
            print "Detached from %s" % self.name

    # Now when debugging we use the EventSift to be able to work with
    # multiple processes while keeping our code simple. :)
    if __name__ == "__main__":
        handler = EventSift(MyEventHandler)
        #handler = MyEventHandler()  # try uncommenting this line...
        with Debug(handler) as debug:
            debug.execl("calc.exe")
            debug.execl("notepad.exe")
            debug.execl("charmap.exe")
            debug.loop()

Subclasses of C{EventSift} can prevent specific event types from
being forwarded by simply defining a method for it. That means your
subclass can handle some event types globally while letting other types
be handled on per-process basis. To forward events manually you can
call C{self.event(event)}.

Example::
    class MySift (EventSift):

        # Don't forward this event.
        def debug_control_c(self, event):
            pass

        # Handle this event globally without forwarding it.
        def output_string(self, event):
            print "Debug string: %s" % event.get_debug_string()

        # Handle this event globally and then forward it.
        def create_process(self, event):
            print "New process created, PID: %d" % event.get_pid()
            return self.event(event)

        # All other events will be forwarded.

Note that overriding the C{event} method would cause no events to be
forwarded at all. To prevent this, call the superclass implementation.

Example::

    def we_want_to_forward_this_event(event):
        "Use whatever logic you want here..."
        # (...return True or False...)

    class MySift (EventSift):

        def event(self, event):

            # If the event matches some custom criteria...
            if we_want_to_forward_this_event(event):

                # Forward it.
                return super(MySift, self).event(event)

            # Otherwise, don't.

@type cls: class
@ivar cls:
    Event handler class. There will be one instance of this class
    per debugged process in the L{forward} dictionary.

@type argv: list
@ivar argv:
    Positional arguments to pass to the constructor of L{cls}.

@type argd: list
@ivar argd:
    Keyword arguments to pass to the constructor of L{cls}.

@type forward: dict
@ivar forward:
    Dictionary that maps each debugged process ID to an instance of L{cls}.

## Class: EventDispatcher

**Description:** Implements debug event dispatching capabilities.

@group Debugging events:
    get_event_handler, set_event_handler, get_handler_method

### Function: __init__(self, debug, raw)

**Description:** @type  debug: L{Debug}
@param debug: Debug object that received the event.

@type  raw: L{DEBUG_EVENT}
@param raw: Raw DEBUG_EVENT structure as used by the Win32 API.

### Function: get_event_name(self)

**Description:** @rtype:  str
@return: User-friendly name of the event.

### Function: get_event_description(self)

**Description:** @rtype:  str
@return: User-friendly description of the event.

### Function: get_event_code(self)

**Description:** @rtype:  int
@return: Debug event code as defined in the Win32 API.

### Function: get_pid(self)

**Description:** @see: L{get_process}

@rtype:  int
@return: Process global ID where the event occured.

### Function: get_tid(self)

**Description:** @see: L{get_thread}

@rtype:  int
@return: Thread global ID where the event occured.

### Function: get_process(self)

**Description:** @see: L{get_pid}

@rtype:  L{Process}
@return: Process where the event occured.

### Function: get_thread(self)

**Description:** @see: L{get_tid}

@rtype:  L{Thread}
@return: Thread where the event occured.

### Function: __init__(self, debug, raw)

### Function: __len__(self)

**Description:** Always returns C{0}, so when evaluating the object as a boolean it's
always C{False}. This prevents L{Debug.cont} from trying to continue
a dummy event.

### Function: get_event_code(self)

### Function: get_pid(self)

### Function: get_tid(self)

### Function: get_process(self)

### Function: get_thread(self)

### Function: eventMethod(self)

### Function: get_exception_name(self)

**Description:** @rtype:  str
@return: Name of the exception as defined by the Win32 API.

### Function: get_exception_description(self)

**Description:** @rtype:  str
@return: User-friendly name of the exception.

### Function: is_first_chance(self)

**Description:** @rtype:  bool
@return: C{True} for first chance exceptions, C{False} for last chance.

### Function: is_last_chance(self)

**Description:** @rtype:  bool
@return: The opposite of L{is_first_chance}.

### Function: is_noncontinuable(self)

**Description:** @see: U{http://msdn.microsoft.com/en-us/library/aa363082(VS.85).aspx}

@rtype:  bool
@return: C{True} if the exception is noncontinuable,
    C{False} otherwise.

    Attempting to continue a noncontinuable exception results in an
    EXCEPTION_NONCONTINUABLE_EXCEPTION exception to be raised.

### Function: is_continuable(self)

**Description:** @rtype:  bool
@return: The opposite of L{is_noncontinuable}.

### Function: is_user_defined_exception(self)

**Description:** Determines if this is an user-defined exception. User-defined
exceptions may contain any exception code that is not system reserved.

Often the exception code is also a valid Win32 error code, but that's
up to the debugged application.

@rtype:  bool
@return: C{True} if the exception is user-defined, C{False} otherwise.

### Function: is_system_defined_exception(self)

**Description:** @rtype:  bool
@return: The opposite of L{is_user_defined_exception}.

### Function: get_exception_code(self)

**Description:** @rtype:  int
@return: Exception code as defined by the Win32 API.

### Function: get_exception_address(self)

**Description:** @rtype:  int
@return: Memory address where the exception occured.

### Function: get_exception_information(self, index)

**Description:** @type  index: int
@param index: Index into the exception information block.

@rtype:  int
@return: Exception information DWORD.

### Function: get_exception_information_as_list(self)

**Description:** @rtype:  list( int )
@return: Exception information block.

### Function: get_fault_type(self)

**Description:** @rtype:  int
@return: Access violation type.
    Should be one of the following constants:

     - L{win32.EXCEPTION_READ_FAULT}
     - L{win32.EXCEPTION_WRITE_FAULT}
     - L{win32.EXCEPTION_EXECUTE_FAULT}

@note: This method is only meaningful for access violation exceptions,
    in-page memory error exceptions and guard page exceptions.

@raise NotImplementedError: Wrong kind of exception.

### Function: get_fault_address(self)

**Description:** @rtype:  int
@return: Access violation memory address.

@note: This method is only meaningful for access violation exceptions,
    in-page memory error exceptions and guard page exceptions.

@raise NotImplementedError: Wrong kind of exception.

### Function: get_ntstatus_code(self)

**Description:** @rtype:  int
@return: NTSTATUS status code that caused the exception.

@note: This method is only meaningful for in-page memory error
    exceptions.

@raise NotImplementedError: Not an in-page memory error.

### Function: is_nested(self)

**Description:** @rtype:  bool
@return: Returns C{True} if there are additional exception records
    associated with this exception. This would mean the exception
    is nested, that is, it was triggered while trying to handle
    at least one previous exception.

### Function: get_raw_exception_record_list(self)

**Description:** Traverses the exception record linked list and builds a Python list.

Nested exception records are received for nested exceptions. This
happens when an exception is raised in the debugee while trying to
handle a previous exception.

@rtype:  list( L{win32.EXCEPTION_RECORD} )
@return:
    List of raw exception record structures as used by the Win32 API.

    There is always at least one exception record, so the list is
    never empty. All other methods of this class read from the first
    exception record only, that is, the most recent exception.

### Function: get_nested_exceptions(self)

**Description:** Traverses the exception record linked list and builds a Python list.

Nested exception records are received for nested exceptions. This
happens when an exception is raised in the debugee while trying to
handle a previous exception.

@rtype:  list( L{ExceptionEvent} )
@return:
    List of ExceptionEvent objects representing each exception record
    found in this event.

    There is always at least one exception record, so the list is
    never empty. All other methods of this class read from the first
    exception record only, that is, the most recent exception.

### Function: get_thread_handle(self)

**Description:** @rtype:  L{ThreadHandle}
@return: Thread handle received from the system.
    Returns C{None} if the handle is not available.

### Function: get_teb(self)

**Description:** @rtype:  int
@return: Pointer to the TEB.

### Function: get_start_address(self)

**Description:** @rtype:  int
@return: Pointer to the first instruction to execute in this thread.

    Returns C{NULL} when the debugger attached to a process
    and the thread already existed.

    See U{http://msdn.microsoft.com/en-us/library/ms679295(VS.85).aspx}

### Function: get_file_handle(self)

**Description:** @rtype:  L{FileHandle} or None
@return: File handle to the main module, received from the system.
    Returns C{None} if the handle is not available.

### Function: get_process_handle(self)

**Description:** @rtype:  L{ProcessHandle}
@return: Process handle received from the system.
    Returns C{None} if the handle is not available.

### Function: get_thread_handle(self)

**Description:** @rtype:  L{ThreadHandle}
@return: Thread handle received from the system.
    Returns C{None} if the handle is not available.

### Function: get_start_address(self)

**Description:** @rtype:  int
@return: Pointer to the first instruction to execute in this process.

    Returns C{NULL} when the debugger attaches to a process.

    See U{http://msdn.microsoft.com/en-us/library/ms679295(VS.85).aspx}

### Function: get_image_base(self)

**Description:** @rtype:  int
@return: Base address of the main module.
@warn: This value is taken from the PE file
    and may be incorrect because of ASLR!

### Function: get_teb(self)

**Description:** @rtype:  int
@return: Pointer to the TEB.

### Function: get_debug_info(self)

**Description:** @rtype:  str
@return: Debugging information.

### Function: get_filename(self)

**Description:** @rtype:  str, None
@return: This method does it's best to retrieve the filename to
the main module of the process. However, sometimes that's not
possible, and C{None} is returned instead.

### Function: get_module_base(self)

**Description:** @rtype:  int
@return: Base address of the main module.

### Function: get_module(self)

**Description:** @rtype:  L{Module}
@return: Main module of the process.

### Function: get_exit_code(self)

**Description:** @rtype:  int
@return: Exit code of the thread.

### Function: get_exit_code(self)

**Description:** @rtype:  int
@return: Exit code of the process.

### Function: get_filename(self)

**Description:** @rtype:  None or str
@return: Filename of the main module.
    C{None} if the filename is unknown.

### Function: get_image_base(self)

**Description:** @rtype:  int
@return: Base address of the main module.

### Function: get_module_base(self)

**Description:** @rtype:  int
@return: Base address of the main module.

### Function: get_module(self)

**Description:** @rtype:  L{Module}
@return: Main module of the process.

### Function: get_module_base(self)

**Description:** @rtype:  int
@return: Base address for the newly loaded DLL.

### Function: get_module(self)

**Description:** @rtype:  L{Module}
@return: Module object for the newly loaded DLL.

### Function: get_file_handle(self)

**Description:** @rtype:  L{FileHandle} or None
@return: File handle to the newly loaded DLL received from the system.
    Returns C{None} if the handle is not available.

### Function: get_filename(self)

**Description:** @rtype:  str, None
@return: This method does it's best to retrieve the filename to
the newly loaded module. However, sometimes that's not
possible, and C{None} is returned instead.

### Function: get_module_base(self)

**Description:** @rtype:  int
@return: Base address for the recently unloaded DLL.

### Function: get_module(self)

**Description:** @rtype:  L{Module}
@return: Module object for the recently unloaded DLL.

### Function: get_file_handle(self)

**Description:** @rtype:  None or L{FileHandle}
@return: File handle to the recently unloaded DLL.
    Returns C{None} if the handle is not available.

### Function: get_filename(self)

**Description:** @rtype:  None or str
@return: Filename of the recently unloaded DLL.
    C{None} if the filename is unknown.

### Function: get_debug_string(self)

**Description:** @rtype:  str, compat.unicode
@return: String sent by the debugee.
    It may be ANSI or Unicode and may end with a null character.

### Function: get_rip_error(self)

**Description:** @rtype:  int
@return: RIP error code as defined by the Win32 API.

### Function: get_rip_type(self)

**Description:** @rtype:  int
@return: RIP type code as defined by the Win32 API.
    May be C{0} or one of the following:
     - L{win32.SLE_ERROR}
     - L{win32.SLE_MINORERROR}
     - L{win32.SLE_WARNING}

### Function: get(cls, debug, raw)

**Description:** @type  debug: L{Debug}
@param debug: Debug object that received the event.

@type  raw: L{DEBUG_EVENT}
@param raw: Raw DEBUG_EVENT structure as used by the Win32 API.

@rtype: L{Event}
@returns: An Event object or one of it's subclasses,
    depending on the event type.

### Function: __init__(self)

**Description:** Class constructor. Don't forget to call it when subclassing!

Forgetting to call the superclass constructor is a common mistake when
you're new to Python. :)

Example::
    class MyEventHandler (EventHandler):

        # Override the constructor to use an extra argument.
        def __init__(self, myArgument):

            # Do something with the argument, like keeping it
            # as an instance variable.
            self.myVariable = myArgument

            # Call the superclass constructor.
            super(MyEventHandler, self).__init__()

        # The rest of your code below...

### Function: __get_hooks_for_dll(self, event)

**Description:** Get the requested API hooks for the current DLL.

Used by L{__hook_dll} and L{__unhook_dll}.

### Function: __hook_dll(self, event)

**Description:** Hook the requested API calls (in self.apiHooks).

This method is called automatically whenever a DLL is loaded.

### Function: __unhook_dll(self, event)

**Description:** Unhook the requested API calls (in self.apiHooks).

This method is called automatically whenever a DLL is unloaded.

### Function: __call__(self, event)

**Description:** Dispatch debug events.

@warn: B{Don't override this method!}

@type  event: L{Event}
@param event: Event object.

### Function: __init__(self, cls)

**Description:** Maintains an instance of your event handler for each process being
debugged, and forwards the events of each process to each corresponding
instance.

@warn: If you subclass L{EventSift} and reimplement this method,
    don't forget to call the superclass constructor!

@see: L{event}

@type  cls: class
@param cls: Event handler class. This must be the class itself, not an
    instance! All additional arguments passed to the constructor of
    the event forwarder will be passed on to the constructor of this
    class as well.

### Function: __call__(self, event)

### Function: event(self, event)

**Description:** Forwards events to the corresponding instance of your event handler
for this process.

If you subclass L{EventSift} and reimplement this method, no event
will be forwarded at all unless you call the superclass implementation.

If your filtering is based on the event type, there's a much easier way
to do it: just implement a handler for it.

### Function: __init__(self, eventHandler)

**Description:** Event dispatcher.

@type  eventHandler: L{EventHandler}
@param eventHandler: (Optional) User-defined event handler.

@raise TypeError: The event handler is of an incorrect type.

@note: The L{eventHandler} parameter may be any callable Python object
    (for example a function, or an instance method).
    However you'll probably find it more convenient to use an instance
    of a subclass of L{EventHandler} here.

### Function: get_event_handler(self)

**Description:** Get the event handler.

@see: L{set_event_handler}

@rtype:  L{EventHandler}
@return: Current event handler object, or C{None}.

### Function: set_event_handler(self, eventHandler)

**Description:** Set the event handler.

@warn: This is normally not needed. Use with care!

@type  eventHandler: L{EventHandler}
@param eventHandler: New event handler object, or C{None}.

@rtype:  L{EventHandler}
@return: Previous event handler object, or C{None}.

@raise TypeError: The event handler is of an incorrect type.

@note: The L{eventHandler} parameter may be any callable Python object
    (for example a function, or an instance method).
    However you'll probably find it more convenient to use an instance
    of a subclass of L{EventHandler} here.

### Function: get_handler_method(eventHandler, event, fallback)

**Description:** Retrieves the appropriate callback method from an L{EventHandler}
instance for the given L{Event} object.

@type  eventHandler: L{EventHandler}
@param eventHandler:
    Event handler object whose methods we are examining.

@type  event: L{Event}
@param event: Debugging event to be handled.

@type  fallback: callable
@param fallback: (Optional) If no suitable method is found in the
    L{EventHandler} instance, return this value.

@rtype:  callable
@return: Bound method that will handle the debugging event.
    Returns C{None} if no such method is defined.

### Function: dispatch(self, event)

**Description:** Sends event notifications to the L{Debug} object and
the L{EventHandler} object provided by the user.

The L{Debug} object will forward the notifications to it's contained
snapshot objects (L{System}, L{Process}, L{Thread} and L{Module}) when
appropriate.

@warning: This method is called automatically from L{Debug.dispatch}.

@see: L{Debug.cont}, L{Debug.loop}, L{Debug.wait}

@type  event: L{Event}
@param event: Event object passed to L{Debug.dispatch}.

@raise WindowsError: Raises an exception on error.
