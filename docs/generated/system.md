## AI Summary

A file named system.py.


## Class: System

**Description:** Interface to a batch of processes, plus some system wide settings.
Contains a snapshot of processes.

@group Platform settings:
    arch, bits, os, wow64, pageSize

@group Instrumentation:
    find_window, get_window_at, get_foreground_window,
    get_desktop_window, get_shell_window

@group Debugging:
    load_dbghelp, fix_symbol_store_path,
    request_debug_privileges, drop_debug_privileges

@group Postmortem debugging:
    get_postmortem_debugger, set_postmortem_debugger,
    get_postmortem_exclusion_list, add_to_postmortem_exclusion_list,
    remove_from_postmortem_exclusion_list

@group System services:
    get_services, get_active_services,
    start_service, stop_service,
    pause_service, resume_service,
    get_service_display_name, get_service_from_display_name

@group Permissions and privileges:
    request_privileges, drop_privileges, adjust_privileges, is_admin

@group Miscellaneous global settings:
    set_kill_on_exit_mode, read_msr, write_msr, enable_step_on_branch_mode,
    get_last_branch_location

@type arch: str
@cvar arch: Name of the processor architecture we're running on.
    For more details see L{win32.version._get_arch}.

@type bits: int
@cvar bits: Size of the machine word in bits for the current architecture.
    For more details see L{win32.version._get_bits}.

@type os: str
@cvar os: Name of the Windows version we're runing on.
    For more details see L{win32.version._get_os}.

@type wow64: bool
@cvar wow64: C{True} if the debugger is a 32 bits process running in a 64
    bits version of Windows, C{False} otherwise.

@type pageSize: int
@cvar pageSize: Page size in bytes. Defaults to 0x1000 but it's
    automatically updated on runtime when importing the module.

@type registry: L{Registry}
@cvar registry: Windows Registry for this machine.

### Function: pageSize(cls)

### Function: find_window(className, windowName)

**Description:** Find the first top-level window in the current desktop to match the
given class name and/or window name. If neither are provided any
top-level window will match.

@see: L{get_window_at}

@type  className: str
@param className: (Optional) Class name of the window to find.
    If C{None} or not used any class name will match the search.

@type  windowName: str
@param windowName: (Optional) Caption text of the window to find.
    If C{None} or not used any caption text will match the search.

@rtype:  L{Window} or None
@return: A window that matches the request. There may be more matching
    windows, but this method only returns one. If no matching window
    is found, the return value is C{None}.

@raise WindowsError: An error occured while processing this request.

### Function: get_window_at(x, y)

**Description:** Get the window located at the given coordinates in the desktop.
If no such window exists an exception is raised.

@see: L{find_window}

@type  x: int
@param x: Horizontal coordinate.
@type  y: int
@param y: Vertical coordinate.

@rtype:  L{Window}
@return: Window at the requested position. If no such window
    exists a C{WindowsError} exception is raised.

@raise WindowsError: An error occured while processing this request.

### Function: get_foreground_window()

**Description:** @rtype:  L{Window}
@return: Returns the foreground window.
@raise WindowsError: An error occured while processing this request.

### Function: get_desktop_window()

**Description:** @rtype:  L{Window}
@return: Returns the desktop window.
@raise WindowsError: An error occured while processing this request.

### Function: get_shell_window()

**Description:** @rtype:  L{Window}
@return: Returns the shell window.
@raise WindowsError: An error occured while processing this request.

### Function: request_debug_privileges(cls, bIgnoreExceptions)

**Description:** Requests debug privileges.

This may be needed to debug processes running as SYSTEM
(such as services) since Windows XP.

@type  bIgnoreExceptions: bool
@param bIgnoreExceptions: C{True} to ignore any exceptions that may be
    raised when requesting debug privileges.

@rtype:  bool
@return: C{True} on success, C{False} on failure.

@raise WindowsError: Raises an exception on error, unless
    C{bIgnoreExceptions} is C{True}.

### Function: drop_debug_privileges(cls, bIgnoreExceptions)

**Description:** Drops debug privileges.

This may be needed to avoid being detected
by certain anti-debug tricks.

@type  bIgnoreExceptions: bool
@param bIgnoreExceptions: C{True} to ignore any exceptions that may be
    raised when dropping debug privileges.

@rtype:  bool
@return: C{True} on success, C{False} on failure.

@raise WindowsError: Raises an exception on error, unless
    C{bIgnoreExceptions} is C{True}.

### Function: request_privileges(cls)

**Description:** Requests privileges.

@type  privileges: int...
@param privileges: Privileges to request.

@raise WindowsError: Raises an exception on error.

### Function: drop_privileges(cls)

**Description:** Drops privileges.

@type  privileges: int...
@param privileges: Privileges to drop.

@raise WindowsError: Raises an exception on error.

### Function: adjust_privileges(state, privileges)

**Description:** Requests or drops privileges.

@type  state: bool
@param state: C{True} to request, C{False} to drop.

@type  privileges: list(int)
@param privileges: Privileges to request or drop.

@raise WindowsError: Raises an exception on error.

### Function: is_admin()

**Description:** @rtype:  bool
@return: C{True} if the current user as Administrator privileges,
    C{False} otherwise. Since Windows Vista and above this means if
    the current process is running with UAC elevation or not.

### Function: get_file_version_info(cls, filename)

**Description:** Get the program version from an executable file, if available.

@type  filename: str
@param filename: Pathname to the executable file to query.

@rtype: tuple(str, str, bool, bool, str, str)
@return: Tuple with version information extracted from the executable
    file metadata, containing the following:
     - File version number (C{"major.minor"}).
     - Product version number (C{"major.minor"}).
     - C{True} for debug builds, C{False} for production builds.
     - C{True} for legacy OS builds (DOS, OS/2, Win16),
       C{False} for modern OS builds.
     - Binary file type.
       May be one of the following values:
        - "application"
        - "dynamic link library"
        - "static link library"
        - "font"
        - "raster font"
        - "TrueType font"
        - "vector font"
        - "driver"
        - "communications driver"
        - "display driver"
        - "installable driver"
        - "keyboard driver"
        - "language driver"
        - "legacy driver"
        - "mouse driver"
        - "network driver"
        - "printer driver"
        - "sound driver"
        - "system driver"
        - "versioned printer driver"
     - Binary creation timestamp.
    Any of the fields may be C{None} if not available.

@raise WindowsError: Raises an exception on error.

### Function: load_dbghelp(cls, pathname)

**Description:** Load the specified version of the C{dbghelp.dll} library.

This library is shipped with the Debugging Tools for Windows, and it's
required to load debug symbols.

Normally you don't need to call this method, as WinAppDbg already tries
to load the latest version automatically - but it may come in handy if
the Debugging Tools are installed in a non standard folder.

Example::
    from winappdbg import Debug

    def simple_debugger( argv ):

        # Instance a Debug object, passing it the event handler callback
        debug = Debug( my_event_handler )
        try:

            # Load a specific dbghelp.dll file
            debug.system.load_dbghelp("C:\Some folder\dbghelp.dll")

            # Start a new process for debugging
            debug.execv( argv )

            # Wait for the debugee to finish
            debug.loop()

        # Stop the debugger
        finally:
            debug.stop()

@see: U{http://msdn.microsoft.com/en-us/library/ms679294(VS.85).aspx}

@type  pathname: str
@param pathname:
    (Optional) Full pathname to the C{dbghelp.dll} library.
    If not provided this method will try to autodetect it.

@rtype:  ctypes.WinDLL
@return: Loaded instance of C{dbghelp.dll}.

@raise NotImplementedError: This feature was not implemented for the
    current architecture.

@raise WindowsError: An error occured while processing this request.

### Function: fix_symbol_store_path(symbol_store_path, remote, force)

**Description:** Fix the symbol store path. Equivalent to the C{.symfix} command in
Microsoft WinDbg.

If the symbol store path environment variable hasn't been set, this
method will provide a default one.

@type  symbol_store_path: str or None
@param symbol_store_path: (Optional) Symbol store path to set.

@type  remote: bool
@param remote: (Optional) Defines the symbol store path to set when the
    C{symbol_store_path} is C{None}.

    If C{True} the default symbol store path is set to the Microsoft
    symbol server. Debug symbols will be downloaded through HTTP.
    This gives the best results but is also quite slow.

    If C{False} the default symbol store path is set to the local
    cache only. This prevents debug symbols from being downloaded and
    is faster, but unless you've installed the debug symbols on this
    machine or downloaded them in a previous debugging session, some
    symbols may be missing.

    If the C{symbol_store_path} argument is not C{None}, this argument
    is ignored entirely.

@type  force: bool
@param force: (Optional) If C{True} the new symbol store path is set
    always. If C{False} the new symbol store path is only set if
    missing.

    This allows you to call this method preventively to ensure the
    symbol server is always set up correctly when running your script,
    but without messing up whatever configuration the user has.

    Example::
        from winappdbg import Debug, System

        def simple_debugger( argv ):

            # Instance a Debug object
            debug = Debug( MyEventHandler() )
            try:

                # Make sure the remote symbol store is set
                System.fix_symbol_store_path(remote = True,
                                              force = False)

                # Start a new process for debugging
                debug.execv( argv )

                # Wait for the debugee to finish
                debug.loop()

            # Stop the debugger
            finally:
                debug.stop()

@rtype:  str or None
@return: The previously set symbol store path if any,
    otherwise returns C{None}.

### Function: set_kill_on_exit_mode(bKillOnExit)

**Description:** Defines the behavior of the debugged processes when the debugging
thread dies. This method only affects the calling thread.

Works on the following platforms:

 - Microsoft Windows XP and above.
 - Wine (Windows Emulator).

Fails on the following platforms:

 - Microsoft Windows 2000 and below.
 - ReactOS.

@type  bKillOnExit: bool
@param bKillOnExit: C{True} to automatically kill processes when the
    debugger thread dies. C{False} to automatically detach from
    processes when the debugger thread dies.

@rtype:  bool
@return: C{True} on success, C{False} on error.

@note:
    This call will fail if a debug port was not created. That is, if
    the debugger isn't attached to at least one process. For more info
    see: U{http://msdn.microsoft.com/en-us/library/ms679307.aspx}

### Function: read_msr(address)

**Description:** Read the contents of the specified MSR (Machine Specific Register).

@type  address: int
@param address: MSR to read.

@rtype:  int
@return: Value of the specified MSR.

@raise WindowsError:
    Raises an exception on error.

@raise NotImplementedError:
    Current architecture is not C{i386} or C{amd64}.

@warning:
    It could potentially brick your machine.
    It works on my machine, but your mileage may vary.

### Function: write_msr(address, value)

**Description:** Set the contents of the specified MSR (Machine Specific Register).

@type  address: int
@param address: MSR to write.

@type  value: int
@param value: Contents to write on the MSR.

@raise WindowsError:
    Raises an exception on error.

@raise NotImplementedError:
    Current architecture is not C{i386} or C{amd64}.

@warning:
    It could potentially brick your machine.
    It works on my machine, but your mileage may vary.

### Function: enable_step_on_branch_mode(cls)

**Description:** When tracing, call this on every single step event
for step on branch mode.

@raise WindowsError:
    Raises C{ERROR_DEBUGGER_INACTIVE} if the debugger is not attached
    to least one process.

@raise NotImplementedError:
    Current architecture is not C{i386} or C{amd64}.

@warning:
    This method uses the processor's machine specific registers (MSR).
    It could potentially brick your machine.
    It works on my machine, but your mileage may vary.

@note:
    It doesn't seem to work in VMWare or VirtualBox machines.
    Maybe it fails in other virtualization/emulation environments,
    no extensive testing was made so far.

### Function: get_last_branch_location(cls)

**Description:** Returns the source and destination addresses of the last taken branch.

@rtype: tuple( int, int )
@return: Source and destination addresses of the last taken branch.

@raise WindowsError:
    Raises an exception on error.

@raise NotImplementedError:
    Current architecture is not C{i386} or C{amd64}.

@warning:
    This method uses the processor's machine specific registers (MSR).
    It could potentially brick your machine.
    It works on my machine, but your mileage may vary.

@note:
    It doesn't seem to work in VMWare or VirtualBox machines.
    Maybe it fails in other virtualization/emulation environments,
    no extensive testing was made so far.

### Function: get_postmortem_debugger(cls, bits)

**Description:** Returns the postmortem debugging settings from the Registry.

@see: L{set_postmortem_debugger}

@type  bits: int
@param bits: Set to C{32} for the 32 bits debugger, or C{64} for the
    64 bits debugger. Set to {None} for the default (L{System.bits}.

@rtype:  tuple( str, bool, int )
@return: A tuple containing the command line string to the postmortem
    debugger, a boolean specifying if user interaction is allowed
    before attaching, and an integer specifying a user defined hotkey.
    Any member of the tuple may be C{None}.
    See L{set_postmortem_debugger} for more details.

@raise WindowsError:
    Raises an exception on error.

### Function: get_postmortem_exclusion_list(cls, bits)

**Description:** Returns the exclusion list for the postmortem debugger.

@see: L{get_postmortem_debugger}

@type  bits: int
@param bits: Set to C{32} for the 32 bits debugger, or C{64} for the
    64 bits debugger. Set to {None} for the default (L{System.bits}).

@rtype:  list( str )
@return: List of excluded application filenames.

@raise WindowsError:
    Raises an exception on error.

### Function: set_postmortem_debugger(cls, cmdline, auto, hotkey, bits)

**Description:** Sets the postmortem debugging settings in the Registry.

@warning: This method requires administrative rights.

@see: L{get_postmortem_debugger}

@type  cmdline: str
@param cmdline: Command line to the new postmortem debugger.
    When the debugger is invoked, the first "%ld" is replaced with the
    process ID and the second "%ld" is replaced with the event handle.
    Don't forget to enclose the program filename in double quotes if
    the path contains spaces.

@type  auto: bool
@param auto: Set to C{True} if no user interaction is allowed, C{False}
    to prompt a confirmation dialog before attaching.
    Use C{None} to leave this value unchanged.

@type  hotkey: int
@param hotkey: Virtual key scan code for the user defined hotkey.
    Use C{0} to disable the hotkey.
    Use C{None} to leave this value unchanged.

@type  bits: int
@param bits: Set to C{32} for the 32 bits debugger, or C{64} for the
    64 bits debugger. Set to {None} for the default (L{System.bits}).

@rtype:  tuple( str, bool, int )
@return: Previously defined command line and auto flag.

@raise WindowsError:
    Raises an exception on error.

### Function: add_to_postmortem_exclusion_list(cls, pathname, bits)

**Description:** Adds the given filename to the exclusion list for postmortem debugging.

@warning: This method requires administrative rights.

@see: L{get_postmortem_exclusion_list}

@type  pathname: str
@param pathname:
    Application pathname to exclude from postmortem debugging.

@type  bits: int
@param bits: Set to C{32} for the 32 bits debugger, or C{64} for the
    64 bits debugger. Set to {None} for the default (L{System.bits}).

@raise WindowsError:
    Raises an exception on error.

### Function: remove_from_postmortem_exclusion_list(cls, pathname, bits)

**Description:** Removes the given filename to the exclusion list for postmortem
debugging from the Registry.

@warning: This method requires administrative rights.

@warning: Don't ever delete entries you haven't created yourself!
    Some entries are set by default for your version of Windows.
    Deleting them might deadlock your system under some circumstances.

    For more details see:
    U{http://msdn.microsoft.com/en-us/library/bb204634(v=vs.85).aspx}

@see: L{get_postmortem_exclusion_list}

@type  pathname: str
@param pathname: Application pathname to remove from the postmortem
    debugging exclusion list.

@type  bits: int
@param bits: Set to C{32} for the 32 bits debugger, or C{64} for the
    64 bits debugger. Set to {None} for the default (L{System.bits}).

@raise WindowsError:
    Raises an exception on error.

### Function: get_services()

**Description:** Retrieve a list of all system services.

@see: L{get_active_services},
    L{start_service}, L{stop_service},
    L{pause_service}, L{resume_service}

@rtype:  list( L{win32.ServiceStatusProcessEntry} )
@return: List of service status descriptors.

### Function: get_active_services()

**Description:** Retrieve a list of all active system services.

@see: L{get_services},
    L{start_service}, L{stop_service},
    L{pause_service}, L{resume_service}

@rtype:  list( L{win32.ServiceStatusProcessEntry} )
@return: List of service status descriptors.

### Function: get_service(name)

**Description:** Get the service descriptor for the given service name.

@see: L{start_service}, L{stop_service},
    L{pause_service}, L{resume_service}

@type  name: str
@param name: Service unique name. You can get this value from the
    C{ServiceName} member of the service descriptors returned by
    L{get_services} or L{get_active_services}.

@rtype:  L{win32.ServiceStatusProcess}
@return: Service status descriptor.

### Function: get_service_display_name(name)

**Description:** Get the service display name for the given service name.

@see: L{get_service}

@type  name: str
@param name: Service unique name. You can get this value from the
    C{ServiceName} member of the service descriptors returned by
    L{get_services} or L{get_active_services}.

@rtype:  str
@return: Service display name.

### Function: get_service_from_display_name(displayName)

**Description:** Get the service unique name given its display name.

@see: L{get_service}

@type  displayName: str
@param displayName: Service display name. You can get this value from
    the C{DisplayName} member of the service descriptors returned by
    L{get_services} or L{get_active_services}.

@rtype:  str
@return: Service unique name.

### Function: start_service(name, argv)

**Description:** Start the service given by name.

@warn: This method requires UAC elevation in Windows Vista and above.

@see: L{stop_service}, L{pause_service}, L{resume_service}

@type  name: str
@param name: Service unique name. You can get this value from the
    C{ServiceName} member of the service descriptors returned by
    L{get_services} or L{get_active_services}.

### Function: stop_service(name)

**Description:** Stop the service given by name.

@warn: This method requires UAC elevation in Windows Vista and above.

@see: L{get_services}, L{get_active_services},
    L{start_service}, L{pause_service}, L{resume_service}

### Function: pause_service(name)

**Description:** Pause the service given by name.

@warn: This method requires UAC elevation in Windows Vista and above.

@note: Not all services support this.

@see: L{get_services}, L{get_active_services},
    L{start_service}, L{stop_service}, L{resume_service}

### Function: resume_service(name)

**Description:** Resume the service given by name.

@warn: This method requires UAC elevation in Windows Vista and above.

@note: Not all services support this.

@see: L{get_services}, L{get_active_services},
    L{start_service}, L{stop_service}, L{pause_service}
