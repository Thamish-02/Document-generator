## AI Summary

A file named kernel32.py.


### Function: RaiseIfLastError(result, func, arguments)

**Description:** Error checking for Win32 API calls with no error-specific return value.

Regardless of the return value, the function calls GetLastError(). If the
code is not C{ERROR_SUCCESS} then a C{WindowsError} exception is raised.

For this to work, the user MUST call SetLastError(ERROR_SUCCESS) prior to
calling the API. Otherwise an exception may be raised even on success,
since most API calls don't clear the error status code.

## Class: Handle

**Description:** Encapsulates Win32 handles to avoid leaking them.

@type inherit: bool
@ivar inherit: C{True} if the handle is to be inherited by child processes,
    C{False} otherwise.

@type protectFromClose: bool
@ivar protectFromClose: Set to C{True} to prevent the handle from being
    closed. Must be set to C{False} before you're done using the handle,
    or it will be left open until the debugger exits. Use with care!

@see:
    L{ProcessHandle}, L{ThreadHandle}, L{FileHandle}, L{SnapshotHandle}

## Class: UserModeHandle

**Description:** Base class for non-kernel handles. Generally this means they are closed
by special Win32 API functions instead of CloseHandle() and some standard
operations (synchronizing, duplicating, inheritance) are not supported.

@type _TYPE: C type
@cvar _TYPE: C type to translate this handle to.
    Subclasses should override this.
    Defaults to L{HANDLE}.

## Class: ProcessHandle

**Description:** Win32 process handle.

@type dwAccess: int
@ivar dwAccess: Current access flags to this handle.
        This is the same value passed to L{OpenProcess}.
        Can only be C{None} if C{aHandle} is also C{None}.
        Defaults to L{PROCESS_ALL_ACCESS}.

@see: L{Handle}

## Class: ThreadHandle

**Description:** Win32 thread handle.

@type dwAccess: int
@ivar dwAccess: Current access flags to this handle.
        This is the same value passed to L{OpenThread}.
        Can only be C{None} if C{aHandle} is also C{None}.
        Defaults to L{THREAD_ALL_ACCESS}.

@see: L{Handle}

## Class: FileHandle

**Description:** Win32 file handle.

@see: L{Handle}

## Class: FileMappingHandle

**Description:** File mapping handle.

@see: L{Handle}

## Class: SnapshotHandle

**Description:** Toolhelp32 snapshot handle.

@see: L{Handle}

## Class: ProcessInformation

**Description:** Process information object returned by L{CreateProcess}.

## Class: MemoryBasicInformation

**Description:** Memory information object returned by L{VirtualQueryEx}.

## Class: ProcThreadAttributeList

**Description:** Extended process and thread attribute support.

To be used with L{STARTUPINFOEX}.
Only available for Windows Vista and above.

@type AttributeList: list of tuple( int, ctypes-compatible object )
@ivar AttributeList: List of (Attribute, Value) pairs.

@type AttributeListBuffer: L{LPPROC_THREAD_ATTRIBUTE_LIST}
@ivar AttributeListBuffer: Memory buffer used to store the attribute list.
    L{InitializeProcThreadAttributeList},
    L{UpdateProcThreadAttribute},
    L{DeleteProcThreadAttributeList} and
    L{STARTUPINFOEX}.

## Class: _OVERLAPPED_STRUCT

## Class: _OVERLAPPED_UNION

## Class: OVERLAPPED

## Class: SECURITY_ATTRIBUTES

## Class: VS_FIXEDFILEINFO

## Class: THREADNAME_INFO

## Class: MEMORY_BASIC_INFORMATION32

## Class: MEMORY_BASIC_INFORMATION64

## Class: MEMORY_BASIC_INFORMATION

## Class: FILETIME

## Class: SYSTEMTIME

## Class: BY_HANDLE_FILE_INFORMATION

## Class: FILE_INFO_BY_HANDLE_CLASS

## Class: PROCESS_INFORMATION

## Class: STARTUPINFO

## Class: STARTUPINFOEX

## Class: STARTUPINFOW

## Class: STARTUPINFOEXW

## Class: JIT_DEBUG_INFO

## Class: EXCEPTION_RECORD32

## Class: EXCEPTION_RECORD64

## Class: EXCEPTION_RECORD

## Class: EXCEPTION_DEBUG_INFO

## Class: CREATE_THREAD_DEBUG_INFO

## Class: CREATE_PROCESS_DEBUG_INFO

## Class: EXIT_THREAD_DEBUG_INFO

## Class: EXIT_PROCESS_DEBUG_INFO

## Class: LOAD_DLL_DEBUG_INFO

## Class: UNLOAD_DLL_DEBUG_INFO

## Class: OUTPUT_DEBUG_STRING_INFO

## Class: RIP_INFO

## Class: _DEBUG_EVENT_UNION_

## Class: DEBUG_EVENT

## Class: _CHAR_INFO_CHAR

## Class: CHAR_INFO

## Class: COORD

## Class: SMALL_RECT

## Class: CONSOLE_SCREEN_BUFFER_INFO

## Class: THREADENTRY32

## Class: PROCESSENTRY32

## Class: MODULEENTRY32

## Class: HEAPENTRY32

## Class: HEAPLIST32

### Function: GetLastError()

### Function: SetLastError(dwErrCode)

### Function: GetErrorMode()

### Function: SetErrorMode(uMode)

### Function: GetThreadErrorMode()

### Function: SetThreadErrorMode(dwNewMode)

### Function: CloseHandle(hHandle)

### Function: DuplicateHandle(hSourceHandle, hSourceProcessHandle, hTargetProcessHandle, dwDesiredAccess, bInheritHandle, dwOptions)

### Function: LocalFree(hMem)

### Function: GetStdHandle(nStdHandle)

### Function: GetConsoleCP()

### Function: GetConsoleOutputCP()

### Function: SetConsoleCP(wCodePageID)

### Function: SetConsoleOutputCP(wCodePageID)

### Function: SetConsoleActiveScreenBuffer(hConsoleOutput)

### Function: GetConsoleScreenBufferInfo(hConsoleOutput)

### Function: SetConsoleWindowInfo(hConsoleOutput, bAbsolute, lpConsoleWindow)

### Function: SetConsoleTextAttribute(hConsoleOutput, wAttributes)

### Function: AllocConsole()

### Function: AttachConsole(dwProcessId)

### Function: FreeConsole()

### Function: GetDllDirectoryA()

### Function: GetDllDirectoryW()

### Function: SetDllDirectoryA(lpPathName)

### Function: SetDllDirectoryW(lpPathName)

### Function: LoadLibraryA(pszLibrary)

### Function: LoadLibraryW(pszLibrary)

### Function: LoadLibraryExA(pszLibrary, dwFlags)

### Function: LoadLibraryExW(pszLibrary, dwFlags)

### Function: GetModuleHandleA(lpModuleName)

### Function: GetModuleHandleW(lpModuleName)

### Function: GetProcAddressA(hModule, lpProcName)

### Function: FreeLibrary(hModule)

### Function: RtlPcToFileHeader(PcValue)

### Function: GetHandleInformation(hObject)

### Function: SetHandleInformation(hObject, dwMask, dwFlags)

### Function: QueryFullProcessImageNameA(hProcess, dwFlags)

### Function: QueryFullProcessImageNameW(hProcess, dwFlags)

### Function: GetLogicalDriveStringsA()

### Function: GetLogicalDriveStringsW()

### Function: QueryDosDeviceA(lpDeviceName)

### Function: QueryDosDeviceW(lpDeviceName)

### Function: MapViewOfFile(hFileMappingObject, dwDesiredAccess, dwFileOffsetHigh, dwFileOffsetLow, dwNumberOfBytesToMap)

### Function: UnmapViewOfFile(lpBaseAddress)

### Function: OpenFileMappingA(dwDesiredAccess, bInheritHandle, lpName)

### Function: OpenFileMappingW(dwDesiredAccess, bInheritHandle, lpName)

### Function: CreateFileMappingA(hFile, lpAttributes, flProtect, dwMaximumSizeHigh, dwMaximumSizeLow, lpName)

### Function: CreateFileMappingW(hFile, lpAttributes, flProtect, dwMaximumSizeHigh, dwMaximumSizeLow, lpName)

### Function: CreateFileA(lpFileName, dwDesiredAccess, dwShareMode, lpSecurityAttributes, dwCreationDisposition, dwFlagsAndAttributes, hTemplateFile)

### Function: CreateFileW(lpFileName, dwDesiredAccess, dwShareMode, lpSecurityAttributes, dwCreationDisposition, dwFlagsAndAttributes, hTemplateFile)

### Function: FlushFileBuffers(hFile)

### Function: FlushViewOfFile(lpBaseAddress, dwNumberOfBytesToFlush)

### Function: SearchPathA(lpPath, lpFileName, lpExtension)

### Function: SearchPathW(lpPath, lpFileName, lpExtension)

### Function: SetSearchPathMode(Flags)

### Function: DeviceIoControl(hDevice, dwIoControlCode, lpInBuffer, nInBufferSize, lpOutBuffer, nOutBufferSize, lpOverlapped)

### Function: GetFileInformationByHandle(hFile)

### Function: GetFileInformationByHandleEx(hFile, FileInformationClass, lpFileInformation, dwBufferSize)

### Function: GetFinalPathNameByHandleA(hFile, dwFlags)

### Function: GetFinalPathNameByHandleW(hFile, dwFlags)

### Function: GetFullPathNameA(lpFileName)

### Function: GetFullPathNameW(lpFileName)

### Function: GetTempPathA()

### Function: GetTempPathW()

### Function: GetTempFileNameA(lpPathName, lpPrefixString, uUnique)

### Function: GetTempFileNameW(lpPathName, lpPrefixString, uUnique)

### Function: GetCurrentDirectoryA()

### Function: GetCurrentDirectoryW()

### Function: SetConsoleCtrlHandler(HandlerRoutine, Add)

### Function: GenerateConsoleCtrlEvent(dwCtrlEvent, dwProcessGroupId)

### Function: WaitForSingleObject(hHandle, dwMilliseconds)

### Function: WaitForSingleObjectEx(hHandle, dwMilliseconds, bAlertable)

### Function: WaitForMultipleObjects(handles, bWaitAll, dwMilliseconds)

### Function: WaitForMultipleObjectsEx(handles, bWaitAll, dwMilliseconds, bAlertable)

### Function: CreateMutexA(lpMutexAttributes, bInitialOwner, lpName)

### Function: CreateMutexW(lpMutexAttributes, bInitialOwner, lpName)

### Function: OpenMutexA(dwDesiredAccess, bInitialOwner, lpName)

### Function: OpenMutexW(dwDesiredAccess, bInitialOwner, lpName)

### Function: CreateEventA(lpMutexAttributes, bManualReset, bInitialState, lpName)

### Function: CreateEventW(lpMutexAttributes, bManualReset, bInitialState, lpName)

### Function: OpenEventA(dwDesiredAccess, bInheritHandle, lpName)

### Function: OpenEventW(dwDesiredAccess, bInheritHandle, lpName)

### Function: ReleaseMutex(hMutex)

### Function: SetEvent(hEvent)

### Function: ResetEvent(hEvent)

### Function: PulseEvent(hEvent)

### Function: WaitForDebugEvent(dwMilliseconds)

### Function: ContinueDebugEvent(dwProcessId, dwThreadId, dwContinueStatus)

### Function: FlushInstructionCache(hProcess, lpBaseAddress, dwSize)

### Function: DebugActiveProcess(dwProcessId)

### Function: DebugActiveProcessStop(dwProcessId)

### Function: CheckRemoteDebuggerPresent(hProcess)

### Function: DebugSetProcessKillOnExit(KillOnExit)

### Function: DebugBreakProcess(hProcess)

### Function: OutputDebugStringA(lpOutputString)

### Function: OutputDebugStringW(lpOutputString)

### Function: ReadProcessMemory(hProcess, lpBaseAddress, nSize)

### Function: WriteProcessMemory(hProcess, lpBaseAddress, lpBuffer)

### Function: VirtualAllocEx(hProcess, lpAddress, dwSize, flAllocationType, flProtect)

### Function: VirtualQueryEx(hProcess, lpAddress)

### Function: VirtualProtectEx(hProcess, lpAddress, dwSize, flNewProtect)

### Function: VirtualFreeEx(hProcess, lpAddress, dwSize, dwFreeType)

### Function: CreateRemoteThread(hProcess, lpThreadAttributes, dwStackSize, lpStartAddress, lpParameter, dwCreationFlags)

### Function: CreateProcessA(lpApplicationName, lpCommandLine, lpProcessAttributes, lpThreadAttributes, bInheritHandles, dwCreationFlags, lpEnvironment, lpCurrentDirectory, lpStartupInfo)

### Function: CreateProcessW(lpApplicationName, lpCommandLine, lpProcessAttributes, lpThreadAttributes, bInheritHandles, dwCreationFlags, lpEnvironment, lpCurrentDirectory, lpStartupInfo)

### Function: InitializeProcThreadAttributeList(dwAttributeCount)

### Function: UpdateProcThreadAttribute(lpAttributeList, Attribute, Value, cbSize)

### Function: DeleteProcThreadAttributeList(lpAttributeList)

### Function: OpenProcess(dwDesiredAccess, bInheritHandle, dwProcessId)

### Function: OpenThread(dwDesiredAccess, bInheritHandle, dwThreadId)

### Function: SuspendThread(hThread)

### Function: ResumeThread(hThread)

### Function: TerminateThread(hThread, dwExitCode)

### Function: TerminateProcess(hProcess, dwExitCode)

### Function: GetCurrentProcessId()

### Function: GetCurrentThreadId()

### Function: GetProcessId(hProcess)

### Function: GetThreadId(hThread)

### Function: GetProcessIdOfThread(hThread)

### Function: GetExitCodeProcess(hProcess)

### Function: GetExitCodeThread(hThread)

### Function: GetProcessVersion(ProcessId)

### Function: GetPriorityClass(hProcess)

### Function: SetPriorityClass(hProcess, dwPriorityClass)

### Function: GetProcessPriorityBoost(hProcess)

### Function: SetProcessPriorityBoost(hProcess, DisablePriorityBoost)

### Function: GetProcessAffinityMask(hProcess)

### Function: SetProcessAffinityMask(hProcess, dwProcessAffinityMask)

### Function: CreateToolhelp32Snapshot(dwFlags, th32ProcessID)

### Function: Process32First(hSnapshot)

### Function: Process32Next(hSnapshot, pe)

### Function: Thread32First(hSnapshot)

### Function: Thread32Next(hSnapshot, te)

### Function: Module32First(hSnapshot)

### Function: Module32Next(hSnapshot, me)

### Function: Heap32First(th32ProcessID, th32HeapID)

### Function: Heap32Next(he)

### Function: Heap32ListFirst(hSnapshot)

### Function: Heap32ListNext(hSnapshot, hl)

### Function: Toolhelp32ReadProcessMemory(th32ProcessID, lpBaseAddress, cbRead)

### Function: GetProcessDEPPolicy(hProcess)

### Function: GetCurrentProcessorNumber()

### Function: FlushProcessWriteBuffers()

### Function: GetGuiResources(hProcess, uiFlags)

### Function: GetProcessHandleCount(hProcess)

### Function: GetProcessTimes(hProcess)

### Function: FileTimeToSystemTime(lpFileTime)

### Function: GetSystemTimeAsFileTime()

### Function: GlobalAddAtomA(lpString)

### Function: GlobalAddAtomW(lpString)

### Function: GlobalFindAtomA(lpString)

### Function: GlobalFindAtomW(lpString)

### Function: GlobalGetAtomNameA(nAtom)

### Function: GlobalGetAtomNameW(nAtom)

### Function: GlobalDeleteAtom(nAtom)

### Function: Wow64SuspendThread(hThread)

### Function: Wow64EnableWow64FsRedirection(Wow64FsEnableRedirection)

**Description:** This function may not work reliably when there are nested calls. Therefore,
this function has been replaced by the L{Wow64DisableWow64FsRedirection}
and L{Wow64RevertWow64FsRedirection} functions.

@see: U{http://msdn.microsoft.com/en-us/library/windows/desktop/aa365744(v=vs.85).aspx}

### Function: Wow64DisableWow64FsRedirection()

### Function: Wow64RevertWow64FsRedirection(OldValue)

### Function: __init__(self, aHandle, bOwnership)

**Description:** @type  aHandle: int
@param aHandle: Win32 handle value.

@type  bOwnership: bool
@param bOwnership:
   C{True} if we own the handle and we need to close it.
   C{False} if someone else will be calling L{CloseHandle}.

### Function: value(self)

### Function: __del__(self)

**Description:** Closes the Win32 handle when the Python object is destroyed.

### Function: __enter__(self)

**Description:** Compatibility with the "C{with}" Python statement.

### Function: __exit__(self, type, value, traceback)

**Description:** Compatibility with the "C{with}" Python statement.

### Function: __copy__(self)

**Description:** Duplicates the Win32 handle when copying the Python object.

@rtype:  L{Handle}
@return: A new handle to the same Win32 object.

### Function: __deepcopy__(self)

**Description:** Duplicates the Win32 handle when copying the Python object.

@rtype:  L{Handle}
@return: A new handle to the same win32 object.

### Function: _as_parameter_(self)

**Description:** Compatibility with ctypes.
Allows passing transparently a Handle object to an API call.

### Function: from_param(value)

**Description:** Compatibility with ctypes.
Allows passing transparently a Handle object to an API call.

@type  value: int
@param value: Numeric handle value.

### Function: close(self)

**Description:** Closes the Win32 handle.

### Function: _close(self)

**Description:** Low-level close method.
This is a private method, do not call it.

### Function: dup(self)

**Description:** @rtype:  L{Handle}
@return: A new handle to the same Win32 object.

### Function: _normalize(value)

**Description:** Normalize handle values.

### Function: wait(self, dwMilliseconds)

**Description:** Wait for the Win32 object to be signaled.

@type  dwMilliseconds: int
@param dwMilliseconds: (Optional) Timeout value in milliseconds.
    Use C{INFINITE} or C{None} for no timeout.

### Function: __repr__(self)

### Function: __get_inherit(self)

### Function: __set_inherit(self, value)

### Function: __get_protectFromClose(self)

### Function: __set_protectFromClose(self, value)

### Function: _close(self)

### Function: _as_parameter_(self)

### Function: from_param(value)

### Function: inherit(self)

### Function: protectFromClose(self)

### Function: dup(self)

### Function: wait(self, dwMilliseconds)

### Function: __init__(self, aHandle, bOwnership, dwAccess)

**Description:** @type  aHandle: int
@param aHandle: Win32 handle value.

@type  bOwnership: bool
@param bOwnership:
   C{True} if we own the handle and we need to close it.
   C{False} if someone else will be calling L{CloseHandle}.

@type  dwAccess: int
@param dwAccess: Current access flags to this handle.
    This is the same value passed to L{OpenProcess}.
    Can only be C{None} if C{aHandle} is also C{None}.
    Defaults to L{PROCESS_ALL_ACCESS}.

### Function: get_pid(self)

**Description:** @rtype:  int
@return: Process global ID.

### Function: __init__(self, aHandle, bOwnership, dwAccess)

**Description:** @type  aHandle: int
@param aHandle: Win32 handle value.

@type  bOwnership: bool
@param bOwnership:
   C{True} if we own the handle and we need to close it.
   C{False} if someone else will be calling L{CloseHandle}.

@type  dwAccess: int
@param dwAccess: Current access flags to this handle.
    This is the same value passed to L{OpenThread}.
    Can only be C{None} if C{aHandle} is also C{None}.
    Defaults to L{THREAD_ALL_ACCESS}.

### Function: get_tid(self)

**Description:** @rtype:  int
@return: Thread global ID.

### Function: get_filename(self)

**Description:** @rtype:  None or str
@return: Name of the open file, or C{None} if unavailable.

### Function: __init__(self, pi)

### Function: __init__(self, mbi)

**Description:** @type  mbi: L{MEMORY_BASIC_INFORMATION} or L{MemoryBasicInformation}
@param mbi: Either a L{MEMORY_BASIC_INFORMATION} structure or another
    L{MemoryBasicInformation} instance.

### Function: __contains__(self, address)

**Description:** Test if the given memory address falls within this memory region.

@type  address: int
@param address: Memory address to test.

@rtype:  bool
@return: C{True} if the given memory address falls within this memory
    region, C{False} otherwise.

### Function: is_free(self)

**Description:** @rtype:  bool
@return: C{True} if the memory in this region is free.

### Function: is_reserved(self)

**Description:** @rtype:  bool
@return: C{True} if the memory in this region is reserved.

### Function: is_commited(self)

**Description:** @rtype:  bool
@return: C{True} if the memory in this region is commited.

### Function: is_image(self)

**Description:** @rtype:  bool
@return: C{True} if the memory in this region belongs to an executable
    image.

### Function: is_mapped(self)

**Description:** @rtype:  bool
@return: C{True} if the memory in this region belongs to a mapped file.

### Function: is_private(self)

**Description:** @rtype:  bool
@return: C{True} if the memory in this region is private.

### Function: is_guard(self)

**Description:** @rtype:  bool
@return: C{True} if all pages in this region are guard pages.

### Function: has_content(self)

**Description:** @rtype:  bool
@return: C{True} if the memory in this region has any data in it.

### Function: is_readable(self)

**Description:** @rtype:  bool
@return: C{True} if all pages in this region are readable.

### Function: is_writeable(self)

**Description:** @rtype:  bool
@return: C{True} if all pages in this region are writeable.

### Function: is_copy_on_write(self)

**Description:** @rtype:  bool
@return: C{True} if all pages in this region are marked as
    copy-on-write. This means the pages are writeable, but changes
    are not propagated to disk.
@note:
    Tipically data sections in executable images are marked like this.

### Function: is_executable(self)

**Description:** @rtype:  bool
@return: C{True} if all pages in this region are executable.
@note: Executable pages are always readable.

### Function: is_executable_and_writeable(self)

**Description:** @rtype:  bool
@return: C{True} if all pages in this region are executable and
    writeable.
@note: The presence of such pages make memory corruption
    vulnerabilities much easier to exploit.

### Function: __init__(self, AttributeList)

**Description:** @type  AttributeList: list of tuple( int, ctypes-compatible object )
@param AttributeList: List of (Attribute, Value) pairs.

### Function: __del__(self)

### Function: __copy__(self)

### Function: __deepcopy__(self)

### Function: value(self)

### Function: _as_parameter_(self)

### Function: from_param(value)
