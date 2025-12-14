## AI Summary

A file named advapi32.py.


## Class: LUID

## Class: LUID_AND_ATTRIBUTES

## Class: TOKEN_PRIVILEGES

## Class: SID_AND_ATTRIBUTES

## Class: TOKEN_USER

## Class: TOKEN_MANDATORY_LABEL

## Class: TOKEN_OWNER

## Class: TOKEN_PRIMARY_GROUP

## Class: TOKEN_APPCONTAINER_INFORMATION

## Class: TOKEN_ORIGIN

## Class: TOKEN_LINKED_TOKEN

## Class: TOKEN_STATISTICS

## Class: _WAITCHAIN_NODE_INFO_STRUCT_1

## Class: _WAITCHAIN_NODE_INFO_STRUCT_2

## Class: _WAITCHAIN_NODE_INFO_UNION

## Class: WAITCHAIN_NODE_INFO

## Class: WaitChainNodeInfo

**Description:** Represents a node in the wait chain.

It's a wrapper on the L{WAITCHAIN_NODE_INFO} structure.

The following members are defined only
if the node is of L{WctThreadType} type:
 - C{ProcessId}
 - C{ThreadId}
 - C{WaitTime}
 - C{ContextSwitches}

@see: L{GetThreadWaitChain}

@type ObjectName: unicode
@ivar ObjectName: Object name. May be an empty string.

@type ObjectType: int
@ivar ObjectType: Object type.
    Should be one of the following values:
     - L{WctCriticalSectionType}
     - L{WctSendMessageType}
     - L{WctMutexType}
     - L{WctAlpcType}
     - L{WctComType}
     - L{WctThreadWaitType}
     - L{WctProcessWaitType}
     - L{WctThreadType}
     - L{WctComActivationType}
     - L{WctUnknownType}

@type ObjectStatus: int
@ivar ObjectStatus: Wait status.
    Should be one of the following values:
     - L{WctStatusNoAccess} I{(ACCESS_DENIED for this object)}
     - L{WctStatusRunning} I{(Thread status)}
     - L{WctStatusBlocked} I{(Thread status)}
     - L{WctStatusPidOnly} I{(Thread status)}
     - L{WctStatusPidOnlyRpcss} I{(Thread status)}
     - L{WctStatusOwned} I{(Dispatcher object status)}
     - L{WctStatusNotOwned} I{(Dispatcher object status)}
     - L{WctStatusAbandoned} I{(Dispatcher object status)}
     - L{WctStatusUnknown} I{(All objects)}
     - L{WctStatusError} I{(All objects)}

@type ProcessId: int
@ivar ProcessId: Process global ID.

@type ThreadId: int
@ivar ThreadId: Thread global ID.

@type WaitTime: int
@ivar WaitTime: Wait time.

@type ContextSwitches: int
@ivar ContextSwitches: Number of context switches.

## Class: ThreadWaitChainSessionHandle

**Description:** Thread wait chain session handle.

Returned by L{OpenThreadWaitChainSession}.

@see: L{Handle}

## Class: SERVICE_STATUS

## Class: SERVICE_STATUS_PROCESS

## Class: ENUM_SERVICE_STATUSA

## Class: ENUM_SERVICE_STATUSW

## Class: ENUM_SERVICE_STATUS_PROCESSA

## Class: ENUM_SERVICE_STATUS_PROCESSW

## Class: ServiceStatus

**Description:** Wrapper for the L{SERVICE_STATUS} structure.

## Class: ServiceStatusProcess

**Description:** Wrapper for the L{SERVICE_STATUS_PROCESS} structure.

## Class: ServiceStatusEntry

**Description:** Service status entry returned by L{EnumServicesStatus}.

## Class: ServiceStatusProcessEntry

**Description:** Service status entry returned by L{EnumServicesStatusEx}.

## Class: TokenHandle

**Description:** Access token handle.

@see: L{Handle}

## Class: RegistryKeyHandle

**Description:** Registry key handle.

## Class: SaferLevelHandle

**Description:** Safer level handle.

@see: U{http://msdn.microsoft.com/en-us/library/ms722425(VS.85).aspx}

## Class: ServiceHandle

**Description:** Service handle.

@see: U{http://msdn.microsoft.com/en-us/library/windows/desktop/ms684330(v=vs.85).aspx}

## Class: ServiceControlManagerHandle

**Description:** Service Control Manager (SCM) handle.

@see: U{http://msdn.microsoft.com/en-us/library/windows/desktop/ms684323(v=vs.85).aspx}

### Function: GetUserNameA()

### Function: GetUserNameW()

### Function: LookupAccountSidA(lpSystemName, lpSid)

### Function: LookupAccountSidW(lpSystemName, lpSid)

### Function: ConvertSidToStringSidA(Sid)

### Function: ConvertSidToStringSidW(Sid)

### Function: ConvertStringSidToSidA(StringSid)

### Function: ConvertStringSidToSidW(StringSid)

### Function: IsValidSid(pSid)

### Function: EqualSid(pSid1, pSid2)

### Function: GetLengthSid(pSid)

### Function: CopySid(pSourceSid)

### Function: FreeSid(pSid)

### Function: OpenProcessToken(ProcessHandle, DesiredAccess)

### Function: OpenThreadToken(ThreadHandle, DesiredAccess, OpenAsSelf)

### Function: DuplicateToken(ExistingTokenHandle, ImpersonationLevel)

### Function: DuplicateTokenEx(hExistingToken, dwDesiredAccess, lpTokenAttributes, ImpersonationLevel, TokenType)

### Function: IsTokenRestricted(hTokenHandle)

### Function: LookupPrivilegeValueA(lpSystemName, lpName)

### Function: LookupPrivilegeValueW(lpSystemName, lpName)

### Function: LookupPrivilegeNameA(lpSystemName, lpLuid)

### Function: LookupPrivilegeNameW(lpSystemName, lpLuid)

### Function: AdjustTokenPrivileges(TokenHandle, NewState)

### Function: GetTokenInformation(hTokenHandle, TokenInformationClass)

### Function: _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)

### Function: CreateProcessWithLogonW(lpUsername, lpDomain, lpPassword, dwLogonFlags, lpApplicationName, lpCommandLine, dwCreationFlags, lpEnvironment, lpCurrentDirectory, lpStartupInfo)

### Function: CreateProcessWithTokenW(hToken, dwLogonFlags, lpApplicationName, lpCommandLine, dwCreationFlags, lpEnvironment, lpCurrentDirectory, lpStartupInfo)

### Function: CreateProcessAsUserA(hToken, lpApplicationName, lpCommandLine, lpProcessAttributes, lpThreadAttributes, bInheritHandles, dwCreationFlags, lpEnvironment, lpCurrentDirectory, lpStartupInfo)

### Function: CreateProcessAsUserW(hToken, lpApplicationName, lpCommandLine, lpProcessAttributes, lpThreadAttributes, bInheritHandles, dwCreationFlags, lpEnvironment, lpCurrentDirectory, lpStartupInfo)

### Function: OpenThreadWaitChainSession(Flags, callback)

### Function: GetThreadWaitChain(WctHandle, Context, Flags, ThreadId, NodeCount)

### Function: CloseThreadWaitChainSession(WctHandle)

### Function: SaferCreateLevel(dwScopeId, dwLevelId, OpenFlags)

### Function: SaferComputeTokenFromLevel(LevelHandle, InAccessToken, dwFlags)

### Function: SaferCloseLevel(hLevelHandle)

### Function: SaferiIsExecutableFileType(szFullPath, bFromShellExecute)

### Function: RegCloseKey(hKey)

### Function: RegConnectRegistryA(lpMachineName, hKey)

### Function: RegConnectRegistryW(lpMachineName, hKey)

### Function: RegCreateKeyA(hKey, lpSubKey)

### Function: RegCreateKeyW(hKey, lpSubKey)

### Function: RegOpenKeyA(hKey, lpSubKey)

### Function: RegOpenKeyW(hKey, lpSubKey)

### Function: RegOpenKeyExA(hKey, lpSubKey, samDesired)

### Function: RegOpenKeyExW(hKey, lpSubKey, samDesired)

### Function: RegOpenCurrentUser(samDesired)

### Function: RegOpenUserClassesRoot(hToken, samDesired)

### Function: RegQueryValueA(hKey, lpSubKey)

### Function: RegQueryValueW(hKey, lpSubKey)

### Function: _internal_RegQueryValueEx(ansi, hKey, lpValueName, bGetData)

### Function: _caller_RegQueryValueEx(ansi)

### Function: RegQueryValueExA(hKey, lpValueName, bGetData)

### Function: RegQueryValueExW(hKey, lpValueName, bGetData)

### Function: RegSetValueEx(hKey, lpValueName, lpData, dwType)

### Function: RegEnumKeyA(hKey, dwIndex)

### Function: RegEnumKeyW(hKey, dwIndex)

### Function: _internal_RegEnumValue(ansi, hKey, dwIndex, bGetData)

### Function: RegEnumValueA(hKey, dwIndex, bGetData)

### Function: RegEnumValueW(hKey, dwIndex, bGetData)

### Function: RegDeleteValueA(hKeySrc, lpValueName)

### Function: RegDeleteValueW(hKeySrc, lpValueName)

### Function: RegDeleteKeyValueA(hKeySrc, lpSubKey, lpValueName)

### Function: RegDeleteKeyValueW(hKeySrc, lpSubKey, lpValueName)

### Function: RegDeleteKeyA(hKeySrc, lpSubKey)

### Function: RegDeleteKeyW(hKeySrc, lpSubKey)

### Function: RegDeleteKeyExA(hKeySrc, lpSubKey, samDesired)

### Function: RegDeleteKeyExW(hKeySrc, lpSubKey, samDesired)

### Function: RegCopyTreeA(hKeySrc, lpSubKey, hKeyDest)

### Function: RegCopyTreeW(hKeySrc, lpSubKey, hKeyDest)

### Function: RegDeleteTreeA(hKey, lpSubKey)

### Function: RegDeleteTreeW(hKey, lpSubKey)

### Function: RegFlushKey(hKey)

### Function: CloseServiceHandle(hSCObject)

### Function: OpenSCManagerA(lpMachineName, lpDatabaseName, dwDesiredAccess)

### Function: OpenSCManagerW(lpMachineName, lpDatabaseName, dwDesiredAccess)

### Function: OpenServiceA(hSCManager, lpServiceName, dwDesiredAccess)

### Function: OpenServiceW(hSCManager, lpServiceName, dwDesiredAccess)

### Function: CreateServiceA(hSCManager, lpServiceName, lpDisplayName, dwDesiredAccess, dwServiceType, dwStartType, dwErrorControl, lpBinaryPathName, lpLoadOrderGroup, lpDependencies, lpServiceStartName, lpPassword)

### Function: CreateServiceW(hSCManager, lpServiceName, lpDisplayName, dwDesiredAccess, dwServiceType, dwStartType, dwErrorControl, lpBinaryPathName, lpLoadOrderGroup, lpDependencies, lpServiceStartName, lpPassword)

### Function: DeleteService(hService)

### Function: GetServiceKeyNameA(hSCManager, lpDisplayName)

### Function: GetServiceKeyNameW(hSCManager, lpDisplayName)

### Function: GetServiceDisplayNameA(hSCManager, lpServiceName)

### Function: GetServiceDisplayNameW(hSCManager, lpServiceName)

### Function: StartServiceA(hService, ServiceArgVectors)

### Function: StartServiceW(hService, ServiceArgVectors)

### Function: ControlService(hService, dwControl)

### Function: QueryServiceStatus(hService)

### Function: QueryServiceStatusEx(hService, InfoLevel)

### Function: EnumServicesStatusA(hSCManager, dwServiceType, dwServiceState)

### Function: EnumServicesStatusW(hSCManager, dwServiceType, dwServiceState)

### Function: EnumServicesStatusExA(hSCManager, InfoLevel, dwServiceType, dwServiceState, pszGroupName)

### Function: EnumServicesStatusExW(hSCManager, InfoLevel, dwServiceType, dwServiceState, pszGroupName)

### Function: __init__(self, aStructure)

### Function: __init__(self, aHandle)

**Description:** @type  aHandle: int
@param aHandle: Win32 handle value.

### Function: _close(self)

### Function: dup(self)

### Function: wait(self, dwMilliseconds)

### Function: inherit(self)

### Function: protectFromClose(self)

### Function: __init__(self, raw)

**Description:** @type  raw: L{SERVICE_STATUS}
@param raw: Raw structure for this service status data.

### Function: __init__(self, raw)

**Description:** @type  raw: L{SERVICE_STATUS_PROCESS}
@param raw: Raw structure for this service status data.

### Function: __init__(self, raw)

**Description:** @type  raw: L{ENUM_SERVICE_STATUSA} or L{ENUM_SERVICE_STATUSW}
@param raw: Raw structure for this service status entry.

### Function: __str__(self)

### Function: __init__(self, raw)

**Description:** @type  raw: L{ENUM_SERVICE_STATUS_PROCESSA} or L{ENUM_SERVICE_STATUS_PROCESSW}
@param raw: Raw structure for this service status entry.

### Function: __str__(self)

### Function: _close(self)

### Function: _close(self)

### Function: _close(self)

### Function: _close(self)
