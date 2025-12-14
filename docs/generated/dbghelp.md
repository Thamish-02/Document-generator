## AI Summary

A file named dbghelp.py.


### Function: _load_latest_dbghelp_dll()

## Class: IMAGEHLP_MODULE

## Class: IMAGEHLP_MODULE64

## Class: IMAGEHLP_MODULEW

## Class: IMAGEHLP_MODULEW64

### Function: MakeSureDirectoryPathExistsA(DirPath)

### Function: SymInitializeA(hProcess, UserSearchPath, fInvadeProcess)

### Function: SymCleanup(hProcess)

### Function: SymRefreshModuleList(hProcess)

### Function: SymSetParentWindow(hwnd)

### Function: SymSetOptions(SymOptions)

### Function: SymGetOptions()

### Function: SymLoadModuleA(hProcess, hFile, ImageName, ModuleName, BaseOfDll, SizeOfDll)

### Function: SymLoadModule64A(hProcess, hFile, ImageName, ModuleName, BaseOfDll, SizeOfDll)

### Function: SymUnloadModule(hProcess, BaseOfDll)

### Function: SymUnloadModule64(hProcess, BaseOfDll)

### Function: SymGetModuleInfoA(hProcess, dwAddr)

### Function: SymGetModuleInfoW(hProcess, dwAddr)

### Function: SymGetModuleInfo64A(hProcess, dwAddr)

### Function: SymGetModuleInfo64W(hProcess, dwAddr)

### Function: SymEnumerateModulesA(hProcess, EnumModulesCallback, UserContext)

### Function: SymEnumerateModulesW(hProcess, EnumModulesCallback, UserContext)

### Function: SymEnumerateModules64A(hProcess, EnumModulesCallback, UserContext)

### Function: SymEnumerateModules64W(hProcess, EnumModulesCallback, UserContext)

### Function: SymEnumerateSymbolsA(hProcess, BaseOfDll, EnumSymbolsCallback, UserContext)

### Function: SymEnumerateSymbolsW(hProcess, BaseOfDll, EnumSymbolsCallback, UserContext)

### Function: SymEnumerateSymbols64A(hProcess, BaseOfDll, EnumSymbolsCallback, UserContext)

### Function: SymEnumerateSymbols64W(hProcess, BaseOfDll, EnumSymbolsCallback, UserContext)

### Function: UnDecorateSymbolNameA(DecoratedName, Flags)

### Function: UnDecorateSymbolNameW(DecoratedName, Flags)

### Function: SymGetSearchPathA(hProcess)

### Function: SymGetSearchPathW(hProcess)

### Function: SymSetSearchPathA(hProcess, SearchPath)

### Function: SymSetSearchPathW(hProcess, SearchPath)

### Function: SymGetHomeDirectoryA(type)

### Function: SymGetHomeDirectoryW(type)

### Function: SymSetHomeDirectoryA(hProcess, dir)

### Function: SymSetHomeDirectoryW(hProcess, dir)

## Class: SYM_INFO

## Class: SYM_INFOW

### Function: SymFromName(hProcess, Name)

### Function: SymFromNameW(hProcess, Name)

### Function: SymFromAddr(hProcess, Address)

### Function: SymFromAddrW(hProcess, Address)

## Class: IMAGEHLP_SYMBOL64

## Class: IMAGEHLP_SYMBOLW64

### Function: SymGetSymFromAddr64(hProcess, Address)

## Class: API_VERSION

### Function: ImagehlpApiVersion()

### Function: ImagehlpApiVersionEx(MajorVersion, MinorVersion, Revision)

## Class: ADDRESS64

## Class: KDHELP64

## Class: STACKFRAME64

### Function: StackWalk64(MachineType, hProcess, hThread, StackFrame, ContextRecord, ReadMemoryRoutine, FunctionTableAccessRoutine, GetModuleBaseRoutine, TranslateAddress)
