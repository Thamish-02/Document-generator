## AI Summary

A file named user32.py.


### Function: MAKE_WPARAM(wParam)

**Description:** Convert arguments to the WPARAM type.
Used automatically by SendMessage, PostMessage, etc.
You shouldn't need to call this function.

### Function: MAKE_LPARAM(lParam)

**Description:** Convert arguments to the LPARAM type.
Used automatically by SendMessage, PostMessage, etc.
You shouldn't need to call this function.

## Class: __WindowEnumerator

**Description:** Window enumerator class. Used internally by the window enumeration APIs.

## Class: WINDOWPLACEMENT

## Class: GUITHREADINFO

## Class: Point

**Description:** Python wrapper over the L{POINT} class.

@type x: int
@ivar x: Horizontal coordinate
@type y: int
@ivar y: Vertical coordinate

## Class: Rect

**Description:** Python wrapper over the L{RECT} class.

@type   left: int
@ivar   left: Horizontal coordinate for the top left corner.
@type    top: int
@ivar    top: Vertical coordinate for the top left corner.
@type  right: int
@ivar  right: Horizontal coordinate for the bottom right corner.
@type bottom: int
@ivar bottom: Vertical coordinate for the bottom right corner.

@type  width: int
@ivar  width: Width in pixels. Same as C{right - left}.
@type height: int
@ivar height: Height in pixels. Same as C{bottom - top}.

## Class: WindowPlacement

**Description:** Python wrapper over the L{WINDOWPLACEMENT} class.

### Function: SetLastErrorEx(dwErrCode, dwType)

### Function: FindWindowA(lpClassName, lpWindowName)

### Function: FindWindowW(lpClassName, lpWindowName)

### Function: FindWindowExA(hwndParent, hwndChildAfter, lpClassName, lpWindowName)

### Function: FindWindowExW(hwndParent, hwndChildAfter, lpClassName, lpWindowName)

### Function: GetClassNameA(hWnd)

### Function: GetClassNameW(hWnd)

### Function: GetWindowTextA(hWnd)

### Function: GetWindowTextW(hWnd)

### Function: SetWindowTextA(hWnd, lpString)

### Function: SetWindowTextW(hWnd, lpString)

### Function: GetWindowLongA(hWnd, nIndex)

### Function: GetWindowLongW(hWnd, nIndex)

### Function: SetWindowLongA(hWnd, nIndex, dwNewLong)

### Function: SetWindowLongW(hWnd, nIndex, dwNewLong)

### Function: GetShellWindow()

### Function: GetWindowThreadProcessId(hWnd)

### Function: GetWindow(hWnd, uCmd)

### Function: GetParent(hWnd)

### Function: GetAncestor(hWnd, gaFlags)

### Function: EnableWindow(hWnd, bEnable)

### Function: ShowWindow(hWnd, nCmdShow)

### Function: ShowWindowAsync(hWnd, nCmdShow)

### Function: GetDesktopWindow()

### Function: GetForegroundWindow()

### Function: IsWindow(hWnd)

### Function: IsWindowVisible(hWnd)

### Function: IsWindowEnabled(hWnd)

### Function: IsZoomed(hWnd)

### Function: IsIconic(hWnd)

### Function: IsChild(hWnd)

### Function: WindowFromPoint(point)

### Function: ChildWindowFromPoint(hWndParent, point)

### Function: RealChildWindowFromPoint(hWndParent, ptParentClientCoords)

### Function: ScreenToClient(hWnd, lpPoint)

### Function: ClientToScreen(hWnd, lpPoint)

### Function: MapWindowPoints(hWndFrom, hWndTo, lpPoints)

### Function: SetForegroundWindow(hWnd)

### Function: GetWindowPlacement(hWnd)

### Function: SetWindowPlacement(hWnd, lpwndpl)

### Function: GetWindowRect(hWnd)

### Function: GetClientRect(hWnd)

### Function: MoveWindow(hWnd, X, Y, nWidth, nHeight, bRepaint)

### Function: GetGUIThreadInfo(idThread)

## Class: __EnumWndProc

### Function: EnumWindows()

## Class: __EnumThreadWndProc

### Function: EnumThreadWindows(dwThreadId)

## Class: __EnumChildProc

### Function: EnumChildWindows(hWndParent)

### Function: SendMessageA(hWnd, Msg, wParam, lParam)

### Function: SendMessageW(hWnd, Msg, wParam, lParam)

### Function: PostMessageA(hWnd, Msg, wParam, lParam)

### Function: PostMessageW(hWnd, Msg, wParam, lParam)

### Function: PostThreadMessageA(idThread, Msg, wParam, lParam)

### Function: PostThreadMessageW(idThread, Msg, wParam, lParam)

### Function: SendMessageTimeoutA(hWnd, Msg, wParam, lParam, fuFlags, uTimeout)

### Function: SendMessageTimeoutW(hWnd, Msg, wParam, lParam)

### Function: SendNotifyMessageA(hWnd, Msg, wParam, lParam)

### Function: SendNotifyMessageW(hWnd, Msg, wParam, lParam)

### Function: SendDlgItemMessageA(hDlg, nIDDlgItem, Msg, wParam, lParam)

### Function: SendDlgItemMessageW(hDlg, nIDDlgItem, Msg, wParam, lParam)

### Function: WaitForInputIdle(hProcess, dwMilliseconds)

### Function: RegisterWindowMessageA(lpString)

### Function: RegisterWindowMessageW(lpString)

### Function: RegisterClipboardFormatA(lpString)

### Function: RegisterClipboardFormatW(lpString)

### Function: GetPropA(hWnd, lpString)

### Function: GetPropW(hWnd, lpString)

### Function: SetPropA(hWnd, lpString, hData)

### Function: SetPropW(hWnd, lpString, hData)

### Function: RemovePropA(hWnd, lpString)

### Function: RemovePropW(hWnd, lpString)

### Function: __init__(self)

### Function: __call__(self, hwnd, lParam)

### Function: __init__(self, x, y)

**Description:** @see: L{POINT}
@type  x: int
@param x: Horizontal coordinate
@type  y: int
@param y: Vertical coordinate

### Function: __iter__(self)

### Function: __len__(self)

### Function: __getitem__(self, index)

### Function: __setitem__(self, index, value)

### Function: _as_parameter_(self)

**Description:** Compatibility with ctypes.
Allows passing transparently a Point object to an API call.

### Function: screen_to_client(self, hWnd)

**Description:** Translates window screen coordinates to client coordinates.

@see: L{client_to_screen}, L{translate}

@type  hWnd: int or L{HWND} or L{system.Window}
@param hWnd: Window handle.

@rtype:  L{Point}
@return: New object containing the translated coordinates.

### Function: client_to_screen(self, hWnd)

**Description:** Translates window client coordinates to screen coordinates.

@see: L{screen_to_client}, L{translate}

@type  hWnd: int or L{HWND} or L{system.Window}
@param hWnd: Window handle.

@rtype:  L{Point}
@return: New object containing the translated coordinates.

### Function: translate(self, hWndFrom, hWndTo)

**Description:** Translate coordinates from one window to another.

@note: To translate multiple points it's more efficient to use the
    L{MapWindowPoints} function instead.

@see: L{client_to_screen}, L{screen_to_client}

@type  hWndFrom: int or L{HWND} or L{system.Window}
@param hWndFrom: Window handle to translate from.
    Use C{HWND_DESKTOP} for screen coordinates.

@type  hWndTo: int or L{HWND} or L{system.Window}
@param hWndTo: Window handle to translate to.
    Use C{HWND_DESKTOP} for screen coordinates.

@rtype:  L{Point}
@return: New object containing the translated coordinates.

### Function: __init__(self, left, top, right, bottom)

**Description:** @see: L{RECT}
@type    left: int
@param   left: Horizontal coordinate for the top left corner.
@type     top: int
@param    top: Vertical coordinate for the top left corner.
@type   right: int
@param  right: Horizontal coordinate for the bottom right corner.
@type  bottom: int
@param bottom: Vertical coordinate for the bottom right corner.

### Function: __iter__(self)

### Function: __len__(self)

### Function: __getitem__(self, index)

### Function: __setitem__(self, index, value)

### Function: _as_parameter_(self)

**Description:** Compatibility with ctypes.
Allows passing transparently a Point object to an API call.

### Function: __get_width(self)

### Function: __get_height(self)

### Function: __set_width(self, value)

### Function: __set_height(self, value)

### Function: screen_to_client(self, hWnd)

**Description:** Translates window screen coordinates to client coordinates.

@see: L{client_to_screen}, L{translate}

@type  hWnd: int or L{HWND} or L{system.Window}
@param hWnd: Window handle.

@rtype:  L{Rect}
@return: New object containing the translated coordinates.

### Function: client_to_screen(self, hWnd)

**Description:** Translates window client coordinates to screen coordinates.

@see: L{screen_to_client}, L{translate}

@type  hWnd: int or L{HWND} or L{system.Window}
@param hWnd: Window handle.

@rtype:  L{Rect}
@return: New object containing the translated coordinates.

### Function: translate(self, hWndFrom, hWndTo)

**Description:** Translate coordinates from one window to another.

@see: L{client_to_screen}, L{screen_to_client}

@type  hWndFrom: int or L{HWND} or L{system.Window}
@param hWndFrom: Window handle to translate from.
    Use C{HWND_DESKTOP} for screen coordinates.

@type  hWndTo: int or L{HWND} or L{system.Window}
@param hWndTo: Window handle to translate to.
    Use C{HWND_DESKTOP} for screen coordinates.

@rtype:  L{Rect}
@return: New object containing the translated coordinates.

### Function: __init__(self, wp)

**Description:** @type  wp: L{WindowPlacement} or L{WINDOWPLACEMENT}
@param wp: Another window placement object.

### Function: _as_parameter_(self)

**Description:** Compatibility with ctypes.
Allows passing transparently a Point object to an API call.

### Function: GetWindowLongPtrA(hWnd, nIndex)

### Function: GetWindowLongPtrW(hWnd, nIndex)

### Function: SetWindowLongPtrA(hWnd, nIndex, dwNewLong)

### Function: SetWindowLongPtrW(hWnd, nIndex, dwNewLong)
