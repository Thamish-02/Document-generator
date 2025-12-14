## AI Summary

A file named window.py.


## Class: Window

**Description:** Interface to an open window in the current desktop.

@group Properties:
    get_handle, get_pid, get_tid,
    get_process, get_thread,
    set_process, set_thread,
    get_classname, get_style, get_extended_style,
    get_text, set_text,
    get_placement, set_placement,
    get_screen_rect, get_client_rect,
    screen_to_client, client_to_screen

@group State:
    is_valid, is_visible, is_enabled, is_maximized, is_minimized, is_child,
    is_zoomed, is_iconic

@group Navigation:
    get_parent, get_children, get_root, get_tree,
    get_child_at

@group Instrumentation:
    enable, disable, show, hide, maximize, minimize, restore, move, kill

@group Low-level access:
    send, post

@type hWnd: int
@ivar hWnd: Window handle.

@type dwProcessId: int
@ivar dwProcessId: Global ID of the process that owns this window.

@type dwThreadId: int
@ivar dwThreadId: Global ID of the thread that owns this window.

@type process: L{Process}
@ivar process: Process that owns this window.
    Use the L{get_process} method instead.

@type thread: L{Thread}
@ivar thread: Thread that owns this window.
    Use the L{get_thread} method instead.

@type classname: str
@ivar classname: Window class name.

@type text: str
@ivar text: Window text (caption).

@type placement: L{win32.WindowPlacement}
@ivar placement: Window placement in the desktop.

### Function: __init__(self, hWnd, process, thread)

**Description:** @type  hWnd: int or L{win32.HWND}
@param hWnd: Window handle.

@type  process: L{Process}
@param process: (Optional) Process that owns this window.

@type  thread: L{Thread}
@param thread: (Optional) Thread that owns this window.

### Function: _as_parameter_(self)

**Description:** Compatibility with ctypes.
Allows passing transparently a Window object to an API call.

### Function: get_handle(self)

**Description:** @rtype:  int
@return: Window handle.
@raise ValueError: No window handle set.

### Function: get_pid(self)

**Description:** @rtype:  int
@return: Global ID of the process that owns this window.

### Function: get_tid(self)

**Description:** @rtype:  int
@return: Global ID of the thread that owns this window.

### Function: __get_pid_and_tid(self)

**Description:** Internally used by get_pid() and get_tid().

### Function: __load_Process_class(self)

### Function: __load_Thread_class(self)

### Function: get_process(self)

**Description:** @rtype:  L{Process}
@return: Parent Process object.

### Function: set_process(self, process)

**Description:** Manually set the parent process. Use with care!

@type  process: L{Process}
@param process: (Optional) Process object. Use C{None} to autodetect.

### Function: get_thread(self)

**Description:** @rtype:  L{Thread}
@return: Parent Thread object.

### Function: set_thread(self, thread)

**Description:** Manually set the thread process. Use with care!

@type  thread: L{Thread}
@param thread: (Optional) Thread object. Use C{None} to autodetect.

### Function: __get_window(self, hWnd)

**Description:** User internally to get another Window from this one.
It'll try to copy the parent Process and Thread references if possible.

### Function: get_classname(self)

**Description:** @rtype:  str
@return: Window class name.

@raise WindowsError: An error occured while processing this request.

### Function: get_style(self)

**Description:** @rtype:  int
@return: Window style mask.

@raise WindowsError: An error occured while processing this request.

### Function: get_extended_style(self)

**Description:** @rtype:  int
@return: Window extended style mask.

@raise WindowsError: An error occured while processing this request.

### Function: get_text(self)

**Description:** @see:    L{set_text}
@rtype:  str
@return: Window text (caption) on success, C{None} on error.

### Function: set_text(self, text)

**Description:** Set the window text (caption).

@see: L{get_text}

@type  text: str
@param text: New window text.

@raise WindowsError: An error occured while processing this request.

### Function: get_placement(self)

**Description:** Retrieve the window placement in the desktop.

@see: L{set_placement}

@rtype:  L{win32.WindowPlacement}
@return: Window placement in the desktop.

@raise WindowsError: An error occured while processing this request.

### Function: set_placement(self, placement)

**Description:** Set the window placement in the desktop.

@see: L{get_placement}

@type  placement: L{win32.WindowPlacement}
@param placement: Window placement in the desktop.

@raise WindowsError: An error occured while processing this request.

### Function: get_screen_rect(self)

**Description:** Get the window coordinates in the desktop.

@rtype:  L{win32.Rect}
@return: Rectangle occupied by the window in the desktop.

@raise WindowsError: An error occured while processing this request.

### Function: get_client_rect(self)

**Description:** Get the window's client area coordinates in the desktop.

@rtype:  L{win32.Rect}
@return: Rectangle occupied by the window's client area in the desktop.

@raise WindowsError: An error occured while processing this request.

### Function: client_to_screen(self, x, y)

**Description:** Translates window client coordinates to screen coordinates.

@note: This is a simplified interface to some of the functionality of
    the L{win32.Point} class.

@see: {win32.Point.client_to_screen}

@type  x: int
@param x: Horizontal coordinate.
@type  y: int
@param y: Vertical coordinate.

@rtype:  tuple( int, int )
@return: Translated coordinates in a tuple (x, y).

@raise WindowsError: An error occured while processing this request.

### Function: screen_to_client(self, x, y)

**Description:** Translates window screen coordinates to client coordinates.

@note: This is a simplified interface to some of the functionality of
    the L{win32.Point} class.

@see: {win32.Point.screen_to_client}

@type  x: int
@param x: Horizontal coordinate.
@type  y: int
@param y: Vertical coordinate.

@rtype:  tuple( int, int )
@return: Translated coordinates in a tuple (x, y).

@raise WindowsError: An error occured while processing this request.

### Function: get_parent(self)

**Description:** @see:    L{get_children}
@rtype:  L{Window} or None
@return: Parent window. Returns C{None} if the window has no parent.
@raise WindowsError: An error occured while processing this request.

### Function: get_children(self)

**Description:** @see:    L{get_parent}
@rtype:  list( L{Window} )
@return: List of child windows.
@raise WindowsError: An error occured while processing this request.

### Function: get_tree(self)

**Description:** @see:    L{get_root}
@rtype:  dict( L{Window} S{->} dict( ... ) )
@return: Dictionary of dictionaries forming a tree of child windows.
@raise WindowsError: An error occured while processing this request.

### Function: get_root(self)

**Description:** @see:    L{get_tree}
@rtype:  L{Window}
@return: If this is a child window, return the top-level window it
    belongs to.
    If this window is already a top-level window, returns itself.
@raise WindowsError: An error occured while processing this request.

### Function: get_child_at(self, x, y, bAllowTransparency)

**Description:** Get the child window located at the given coordinates. If no such
window exists an exception is raised.

@see: L{get_children}

@type  x: int
@param x: Horizontal coordinate.

@type  y: int
@param y: Vertical coordinate.

@type  bAllowTransparency: bool
@param bAllowTransparency: If C{True} transparent areas in windows are
    ignored, returning the window behind them. If C{False} transparent
    areas are treated just like any other area.

@rtype:  L{Window}
@return: Child window at the requested position, or C{None} if there
    is no window at those coordinates.

### Function: is_valid(self)

**Description:** @rtype:  bool
@return: C{True} if the window handle is still valid.

### Function: is_visible(self)

**Description:** @see: {show}, {hide}
@rtype:  bool
@return: C{True} if the window is in a visible state.

### Function: is_enabled(self)

**Description:** @see: {enable}, {disable}
@rtype:  bool
@return: C{True} if the window is in an enabled state.

### Function: is_maximized(self)

**Description:** @see: L{maximize}
@rtype:  bool
@return: C{True} if the window is maximized.

### Function: is_minimized(self)

**Description:** @see: L{minimize}
@rtype:  bool
@return: C{True} if the window is minimized.

### Function: is_child(self)

**Description:** @see: L{get_parent}
@rtype:  bool
@return: C{True} if the window is a child window.

### Function: enable(self)

**Description:** Enable the user input for the window.

@see: L{disable}

@raise WindowsError: An error occured while processing this request.

### Function: disable(self)

**Description:** Disable the user input for the window.

@see: L{enable}

@raise WindowsError: An error occured while processing this request.

### Function: show(self, bAsync)

**Description:** Make the window visible.

@see: L{hide}

@type  bAsync: bool
@param bAsync: Perform the request asynchronously.

@raise WindowsError: An error occured while processing this request.

### Function: hide(self, bAsync)

**Description:** Make the window invisible.

@see: L{show}

@type  bAsync: bool
@param bAsync: Perform the request asynchronously.

@raise WindowsError: An error occured while processing this request.

### Function: maximize(self, bAsync)

**Description:** Maximize the window.

@see: L{minimize}, L{restore}

@type  bAsync: bool
@param bAsync: Perform the request asynchronously.

@raise WindowsError: An error occured while processing this request.

### Function: minimize(self, bAsync)

**Description:** Minimize the window.

@see: L{maximize}, L{restore}

@type  bAsync: bool
@param bAsync: Perform the request asynchronously.

@raise WindowsError: An error occured while processing this request.

### Function: restore(self, bAsync)

**Description:** Unmaximize and unminimize the window.

@see: L{maximize}, L{minimize}

@type  bAsync: bool
@param bAsync: Perform the request asynchronously.

@raise WindowsError: An error occured while processing this request.

### Function: move(self, x, y, width, height, bRepaint)

**Description:** Moves and/or resizes the window.

@note: This is request is performed syncronously.

@type  x: int
@param x: (Optional) New horizontal coordinate.

@type  y: int
@param y: (Optional) New vertical coordinate.

@type  width: int
@param width: (Optional) Desired window width.

@type  height: int
@param height: (Optional) Desired window height.

@type  bRepaint: bool
@param bRepaint:
    (Optional) C{True} if the window should be redrawn afterwards.

@raise WindowsError: An error occured while processing this request.

### Function: kill(self)

**Description:** Signals the program to quit.

@note: This is an asyncronous request.

@raise WindowsError: An error occured while processing this request.

### Function: send(self, uMsg, wParam, lParam, dwTimeout)

**Description:** Send a low-level window message syncronically.

@type  uMsg: int
@param uMsg: Message code.

@param wParam:
    The type and meaning of this parameter depends on the message.

@param lParam:
    The type and meaning of this parameter depends on the message.

@param dwTimeout: Optional timeout for the operation.
    Use C{None} to wait indefinitely.

@rtype:  int
@return: The meaning of the return value depends on the window message.
    Typically a value of C{0} means an error occured. You can get the
    error code by calling L{win32.GetLastError}.

### Function: post(self, uMsg, wParam, lParam)

**Description:** Post a low-level window message asyncronically.

@type  uMsg: int
@param uMsg: Message code.

@param wParam:
    The type and meaning of this parameter depends on the message.

@param lParam:
    The type and meaning of this parameter depends on the message.

@raise WindowsError: An error occured while sending the message.
