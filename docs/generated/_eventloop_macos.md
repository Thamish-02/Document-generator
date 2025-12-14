## AI Summary

A file named _eventloop_macos.py.


### Function: _utf8(s)

**Description:** ensure utf8 bytes

### Function: n(name)

**Description:** create a selector name (for ObjC methods)

### Function: C(classname)

**Description:** get an ObjC Class by name

### Function: _NSApp()

**Description:** Return the global NSApplication instance (NSApp)

### Function: _wake(NSApp)

**Description:** Wake the Application

### Function: stop(timer, loop)

**Description:** Callback to fire when there's input to be read

### Function: _stop_after(delay)

**Description:** Register callback to stop eventloop after a delay

### Function: mainloop(duration)

**Description:** run the Cocoa eventloop for the specified duration (seconds)
