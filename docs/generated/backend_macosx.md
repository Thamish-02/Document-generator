## AI Summary

A file named backend_macosx.py.


## Class: TimerMac

**Description:** Subclass of `.TimerBase` using CFRunLoop timer events.

### Function: _allow_interrupt_macos()

**Description:** A context manager that allows terminating a plot by sending a SIGINT.

## Class: FigureCanvasMac

## Class: NavigationToolbar2Mac

## Class: FigureManagerMac

## Class: _BackendMac

### Function: __init__(self, figure)

### Function: draw(self)

**Description:** Render the figure and update the macosx canvas.

### Function: draw_idle(self)

### Function: _single_shot_timer(self, callback)

**Description:** Add a single shot timer with the given callback

### Function: _draw_idle(self)

**Description:** Draw method for singleshot timer

This draw method can be added to a singleshot timer, which can
accumulate draws while the eventloop is spinning. This method will
then only draw the first time and short-circuit the others.

### Function: blit(self, bbox)

### Function: resize(self, width, height)

### Function: start_event_loop(self, timeout)

### Function: __init__(self, canvas)

### Function: draw_rubberband(self, event, x0, y0, x1, y1)

### Function: remove_rubberband(self)

### Function: save_figure(self)

### Function: __init__(self, canvas, num)

### Function: _close_button_pressed(self)

### Function: destroy(self)

### Function: start_main_loop(cls)

### Function: show(self)

### Function: callback_func(callback, timer)
