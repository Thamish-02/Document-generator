## AI Summary

A file named timing.py.


### Function: timings_out(reps, func)

**Description:** timings_out(reps,func,*args,**kw) -> (t_total,t_per_call,output)

Execute a function reps times, return a tuple with the elapsed total
CPU time in seconds, the time per call and the function's output.

Under Unix, the return value is the sum of user+system time consumed by
the process, computed via the resource module.  This prevents problems
related to the wraparound effect which the time.clock() function has.

Under Windows the return value is in wall clock seconds. See the
documentation for the time module for more details.

### Function: timings(reps, func)

**Description:** timings(reps,func,*args,**kw) -> (t_total,t_per_call)

Execute a function reps times, return a tuple with the elapsed total CPU
time in seconds and the time per call. These are just the first two values
in timings_out().

### Function: timing(func)

**Description:** timing(func,*args,**kw) -> t_total

Execute a function once, return the elapsed total CPU time in
seconds. This is just the first value in timings_out().

### Function: clocku()

**Description:** clocku() -> floating point number

Return the *USER* CPU time in seconds since the start of the process.
This is done via a call to resource.getrusage, so it avoids the
wraparound problems in time.clock().

### Function: clocks()

**Description:** clocks() -> floating point number

Return the *SYSTEM* CPU time in seconds since the start of the process.
This is done via a call to resource.getrusage, so it avoids the
wraparound problems in time.clock().

### Function: clock()

**Description:** clock() -> floating point number

Return the *TOTAL USER+SYSTEM* CPU time in seconds since the start of
the process.  This is done via a call to resource.getrusage, so it
avoids the wraparound problems in time.clock().

### Function: clock2()

**Description:** clock2() -> (t_user,t_system)

Similar to clock(), but return a tuple of user/system times.

### Function: clock2()

**Description:** Under windows, system CPU time can't be measured.

This just returns process_time() and zero.
