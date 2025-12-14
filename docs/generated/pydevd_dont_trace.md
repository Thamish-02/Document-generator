## AI Summary

A file named pydevd_dont_trace.py.


### Function: default_should_trace_hook(code, absolute_filename)

**Description:** Return True if this frame should be traced, False if tracing should be blocked.

### Function: clear_trace_filter_cache()

**Description:** Clear the trace filter cache.
Call this after reloading.

### Function: trace_filter(mode)

**Description:** Set the trace filter mode.

mode: Whether to enable the trace hook.
  True: Trace filtering on (skipping methods tagged @DontTrace)
  False: Trace filtering off (trace methods tagged @DontTrace)
  None/default: Toggle trace filtering.
