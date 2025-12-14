## AI Summary

A file named backgroundjobs.py.


## Class: BackgroundJobManager

**Description:** Class to manage a pool of backgrounded threaded jobs.

Below, we assume that 'jobs' is a BackgroundJobManager instance.

Usage summary (see the method docstrings for details):

  jobs.new(...) -> start a new job
  
  jobs() or jobs.status() -> print status summary of all jobs

  jobs[N] -> returns job number N.

  foo = jobs[N].result -> assign to variable foo the result of job N

  jobs[N].traceback() -> print the traceback of dead job N

  jobs.remove(N) -> remove (finished) job N

  jobs.flush() -> remove all finished jobs
  
As a convenience feature, BackgroundJobManager instances provide the
utility result and traceback methods which retrieve the corresponding
information from the jobs list:

  jobs.result(N) <--> jobs[N].result
  jobs.traceback(N) <--> jobs[N].traceback()

While this appears minor, it allows you to use tab completion
interactively on the job manager instance.

## Class: BackgroundJobBase

**Description:** Base class to build BackgroundJob classes.

The derived classes must implement:

- Their own __init__, since the one here raises NotImplementedError.  The
  derived constructor must call self._init() at the end, to provide common
  initialization.

- A strform attribute used in calls to __str__.

- A call() method, which will make the actual execution call and must
  return a value to be held in the 'result' field of the job object.

## Class: BackgroundJobExpr

**Description:** Evaluate an expression as a background job (uses a separate thread).

## Class: BackgroundJobFunc

**Description:** Run a function call as a background job (uses a separate thread).

### Function: __init__(self)

### Function: running(self)

### Function: dead(self)

### Function: completed(self)

### Function: new(self, func_or_exp)

**Description:** Add a new background job and start it in a separate thread.

There are two types of jobs which can be created:

1. Jobs based on expressions which can be passed to an eval() call.
The expression must be given as a string.  For example:

  job_manager.new('myfunc(x,y,z=1)'[,glob[,loc]])

The given expression is passed to eval(), along with the optional
global/local dicts provided.  If no dicts are given, they are
extracted automatically from the caller's frame.

A Python statement is NOT a valid eval() expression.  Basically, you
can only use as an eval() argument something which can go on the right
of an '=' sign and be assigned to a variable.

For example,"print 'hello'" is not valid, but '2+3' is.

2. Jobs given a function object, optionally passing additional
positional arguments:

  job_manager.new(myfunc, x, y)

The function is called with the given arguments.

If you need to pass keyword arguments to your function, you must
supply them as a dict named kw:

  job_manager.new(myfunc, x, y, kw=dict(z=1))

The reason for this asymmetry is that the new() method needs to
maintain access to its own keywords, and this prevents name collisions
between arguments to new() and arguments to your own functions.

In both cases, the result is stored in the job.result field of the
background job object.

You can set `daemon` attribute of the thread by giving the keyword
argument `daemon`.

Notes and caveats:

1. All threads running share the same standard output.  Thus, if your
background jobs generate output, it will come out on top of whatever
you are currently writing.  For this reason, background jobs are best
used with silent functions which simply return their output.

2. Threads also all work within the same global namespace, and this
system does not lock interactive variables.  So if you send job to the
background which operates on a mutable object for a long time, and
start modifying that same mutable object interactively (or in another
backgrounded job), all sorts of bizarre behaviour will occur.

3. If a background job is spending a lot of time inside a C extension
module which does not release the Python Global Interpreter Lock
(GIL), this will block the IPython prompt.  This is simply because the
Python interpreter can only switch between threads at Python
bytecodes.  While the execution is inside C code, the interpreter must
simply wait unless the extension module releases the GIL.

4. There is no way, due to limitations in the Python threads library,
to kill a thread once it has started.

### Function: __getitem__(self, job_key)

### Function: __call__(self)

**Description:** An alias to self.status(),

This allows you to simply call a job manager instance much like the
Unix `jobs` shell command.

### Function: _update_status(self)

**Description:** Update the status of the job lists.

This method moves finished jobs to one of two lists:
  - self.completed: jobs which completed successfully
  - self.dead: jobs which finished but died.

It also copies those jobs to corresponding _report lists.  These lists
are used to report jobs completed/dead since the last update, and are
then cleared by the reporting function after each call.

### Function: _group_report(self, group, name)

**Description:** Report summary for a given job group.

Return True if the group had any elements.

### Function: _group_flush(self, group, name)

**Description:** Flush a given job group

Return True if the group had any elements.

### Function: _status_new(self)

**Description:** Print the status of newly finished jobs.

Return True if any new jobs are reported.

This call resets its own state every time, so it only reports jobs
which have finished since the last time it was called.

### Function: status(self, verbose)

**Description:** Print a status of all jobs currently being managed.

### Function: remove(self, num)

**Description:** Remove a finished (completed or dead) job.

### Function: flush(self)

**Description:** Flush all finished jobs (completed and dead) from lists.

Running jobs are never flushed.

It first calls _status_new(), to update info. If any jobs have
completed since the last _status_new() call, the flush operation
aborts.

### Function: result(self, num)

**Description:** result(N) -> return the result of job N.

### Function: _traceback(self, job)

### Function: traceback(self, job)

### Function: __init__(self)

**Description:** Must be implemented in subclasses.

Subclasses must call :meth:`_init` for standard initialisation.

### Function: _init(self)

**Description:** Common initialization for all BackgroundJob objects

### Function: __str__(self)

### Function: __repr__(self)

### Function: traceback(self)

### Function: run(self)

### Function: __init__(self, expression, glob, loc)

**Description:** Create a new job from a string which can be fed to eval().

global/locals dicts can be provided, which will be passed to the eval
call.

### Function: call(self)

### Function: __init__(self, func)

**Description:** Create a new job from a callable object.

Any positional arguments and keyword args given to this constructor
after the initial callable are passed directly to it.

### Function: call(self)
