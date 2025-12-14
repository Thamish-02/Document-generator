## AI Summary

A file named pydevd.py.


### Function: install_breakpointhook(pydevd_breakpointhook)

## Class: PyDBCommandThread

## Class: CheckAliveThread

## Class: AbstractSingleNotificationBehavior

**Description:** The basic usage should be:

# Increment the request time for the suspend.
single_notification_behavior.increment_suspend_time()

# Notify that this is a pause request (when a pause, not a breakpoint).
single_notification_behavior.on_pause()

# Mark threads to be suspended.
set_suspend(...)

# On do_wait_suspend, use notify_thread_suspended:
def do_wait_suspend(...):
    with single_notification_behavior.notify_thread_suspended(thread_id, thread, reason):
        ...

## Class: ThreadsSuspendedSingleNotification

## Class: _Authentication

## Class: PyDB

**Description:** Main debugging class
Lots of stuff going on here:

PyDB starts two threads on startup that connect to remote debugger (RDB)
The threads continuously read & write commands to RDB.
PyDB communicates with these threads through command queues.
   Every RDB command is processed by calling process_net_command.
   Every PyDB net command is sent to the net by posting NetCommand to WriterThread queue

   Some commands need to be executed on the right thread (suspend/resume & friends)
   These are placed on the internal command queue.

## Class: IDAPMessagesListener

### Function: add_dap_messages_listener(dap_messages_listener)

**Description:** Adds a listener for the DAP (debug adapter protocol) messages.

:type dap_messages_listener: IDAPMessagesListener

:note: messages from the xml backend are not notified through this API.

:note: the notifications are sent from threads and they are not synchronized (so,
it's possible that a message is sent and received from different threads at the same time).

### Function: send_json_message(msg)

**Description:** API to send some custom json message.

:param dict|pydevd_schema.BaseSchema msg:
    The custom message to be sent.

:return bool:
    True if the message was added to the queue to be sent and False otherwise.

### Function: enable_qt_support(qt_support_mode)

### Function: start_dump_threads_thread(filename_template, timeout, recurrent)

**Description:** Helper to dump threads after a timeout.

:param filename_template:
    A template filename, such as 'c:/temp/thread_dump_%s.txt', where the %s will
    be replaced by the time for the dump.
:param timeout:
    The timeout (in seconds) for the dump.
:param recurrent:
    If True we'll keep on doing thread dumps.

### Function: dump_threads(stream)

**Description:** Helper to dump thread info (default is printing to stderr).

### Function: usage(doExit)

### Function: _init_stdout_redirect()

### Function: _init_stderr_redirect()

### Function: _enable_attach(address, dont_trace_start_patterns, dont_trace_end_patterns, patch_multiprocessing, access_token, client_access_token)

**Description:** Starts accepting connections at the given host/port. The debugger will not be initialized nor
configured, it'll only start accepting connections (and will have the tracing setup in this
thread).

Meant to be used with the DAP (Debug Adapter Protocol) with _wait_for_attach().

:param address: (host, port)
:type address: tuple(str, int)

### Function: _wait_for_attach(cancel)

**Description:** Meant to be called after _enable_attach() -- the current thread will only unblock after a
connection is in place and the DAP (Debug Adapter Protocol) sends the ConfigurationDone
request.

### Function: _is_attached()

**Description:** Can be called any time to check if the connection was established and the DAP (Debug Adapter Protocol) has sent
the ConfigurationDone request.

### Function: settrace(host, stdout_to_server, stderr_to_server, port, suspend, trace_only_current_thread, overwrite_prev_trace, patch_multiprocessing, stop_at_frame, block_until_connected, wait_for_ready_to_run, dont_trace_start_patterns, dont_trace_end_patterns, access_token, client_access_token, notify_stdin, protocol, ppid)

**Description:** Sets the tracing function with the pydev debug function and initializes needed facilities.

:param host: the user may specify another host, if the debug server is not in the same machine (default is the local
    host)

:param stdout_to_server: when this is true, the stdout is passed to the debug server

:param stderr_to_server: when this is true, the stderr is passed to the debug server
    so that they are printed in its console and not in this process console.

:param port: specifies which port to use for communicating with the server (note that the server must be started
    in the same port). @note: currently it's hard-coded at 5678 in the client

:param suspend: whether a breakpoint should be emulated as soon as this function is called.

:param trace_only_current_thread: determines if only the current thread will be traced or all current and future
    threads will also have the tracing enabled.

:param overwrite_prev_trace: deprecated

:param patch_multiprocessing: if True we'll patch the functions which create new processes so that launched
    processes are debugged.

:param stop_at_frame: if passed it'll stop at the given frame, otherwise it'll stop in the function which
    called this method.

:param wait_for_ready_to_run: if True settrace will block until the ready_to_run flag is set to True,
    otherwise, it'll set ready_to_run to True and this function won't block.

    Note that if wait_for_ready_to_run == False, there are no guarantees that the debugger is synchronized
    with what's configured in the client (IDE), the only guarantee is that when leaving this function
    the debugger will be already connected.

:param dont_trace_start_patterns: if set, then any path that starts with one fo the patterns in the collection
    will not be traced

:param dont_trace_end_patterns: if set, then any path that ends with one fo the patterns in the collection
    will not be traced

:param access_token: token to be sent from the client (i.e.: IDE) to the debugger when a connection
    is established (verified by the debugger).

:param client_access_token: token to be sent from the debugger to the client (i.e.: IDE) when
    a connection is established (verified by the client).

:param notify_stdin:
    If True sys.stdin will be patched to notify the client when a message is requested
    from the IDE. This is done so that when reading the stdin the client is notified.
    Clients may need this to know when something that is being written should be interpreted
    as an input to the process or as a command to be evaluated.
    Note that parallel-python has issues with this (because it tries to assert that sys.stdin
    is of a given type instead of just checking that it has what it needs).

:param protocol:
    When using in Eclipse the protocol should not be passed, but when used in VSCode
    or some other IDE/editor that accepts the Debug Adapter Protocol then 'dap' should
    be passed.

:param ppid:
    Override the parent process id (PPID) for the current debugging session. This PPID is
    reported to the debug client (IDE) and can be used to act like a child process of an
    existing debugged process without being a child process.

### Function: _locked_settrace(host, stdout_to_server, stderr_to_server, port, suspend, trace_only_current_thread, patch_multiprocessing, stop_at_frame, block_until_connected, wait_for_ready_to_run, dont_trace_start_patterns, dont_trace_end_patterns, access_token, client_access_token, __setup_holder__, notify_stdin, ppid)

### Function: stoptrace()

## Class: Dispatcher

## Class: DispatchReader

### Function: dispatch()

### Function: settrace_forked(setup_tracing)

**Description:** When creating a fork from a process in the debugger, we need to reset the whole debugger environment!

### Function: skip_subprocess_arg_patch()

**Description:** May be used to skip the monkey-patching that pydevd does to
skip changing arguments to embed the debugger into child processes.

i.e.:

with pydevd.skip_subprocess_arg_patch():
    subprocess.call(...)

### Function: add_dont_terminate_child_pid(pid)

**Description:** May be used to ask pydevd to skip the termination of some process
when it's asked to terminate (debug adapter protocol only).

:param int pid:
    The pid to be ignored.

i.e.:

process = subprocess.Popen(...)
pydevd.add_dont_terminate_child_pid(process.pid)

## Class: SetupHolder

### Function: apply_debugger_options(setup_options)

**Description:** :type setup_options: dict[str, bool]

### Function: patch_stdin()

### Function: _internal_patch_stdin(py_db, sys, getpass_mod)

**Description:** Note: don't use this function directly, use `patch_stdin()` instead.
(this function is only meant to be used on test-cases to avoid patching the actual globals).

### Function: log_to(log_file, log_level)

**Description:** In pydevd it's possible to log by setting the following environment variables:

PYDEVD_DEBUG=1 (sets the default log level to 3 along with other default options)
PYDEVD_DEBUG_FILE=</path/to/file.log>

Note that the file will have the pid of the process added to it (so, logging to
/path/to/file.log would actually start logging to /path/to/file.<pid>.log -- if subprocesses are
logged, each new subprocess will have the logging set to its own pid).

Usually setting the environment variable is preferred as it'd log information while
pydevd is still doing its imports and not just after this method is called, but on
cases where this is hard to do this function may be called to set the tracing after
pydevd itself is already imported.

### Function: _log_initial_info()

### Function: config(protocol, debug_mode, preimport)

### Function: main()

### Function: __init__(self, py_db)

### Function: _on_run(self)

### Function: do_kill_pydev_thread(self)

### Function: __init__(self, py_db)

### Function: _on_run(self)

### Function: join(self, timeout)

### Function: do_kill_pydev_thread(self)

### Function: __init__(self, py_db)

### Function: send_suspend_notification(self, thread_id, thread, stop_reason)

### Function: send_resume_notification(self, thread_id)

### Function: increment_suspend_time(self)

### Function: on_pause(self)

### Function: _notify_after_timeout(self, global_suspend_time)

### Function: on_thread_suspend(self, thread_id, thread, stop_reason)

### Function: on_thread_resume(self, thread_id, thread)

### Function: notify_thread_suspended(self, thread_id, thread, stop_reason)

### Function: __init__(self, py_db)

### Function: add_on_resumed_callback(self, callback)

### Function: send_resume_notification(self, thread_id)

### Function: send_suspend_notification(self, thread_id, thread, stop_reason)

### Function: notify_thread_suspended(self, thread_id, thread, stop_reason)

### Function: __init__(self)

### Function: is_authenticated(self)

### Function: login(self, access_token)

### Function: logout(self)

### Function: __init__(self, set_as_global)

### Function: collect_try_except_info(self, code_obj)

### Function: setup_auto_reload_watcher(self, enable_auto_reload, watch_dirs, poll_target_time, exclude_patterns, include_patterns)

### Function: get_arg_ppid(self)

### Function: wait_for_ready_to_run(self)

### Function: on_initialize(self)

**Description:** Note: only called when using the DAP (Debug Adapter Protocol).

### Function: on_configuration_done(self)

**Description:** Note: only called when using the DAP (Debug Adapter Protocol).

### Function: is_attached(self)

### Function: on_disconnect(self)

**Description:** Note: only called when using the DAP (Debug Adapter Protocol).

### Function: set_ignore_system_exit_codes(self, ignore_system_exit_codes)

### Function: ignore_system_exit_code(self, system_exit_exc)

### Function: block_until_configuration_done(self, cancel)

### Function: add_fake_frame(self, thread_id, frame_id, frame)

### Function: handle_breakpoint_condition(self, info, pybreakpoint, new_frame)

### Function: handle_breakpoint_expression(self, pybreakpoint, info, new_frame)

### Function: _internal_get_file_type(self, abs_real_path_and_basename)

### Function: dont_trace_external_files(self, abs_path)

**Description:** :param abs_path:
    The result from get_abs_path_real_path_and_base_from_file or
    get_abs_path_real_path_and_base_from_frame.

:return
    True :
        If files should NOT be traced.

    False:
        If files should be traced.

### Function: get_file_type(self, frame, abs_real_path_and_basename, _cache_file_type)

**Description:** :param abs_real_path_and_basename:
    The result from get_abs_path_real_path_and_base_from_file or
    get_abs_path_real_path_and_base_from_frame.

:return
    _pydevd_bundle.pydevd_dont_trace_files.PYDEV_FILE:
        If it's a file internal to the debugger which shouldn't be
        traced nor shown to the user.

    _pydevd_bundle.pydevd_dont_trace_files.LIB_FILE:
        If it's a file in a library which shouldn't be traced.

    None:
        If it's a regular user file which should be traced.

### Function: is_cache_file_type_empty(self)

### Function: get_cache_file_type(self, _cache)

### Function: get_thread_local_trace_func(self)

### Function: enable_tracing(self, thread_trace_func, apply_to_all_threads)

**Description:** Enables tracing.

If in regular mode (tracing), will set the tracing function to the tracing
function for this thread -- by default it's `PyDB.trace_dispatch`, but after
`PyDB.enable_tracing` is called with a `thread_trace_func`, the given function will
be the default for the given thread.

:param bool apply_to_all_threads:
    If True we'll set the tracing function in all threads, not only in the current thread.
    If False only the tracing for the current function should be changed.
    In general apply_to_all_threads should only be true if this is the first time
    this function is called on a multi-threaded program (either programmatically or attach
    to pid).

### Function: disable_tracing(self)

### Function: on_breakpoints_changed(self, removed)

**Description:** When breakpoints change, we have to re-evaluate all the assumptions we've made so far.

### Function: set_tracing_for_untraced_contexts(self, breakpoints_changed)

### Function: multi_threads_single_notification(self)

### Function: multi_threads_single_notification(self, notify)

### Function: threads_suspended_single_notification(self)

### Function: get_plugin_lazy_init(self)

### Function: in_project_scope(self, frame, absolute_filename)

**Description:** Note: in general this method should not be used (apply_files_filter should be used
in most cases as it also handles the project scope check).

:param frame:
    The frame we want to check.

:param absolute_filename:
    Must be the result from get_abs_path_real_path_and_base_from_frame(frame)[0] (can
    be used to speed this function a bit if it's already available to the caller, but
    in general it's not needed).

### Function: in_project_roots_filename_uncached(self, absolute_filename)

### Function: _clear_caches(self)

### Function: clear_dont_trace_start_end_patterns_caches(self)

### Function: _exclude_by_filter(self, frame, absolute_filename)

**Description:** :return: True if it should be excluded, False if it should be included and None
    if no rule matched the given file.

:note: it'll be normalized as needed inside of this method.

### Function: apply_files_filter(self, frame, original_filename, force_check_project_scope)

**Description:** Should only be called if `self.is_files_filter_enabled == True` or `force_check_project_scope == True`.

Note that it covers both the filter by specific paths includes/excludes as well
as the check which filters out libraries if not in the project scope.

:param original_filename:
    Note can either be the original filename or the absolute version of that filename.

:param force_check_project_scope:
    Check that the file is in the project scope even if the global setting
    is off.

:return bool:
    True if it should be excluded when stepping and False if it should be
    included.

### Function: exclude_exception_by_filter(self, exception_breakpoint, trace)

### Function: set_project_roots(self, project_roots)

### Function: set_exclude_filters(self, exclude_filters)

### Function: set_use_libraries_filter(self, use_libraries_filter)

### Function: get_use_libraries_filter(self)

### Function: get_require_module_for_filters(self)

### Function: has_user_threads_alive(self)

### Function: initialize_network(self, sock, terminate_on_socket_close)

### Function: connect(self, host, port)

### Function: create_wait_for_connection_thread(self)

### Function: set_server_socket_ready(self)

### Function: wait_for_server_socket_ready(self)

### Function: dap_messages_listeners(self)

### Function: add_dap_messages_listener(self, listener)

## Class: _WaitForConnectionThread

### Function: get_internal_queue_and_event(self, thread_id)

**Description:** returns internal command queue for a given thread.
if new queue is created, notify the RDB about it

### Function: post_method_as_internal_command(self, thread_id, method)

### Function: post_internal_command(self, int_cmd, thread_id)

**Description:** if thread_id is *, post to the '*' queue

### Function: enable_output_redirection(self, redirect_stdout, redirect_stderr)

### Function: check_output_redirect(self)

### Function: init_gui_support(self)

### Function: _activate_gui_if_needed(self)

### Function: _call_input_hook(self)

### Function: notify_skipped_step_in_because_of_filters(self, frame)

### Function: notify_thread_created(self, thread_id, thread, use_lock)

### Function: notify_thread_not_alive(self, thread_id, use_lock)

**Description:** if thread is not alive, cancel trace_dispatch processing

### Function: set_enable_thread_notifications(self, enable)

### Function: process_internal_commands(self, process_thread_ids)

**Description:** This function processes internal commands.

### Function: consolidate_breakpoints(self, canonical_normalized_filename, id_to_breakpoint, file_to_line_to_breakpoints)

### Function: add_break_on_exception(self, exception, condition, expression, notify_on_handled_exceptions, notify_on_unhandled_exceptions, notify_on_user_unhandled_exceptions, notify_on_first_raise_only, ignore_libraries)

### Function: set_suspend(self, thread, stop_reason, suspend_other_threads, is_pause, original_step_cmd, suspend_requested)

**Description:** :param thread:
    The thread which should be suspended.

:param stop_reason:
    Reason why the thread was suspended.

:param suspend_other_threads:
    Whether to force other threads to be suspended (i.e.: when hitting a breakpoint
    with a suspend all threads policy).

:param is_pause:
    If this is a pause to suspend all threads, any thread can be considered as the 'main'
    thread paused.

:param original_step_cmd:
    If given we may change the stop reason to this.

:param suspend_requested:
    If the execution will be suspended right away then this may be false, otherwise,
    if the thread should be stopped due to this suspend at a later time then it
    should be true.

### Function: _send_breakpoint_condition_exception(self, thread, conditional_breakpoint_exception_tuple)

**Description:** If conditional breakpoint raises an exception during evaluation
send exception details to java

### Function: send_caught_exception_stack(self, thread, arg, curr_frame_id)

**Description:** Sends details on the exception which was caught (and where we stopped) to the java side.

arg is: exception type, description, traceback object

### Function: send_caught_exception_stack_proceeded(self, thread)

**Description:** Sends that some thread was resumed and is no longer showing an exception trace.

### Function: send_process_created_message(self)

**Description:** Sends a message that a new process has been created.

### Function: send_process_about_to_be_replaced(self)

**Description:** Sends a message that a new process has been created.

### Function: set_next_statement(self, frame, event, func_name, next_line)

### Function: cancel_async_evaluation(self, thread_id, frame_id)

### Function: find_frame(self, thread_id, frame_id)

**Description:** returns a frame on the thread that has a given frame_id

### Function: do_wait_suspend(self, thread, frame, event, arg, exception_type)

**Description:** busy waits until the thread state changes to RUN
it expects thread's state as attributes of the thread.
Upon running, processes any outstanding Stepping commands.

:param exception_type:
    If pausing due to an exception, its type.

### Function: _do_wait_suspend(self, thread, frame, event, arg, trace_suspend_type, from_this_thread, frames_tracker)

### Function: do_stop_on_unhandled_exception(self, thread, frame, frames_byid, arg)

### Function: set_trace_for_frame_and_parents(self, thread_ident, frame)

### Function: _create_pydb_command_thread(self)

### Function: _create_check_output_thread(self)

### Function: start_auxiliary_daemon_threads(self)

### Function: __wait_for_threads_to_finish(self, timeout)

### Function: dispose_and_kill_all_pydevd_threads(self, wait, timeout)

**Description:** When this method is called we finish the debug session, terminate threads
and if this was registered as the global instance, unregister it -- afterwards
it should be possible to create a new instance and set as global to start
a new debug session.

:param bool wait:
    If True we'll wait for the threads to be actually finished before proceeding
    (based on the available timeout).
    Note that this must be thread-safe and if one thread is waiting the other thread should
    also wait.

### Function: prepare_to_run(self)

**Description:** Shared code to prepare debugging by installing traces and registering threads

### Function: patch_threads(self)

### Function: run(self, file, globals, locals, is_module, set_trace)

### Function: _exec(self, is_module, entry_point_fn, module_name, file, globals, locals)

**Description:** This function should have frames tracked by unhandled exceptions (the `_exec` name is important).

### Function: wait_for_commands(self, globals)

### Function: before_send(self, message_as_dict)

**Description:** Called just before a message is sent to the IDE.

:type message_as_dict: dict

### Function: after_receive(self, message_as_dict)

**Description:** Called just after a message is received from the IDE.

:type message_as_dict: dict

### Function: _threads_on_timeout()

### Function: __init__(self)

### Function: connect(self, host, port)

### Function: close(self)

### Function: __init__(self, dispatcher)

### Function: _on_run(self)

### Function: do_kill_pydev_thread(self)

### Function: process_command(self, cmd_id, seq, text)

### Function: getpass()

### Function: pydevd_breakpointhook()

### Function: can_exit()

### Function: __init__(self, py_db)

### Function: run(self)

### Function: do_kill_pydev_thread(self)

## Class: _ReturnGuiLoopControlHelper

### Function: return_control()

### Function: after_sent()

### Function: new_trace_dispatch(frame, event, arg)

### Function: accept_directory(absolute_filename, cache)

### Function: accept_file(absolute_filename, cache)

### Function: get_pydb_daemon_threads_to_wait()
