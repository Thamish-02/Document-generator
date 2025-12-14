## AI Summary

A file named pydev_monkey.py.


### Function: skip_subprocess_arg_patch()

### Function: _get_apply_arg_patching()

### Function: _get_setup_updated_with_protocol_and_ppid(setup, is_exec)

## Class: _LastFutureImportFinder

### Function: _get_offset_from_line_col(code, line, col)

### Function: _separate_future_imports(code)

**Description:** :param code:
    The code from where we want to get the __future__ imports (note that it's possible that
    there's no such entry).

:return tuple(str, str):
    The return is a tuple(future_import, code).

    If the future import is not available a return such as ('', code) is given, otherwise, the
    future import will end with a ';' (so that it can be put right before the pydevd attach
    code).

### Function: _get_python_c_args(host, port, code, args, setup)

### Function: _get_host_port()

### Function: _is_managed_arg(arg)

### Function: _on_forked_process(setup_tracing)

### Function: _on_set_trace_for_new_thread(global_debugger)

### Function: _get_str_type_compatible(s, args)

**Description:** This method converts `args` to byte/unicode based on the `s' type.

### Function: is_python(path)

## Class: InvalidTypeInArgsException

### Function: remove_quotes_from_args(args)

### Function: quote_arg_win32(arg)

### Function: quote_args(args)

### Function: patch_args(args, is_exec)

**Description:** :param list args:
    Arguments to patch.

:param bool is_exec:
    If it's an exec, the current process will be replaced (this means we have
    to keep the same ppid).

### Function: str_to_args_windows(args)

### Function: patch_arg_str_win(arg_str)

### Function: monkey_patch_module(module, funcname, create_func)

### Function: monkey_patch_os(funcname, create_func)

### Function: warn_multiproc()

### Function: create_warn_multiproc(original_name)

### Function: create_execl(original_name)

### Function: create_execv(original_name)

### Function: create_execve(original_name)

**Description:** os.execve(path, args, env)
os.execvpe(file, args, env)

### Function: create_spawnl(original_name)

### Function: create_spawnv(original_name)

### Function: create_spawnve(original_name)

**Description:** os.spawnve(mode, path, args, env)
os.spawnvpe(mode, file, args, env)

### Function: create_posix_spawn(original_name)

**Description:** os.posix_spawn(executable, args, env, **kwargs)

### Function: create_fork_exec(original_name)

**Description:** _posixsubprocess.fork_exec(args, executable_list, close_fds, ... (13 more))

### Function: create_warn_fork_exec(original_name)

**Description:** _posixsubprocess.fork_exec(args, executable_list, close_fds, ... (13 more))

### Function: create_subprocess_fork_exec(original_name)

**Description:** subprocess._fork_exec(args, executable_list, close_fds, ... (13 more))

### Function: create_subprocess_warn_fork_exec(original_name)

**Description:** subprocess._fork_exec(args, executable_list, close_fds, ... (13 more))

### Function: create_CreateProcess(original_name)

**Description:** CreateProcess(*args, **kwargs)

### Function: create_CreateProcessWarnMultiproc(original_name)

**Description:** CreateProcess(*args, **kwargs)

### Function: create_fork(original_name)

### Function: send_process_created_message()

### Function: send_process_about_to_be_replaced()

### Function: patch_new_process_functions()

### Function: patch_new_process_functions_with_warning()

## Class: _NewThreadStartupWithTrace

## Class: _NewThreadStartupWithoutTrace

### Function: _get_threading_modules_to_patch()

### Function: patch_thread_module(thread_module)

### Function: patch_thread_modules()

### Function: undo_patch_thread_modules()

### Function: disable_trace_thread_modules()

**Description:** Can be used to temporarily stop tracing threads created with thread.start_new_thread.

### Function: enable_trace_thread_modules()

**Description:** Can be used to start tracing threads created with thread.start_new_thread again.

### Function: get_original_start_new_thread(threading_module)

### Function: __init__(self)

### Function: visit_ImportFrom(self, node)

### Function: new_warn_multiproc()

### Function: new_execl(path)

**Description:** os.execl(path, arg0, arg1, ...)
os.execle(path, arg0, arg1, ..., env)
os.execlp(file, arg0, arg1, ...)
os.execlpe(file, arg0, arg1, ..., env)

### Function: new_execv(path, args)

**Description:** os.execv(path, args)
os.execvp(file, args)

### Function: new_execve(path, args, env)

### Function: new_spawnl(mode, path)

**Description:** os.spawnl(mode, path, arg0, arg1, ...)
os.spawnlp(mode, file, arg0, arg1, ...)

### Function: new_spawnv(mode, path, args)

**Description:** os.spawnv(mode, path, args)
os.spawnvp(mode, file, args)

### Function: new_spawnve(mode, path, args, env)

### Function: new_posix_spawn(executable, args, env)

### Function: new_fork_exec(args)

### Function: new_warn_fork_exec()

### Function: new_fork_exec(args)

### Function: new_warn_fork_exec()

### Function: new_CreateProcess(app_name, cmd_line)

### Function: new_CreateProcess()

### Function: new_fork()

### Function: __init__(self, original_func, args, kwargs)

### Function: __call__(self)

### Function: __init__(self, original_func, args, kwargs)

### Function: __call__(self)

## Class: ClassWithPydevStartNewThread

## Class: ClassWithPydevStartJoinableThread

### Function: pydev_start_new_thread(self, function, args, kwargs)

**Description:** We need to replace the original thread_module.start_new_thread with this function so that threads started
through it and not through the threading module are properly traced.

### Function: pydev_start_joinable_thread(self, function)

**Description:** We need to replace the original thread_module._start_joinable_thread with this function so that threads started
through it and not through the threading module are properly traced.
