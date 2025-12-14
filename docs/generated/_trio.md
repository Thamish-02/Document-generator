## AI Summary

A file named _trio.py.


## Class: CancelScope

## Class: TaskGroup

## Class: BlockingPortal

## Class: ReceiveStreamWrapper

## Class: SendStreamWrapper

## Class: Process

## Class: _ProcessPoolShutdownInstrument

## Class: _TrioSocketMixin

## Class: SocketStream

## Class: UNIXSocketStream

## Class: TCPSocketListener

## Class: UNIXSocketListener

## Class: UDPSocket

## Class: ConnectedUDPSocket

## Class: UNIXDatagramSocket

## Class: ConnectedUNIXDatagramSocket

## Class: Event

## Class: Lock

## Class: Semaphore

## Class: CapacityLimiter

## Class: _SignalReceiver

## Class: TestRunner

## Class: TrioTaskInfo

## Class: TrioBackend

### Function: __new__(cls, original)

### Function: __init__(self, original)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: cancel(self, reason)

### Function: deadline(self)

### Function: deadline(self, value)

### Function: cancel_called(self)

### Function: cancelled_caught(self)

### Function: shield(self)

### Function: shield(self, value)

### Function: __init__(self)

### Function: start_soon(self, func)

### Function: __new__(cls)

### Function: __init__(self)

### Function: _spawn_task_from_thread(self, func, args, kwargs, name, future)

### Function: terminate(self)

### Function: kill(self)

### Function: send_signal(self, signal)

### Function: pid(self)

### Function: returncode(self)

### Function: stdin(self)

### Function: stdout(self)

### Function: stderr(self)

### Function: after_run(self)

### Function: __init__(self, trio_socket)

### Function: _check_closed(self)

### Function: _raw_socket(self)

### Function: _convert_socket_error(self, exc)

### Function: __init__(self, trio_socket)

### Function: __init__(self, raw_socket)

### Function: __init__(self, raw_socket)

### Function: __init__(self, trio_socket)

### Function: __init__(self, trio_socket)

### Function: __init__(self, trio_socket)

### Function: __init__(self, trio_socket)

### Function: __new__(cls)

### Function: __init__(self)

### Function: is_set(self)

### Function: statistics(self)

### Function: set(self)

### Function: __new__(cls)

### Function: __init__(self)

### Function: _convert_runtime_error_msg(exc)

### Function: acquire_nowait(self)

### Function: locked(self)

### Function: release(self)

### Function: statistics(self)

### Function: __new__(cls, initial_value)

### Function: __init__(self, initial_value)

### Function: acquire_nowait(self)

### Function: max_value(self)

### Function: value(self)

### Function: release(self)

### Function: statistics(self)

### Function: __new__(cls, total_tokens)

### Function: __init__(self, total_tokens)

### Function: total_tokens(self)

### Function: total_tokens(self, value)

### Function: borrowed_tokens(self)

### Function: available_tokens(self)

### Function: acquire_nowait(self)

### Function: acquire_on_behalf_of_nowait(self, borrower)

### Function: release(self)

### Function: release_on_behalf_of(self, borrower)

### Function: statistics(self)

### Function: __init__(self, signals)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: __aiter__(self)

### Function: __init__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: _main_task_finished(self, outcome)

### Function: _call_in_runner_task(self, func)

### Function: run_asyncgen_fixture(self, fixture_func, kwargs)

### Function: run_fixture(self, fixture_func, kwargs)

### Function: run_test(self, test_func, kwargs)

### Function: __init__(self, task)

### Function: has_pending_cancellation(self)

### Function: run(cls, func, args, kwargs, options)

### Function: current_token(cls)

### Function: current_time(cls)

### Function: cancelled_exception_class(cls)

### Function: create_cancel_scope(cls)

### Function: current_effective_deadline(cls)

### Function: create_task_group(cls)

### Function: create_event(cls)

### Function: create_lock(cls)

### Function: create_semaphore(cls, initial_value)

### Function: create_capacity_limiter(cls, total_tokens)

### Function: check_cancelled(cls)

### Function: run_async_from_thread(cls, func, args, token)

### Function: run_sync_from_thread(cls, func, args, token)

### Function: create_blocking_portal(cls)

### Function: setup_process_pool_exit_at_shutdown(cls, workers)

### Function: create_tcp_listener(cls, sock)

### Function: create_unix_listener(cls, sock)

### Function: notify_closing(cls, obj)

### Function: current_default_thread_limiter(cls)

### Function: open_signal_receiver(cls)

### Function: get_current_task(cls)

### Function: get_running_tasks(cls)

### Function: create_test_runner(cls, options)

### Function: wrapper()

### Function: convert_item(item)
