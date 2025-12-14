## AI Summary

A file named _asyncio.py.


### Function: find_root_task()

### Function: get_callable_name(func)

### Function: _task_started(task)

**Description:** Return ``True`` if the task has been started and has not finished.

### Function: is_anyio_cancellation(exc)

## Class: CancelScope

## Class: TaskState

**Description:** Encapsulates auxiliary task information that cannot be added to the Task instance
itself because there are no guarantees about its implementation.

## Class: _AsyncioTaskStatus

## Class: TaskGroup

## Class: WorkerThread

## Class: BlockingPortal

## Class: StreamReaderWrapper

## Class: StreamWriterWrapper

## Class: Process

### Function: _forcibly_shutdown_process_pool_on_exit(workers, _task)

**Description:** Forcibly shuts down worker processes belonging to this event loop.

## Class: StreamProtocol

## Class: DatagramProtocol

## Class: SocketStream

## Class: _RawSocketMixin

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

## Class: AsyncIOTaskInfo

## Class: TestRunner

## Class: AsyncIOBackend

## Class: _State

## Class: Runner

### Function: _cancel_all_tasks(loop)

### Function: __new__(cls)

### Function: __init__(self, deadline, shield)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: _effectively_cancelled(self)

### Function: _parent_cancellation_is_visible_to_us(self)

### Function: _timeout(self)

### Function: _deliver_cancellation(self, origin)

**Description:** Deliver cancellation to directly contained tasks and nested cancel scopes.

Schedule another run at the end if we still have tasks eligible for
cancellation.

:param origin: the cancel scope that originated the cancellation
:return: ``True`` if the delivery needs to be retried on the next cycle

### Function: _restart_cancellation_in_parent(self)

**Description:** Restart the cancellation effort in the closest directly cancelled parent scope.

### Function: cancel(self, reason)

### Function: deadline(self)

### Function: deadline(self, value)

### Function: cancel_called(self)

### Function: cancelled_caught(self)

### Function: shield(self)

### Function: shield(self, value)

### Function: __init__(self, parent_id, cancel_scope)

### Function: __init__(self, future, parent_id)

### Function: started(self, value)

### Function: __init__(self)

### Function: _spawn(self, func, args, name, task_status_future)

### Function: start_soon(self, func)

### Function: __init__(self, root_task, workers, idle_workers)

### Function: _report_result(self, future, result, exc)

### Function: run(self)

### Function: stop(self, f)

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

### Function: connection_made(self, transport)

### Function: connection_lost(self, exc)

### Function: data_received(self, data)

### Function: eof_received(self)

### Function: pause_writing(self)

### Function: resume_writing(self)

### Function: connection_made(self, transport)

### Function: connection_lost(self, exc)

### Function: datagram_received(self, data, addr)

### Function: error_received(self, exc)

### Function: pause_writing(self)

### Function: resume_writing(self)

### Function: __init__(self, transport, protocol)

### Function: _raw_socket(self)

### Function: __init__(self, raw_socket)

### Function: _raw_socket(self)

### Function: _wait_until_readable(self, loop)

### Function: _wait_until_writable(self, loop)

### Function: __init__(self, raw_socket)

### Function: _raw_socket(self)

### Function: __init__(self, raw_socket)

### Function: _raw_socket(self)

### Function: __init__(self, transport, protocol)

### Function: _raw_socket(self)

### Function: __init__(self, transport, protocol)

### Function: _raw_socket(self)

### Function: __new__(cls)

### Function: __init__(self)

### Function: set(self)

### Function: is_set(self)

### Function: statistics(self)

### Function: __new__(cls)

### Function: __init__(self)

### Function: acquire_nowait(self)

### Function: locked(self)

### Function: release(self)

### Function: statistics(self)

### Function: __new__(cls, initial_value)

### Function: __init__(self, initial_value)

### Function: acquire_nowait(self)

### Function: release(self)

### Function: value(self)

### Function: max_value(self)

### Function: statistics(self)

### Function: __new__(cls, total_tokens)

### Function: __init__(self, total_tokens)

### Function: total_tokens(self)

### Function: total_tokens(self, value)

### Function: borrowed_tokens(self)

### Function: available_tokens(self)

### Function: _notify_next_waiter(self)

**Description:** Notify the next task in line if this limiter has free capacity now.

### Function: acquire_nowait(self)

### Function: acquire_on_behalf_of_nowait(self, borrower)

### Function: release(self)

### Function: release_on_behalf_of(self, borrower)

### Function: statistics(self)

### Function: __init__(self, signals)

### Function: _deliver(self, signum)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: __aiter__(self)

### Function: __init__(self, task)

### Function: has_pending_cancellation(self)

### Function: __init__(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: get_loop(self)

### Function: _exception_handler(self, loop, context)

### Function: _raise_async_exceptions(self)

### Function: run_asyncgen_fixture(self, fixture_func, kwargs)

### Function: run_fixture(self, fixture_func, kwargs)

### Function: run_test(self, test_func, kwargs)

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

### Function: __init__(self)

### Function: __enter__(self)

### Function: __exit__(self, exc_type, exc_val, exc_tb)

### Function: close(self)

**Description:** Shutdown and close event loop.

### Function: get_loop(self)

**Description:** Return embedded event loop.

### Function: run(self, coro)

**Description:** Run a coroutine inside the embedded event loop.

### Function: _lazy_init(self)

### Function: _on_sigint(self, signum, frame, main_task)

### Function: _do_shutdown(future)

### Function: task_done(_task)

### Function: callback(f)

### Function: callback(f)

### Function: wrapper()

### Function: cb()

### Function: cb()
