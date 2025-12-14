## AI Summary

A file named interactive.py.


## Class: DummyEvent

**Description:** Dummy event object used internally by L{ConsoleDebugger}.

## Class: CmdError

**Description:** Exception raised when a command parsing error occurs.
Used internally by L{ConsoleDebugger}.

## Class: ConsoleDebugger

**Description:** Interactive console debugger.

@see: L{Debug.interactive}

### Function: get_pid(self)

### Function: get_tid(self)

### Function: get_process(self)

### Function: get_thread(self)

### Function: __init__(self)

**Description:** Interactive console debugger.

@see: L{Debug.interactive}

### Function: start_using_debugger(self, debug)

### Function: stop_using_debugger(self)

### Function: destroy_debugger(self, autodetach)

### Function: lastEvent(self)

### Function: set_fake_last_event(self, process)

### Function: join_tokens(self, token_list)

### Function: split_tokens(self, arg, min_count, max_count)

### Function: input_thread(self, token)

### Function: input_thread_list(self, token_list)

### Function: input_process(self, token)

### Function: input_process_list(self, token_list)

### Function: input_command_line(self, command_line)

### Function: input_hexadecimal_integer(self, token)

### Function: input_integer(self, token)

### Function: input_address(self, token, pid, tid)

### Function: input_address_range(self, token_list, pid, tid)

### Function: is_register(self, token)

### Function: input_register(self, token, tid)

### Function: input_full_address_range(self, token_list)

### Function: input_breakpoint(self, token_list)

### Function: input_display(self, token_list, default_size)

### Function: print_module_load(self, event)

### Function: print_module_unload(self, event)

### Function: print_process_start(self, event)

### Function: print_thread_start(self, event)

### Function: print_process_end(self, event)

### Function: print_thread_end(self, event)

### Function: print_debug_string(self, event)

### Function: print_event(self, event)

### Function: print_exception(self, event)

### Function: print_event_location(self, event)

### Function: print_breakpoint_location(self, event)

### Function: print_current_location(self, process, thread, pc)

### Function: print_memory_display(self, arg, method)

### Function: get_process_id_from_prefix(self)

### Function: get_thread_id_from_prefix(self)

### Function: get_process_from_prefix(self)

### Function: get_thread_from_prefix(self)

### Function: get_process_and_thread_ids_from_prefix(self)

### Function: get_process_and_thread_from_prefix(self)

### Function: get_process(self, pid)

### Function: get_thread(self, tid)

### Function: read_memory(self, address, size, pid)

### Function: write_memory(self, address, data, pid)

### Function: change_register(self, register, value, tid)

### Function: find_in_memory(self, query, process)

### Function: kill_process(self, pid)

### Function: kill_thread(self, tid)

### Function: prompt_user(self)

### Function: ask_user(self, msg, prompt)

### Function: autocomplete(self, cmd)

### Function: get_help(self, commands)

### Function: split_prefix(self, line)

### Function: prompt(self)

### Function: get_names(self)

### Function: parseline(self, line)

### Function: preloop(self)

### Function: get_lastcmd(self)

### Function: set_lastcmd(self, lastcmd)

### Function: postcmd(self, stop, line)

### Function: do_help(self, arg)

**Description:** ? - show the list of available commands
? * - show help for all commands
? <command> [command...] - show help for the given command(s)
help - show the list of available commands
help * - show help for all commands
help <command> [command...] - show help for the given command(s)

### Function: do_shell(self, arg)

**Description:** ! - spawn a system shell
shell - spawn a system shell
! <command> [arguments...] - execute a single shell command
shell <command> [arguments...] - execute a single shell command

## Class: _PythonExit

### Function: _spawn_python_shell(self, arg)

### Function: do_python(self, arg)

**Description:** # - spawn a python interpreter
python - spawn a python interpreter
# <statement> - execute a single python statement
python <statement> - execute a single python statement

### Function: do_quit(self, arg)

**Description:** quit - close the debugging session
q - close the debugging session

### Function: do_attach(self, arg)

**Description:** attach <target> [target...] - attach to the given process(es)

### Function: do_detach(self, arg)

**Description:** [~process] detach - detach from the current process
detach - detach from the current process
detach <target> [target...] - detach from the given process(es)

### Function: do_windowed(self, arg)

**Description:** windowed <target> [arguments...] - run a windowed program for debugging

### Function: do_console(self, arg)

**Description:** console <target> [arguments...] - run a console program for debugging

### Function: do_continue(self, arg)

**Description:** continue - continue execution
g - continue execution
go - continue execution

### Function: do_gh(self, arg)

**Description:** gh - go with exception handled

### Function: do_gn(self, arg)

**Description:** gn - go with exception not handled

### Function: do_refresh(self, arg)

**Description:** refresh - refresh the list of running processes and threads
[~process] refresh - refresh the list of running threads

### Function: do_processlist(self, arg)

**Description:** pl - show the processes being debugged
processlist - show the processes being debugged

### Function: do_threadlist(self, arg)

**Description:** tl - show the threads being debugged
threadlist - show the threads being debugged

### Function: do_kill(self, arg)

**Description:** [~process] kill - kill a process
[~thread] kill - kill a thread
kill - kill the current process
kill * - kill all debugged processes
kill <processes and/or threads...> - kill the given processes and threads

### Function: do_modload(self, arg)

**Description:** [~process] modload <filename.dll> - load a DLL module

### Function: do_stack(self, arg)

**Description:** [~thread] k - show the stack trace
[~thread] stack - show the stack trace

### Function: do_break(self, arg)

**Description:** break - force a debug break in all debugees
break <process> [process...] - force a debug break

### Function: do_step(self, arg)

**Description:** p - step on the current assembly instruction
next - step on the current assembly instruction
step - step on the current assembly instruction

### Function: do_trace(self, arg)

**Description:** t - trace at the current assembly instruction
trace - trace at the current assembly instruction

### Function: do_bp(self, arg)

**Description:** [~process] bp <address> - set a code breakpoint

### Function: do_ba(self, arg)

**Description:** [~thread] ba <a|w|e> <1|2|4|8> <address> - set hardware breakpoint

### Function: do_bm(self, arg)

**Description:** [~process] bm <address-address> - set memory breakpoint

### Function: do_bl(self, arg)

**Description:** bl - list the breakpoints for the current process
bl * - list the breakpoints for all processes
[~process] bl - list the breakpoints for the given process
bl <process> [process...] - list the breakpoints for each given process

### Function: do_bo(self, arg)

**Description:** [~process] bo <address> - make a code breakpoint one-shot
[~thread] bo <address> - make a hardware breakpoint one-shot
[~process] bo <address-address> - make a memory breakpoint one-shot
[~process] bo <address> <size> - make a memory breakpoint one-shot

### Function: do_be(self, arg)

**Description:** [~process] be <address> - enable a code breakpoint
[~thread] be <address> - enable a hardware breakpoint
[~process] be <address-address> - enable a memory breakpoint
[~process] be <address> <size> - enable a memory breakpoint

### Function: do_bd(self, arg)

**Description:** [~process] bd <address> - disable a code breakpoint
[~thread] bd <address> - disable a hardware breakpoint
[~process] bd <address-address> - disable a memory breakpoint
[~process] bd <address> <size> - disable a memory breakpoint

### Function: do_bc(self, arg)

**Description:** [~process] bc <address> - clear a code breakpoint
[~thread] bc <address> - clear a hardware breakpoint
[~process] bc <address-address> - clear a memory breakpoint
[~process] bc <address> <size> - clear a memory breakpoint

### Function: do_disassemble(self, arg)

**Description:** [~thread] u [register] - show code disassembly
[~process] u [address] - show code disassembly
[~thread] disassemble [register] - show code disassembly
[~process] disassemble [address] - show code disassembly

### Function: do_search(self, arg)

**Description:** [~process] s [address-address] <search string>
[~process] search [address-address] <search string>

### Function: do_searchhex(self, arg)

**Description:** [~process] sh [address-address] <hexadecimal pattern>
[~process] searchhex [address-address] <hexadecimal pattern>

### Function: do_d(self, arg)

**Description:** [~thread] d <register> - show memory contents
[~thread] d <register-register> - show memory contents
[~thread] d <register> <size> - show memory contents
[~process] d <address> - show memory contents
[~process] d <address-address> - show memory contents
[~process] d <address> <size> - show memory contents

### Function: do_db(self, arg)

**Description:** [~thread] db <register> - show memory contents as bytes
[~thread] db <register-register> - show memory contents as bytes
[~thread] db <register> <size> - show memory contents as bytes
[~process] db <address> - show memory contents as bytes
[~process] db <address-address> - show memory contents as bytes
[~process] db <address> <size> - show memory contents as bytes

### Function: do_dw(self, arg)

**Description:** [~thread] dw <register> - show memory contents as words
[~thread] dw <register-register> - show memory contents as words
[~thread] dw <register> <size> - show memory contents as words
[~process] dw <address> - show memory contents as words
[~process] dw <address-address> - show memory contents as words
[~process] dw <address> <size> - show memory contents as words

### Function: do_dd(self, arg)

**Description:** [~thread] dd <register> - show memory contents as dwords
[~thread] dd <register-register> - show memory contents as dwords
[~thread] dd <register> <size> - show memory contents as dwords
[~process] dd <address> - show memory contents as dwords
[~process] dd <address-address> - show memory contents as dwords
[~process] dd <address> <size> - show memory contents as dwords

### Function: do_dq(self, arg)

**Description:** [~thread] dq <register> - show memory contents as qwords
[~thread] dq <register-register> - show memory contents as qwords
[~thread] dq <register> <size> - show memory contents as qwords
[~process] dq <address> - show memory contents as qwords
[~process] dq <address-address> - show memory contents as qwords
[~process] dq <address> <size> - show memory contents as qwords

### Function: do_ds(self, arg)

**Description:** [~thread] ds <register> - show memory contents as ANSI string
[~process] ds <address> - show memory contents as ANSI string

### Function: do_du(self, arg)

**Description:** [~thread] du <register> - show memory contents as Unicode string
[~process] du <address> - show memory contents as Unicode string

### Function: do_register(self, arg)

**Description:** [~thread] r - print(the value of all registers
[~thread] r <register> - print(the value of a register
[~thread] r <register>=<value> - change the value of a register
[~thread] register - print(the value of all registers
[~thread] register <register> - print(the value of a register
[~thread] register <register>=<value> - change the value of a register

### Function: do_eb(self, arg)

**Description:** [~process] eb <address> <data> - write the data to the specified address

### Function: do_find(self, arg)

**Description:** [~process] f <string> - find the string in the process memory
[~process] find <string> - find the string in the process memory

### Function: do_memory(self, arg)

**Description:** [~process] m - show the process memory map
[~process] memory - show the process memory map

### Function: event(self, event)

### Function: exception(self, event)

### Function: breakpoint(self, event)

### Function: wow64_breakpoint(self, event)

### Function: single_step(self, event)

### Function: ms_vc_exception(self, event)

### Function: create_process(self, event)

### Function: exit_process(self, event)

### Function: create_thread(self, event)

### Function: exit_thread(self, event)

### Function: load_dll(self, event)

### Function: unload_dll(self, event)

### Function: output_string(self, event)

### Function: load_history(self)

### Function: save_history(self)

### Function: loop(self)

### Function: __repr__(self)

### Function: __call__(self)
