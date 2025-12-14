## AI Summary

A file named pydevd_schema.py.


## Class: ProtocolMessage

**Description:** Base class of requests, responses, and events.

Note: automatically generated code. Do not edit manually.

## Class: Request

**Description:** A client or debug adapter initiated request.

Note: automatically generated code. Do not edit manually.

## Class: Event

**Description:** A debug adapter initiated event.

Note: automatically generated code. Do not edit manually.

## Class: Response

**Description:** Response for a request.

Note: automatically generated code. Do not edit manually.

## Class: ErrorResponse

**Description:** On error (whenever `success` is false), the body can provide more details.

Note: automatically generated code. Do not edit manually.

## Class: CancelRequest

**Description:** The `cancel` request is used by the client in two situations:

- to indicate that it is no longer interested in the result produced by a specific request issued
earlier

- to cancel a progress sequence.

Clients should only call this request if the corresponding capability `supportsCancelRequest` is
true.

This request has a hint characteristic: a debug adapter can only be expected to make a 'best effort'
in honoring this request but there are no guarantees.

The `cancel` request may return an error if it could not cancel an operation but a client should
refrain from presenting this error to end users.

The request that got cancelled still needs to send a response back. This can either be a normal
result (`success` attribute true) or an error response (`success` attribute false and the `message`
set to `cancelled`).

Returning partial results from a cancelled request is possible but please note that a client has no
generic way for detecting that a response is partial or not.

The progress that got cancelled still needs to send a `progressEnd` event back.

A client should not assume that progress just got cancelled after sending the `cancel` request.

Note: automatically generated code. Do not edit manually.

## Class: CancelArguments

**Description:** Arguments for `cancel` request.

Note: automatically generated code. Do not edit manually.

## Class: CancelResponse

**Description:** Response to `cancel` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: InitializedEvent

**Description:** This event indicates that the debug adapter is ready to accept configuration requests (e.g.
`setBreakpoints`, `setExceptionBreakpoints`).

A debug adapter is expected to send this event when it is ready to accept configuration requests
(but not before the `initialize` request has finished).

The sequence of events/requests is as follows:

- adapters sends `initialized` event (after the `initialize` request has returned)

- client sends zero or more `setBreakpoints` requests

- client sends one `setFunctionBreakpoints` request (if corresponding capability
`supportsFunctionBreakpoints` is true)

- client sends a `setExceptionBreakpoints` request if one or more `exceptionBreakpointFilters` have
been defined (or if `supportsConfigurationDoneRequest` is not true)

- client sends other future configuration requests

- client sends one `configurationDone` request to indicate the end of the configuration.

Note: automatically generated code. Do not edit manually.

## Class: StoppedEvent

**Description:** The event indicates that the execution of the debuggee has stopped due to some condition.

This can be caused by a breakpoint previously set, a stepping request has completed, by executing a
debugger statement etc.

Note: automatically generated code. Do not edit manually.

## Class: ContinuedEvent

**Description:** The event indicates that the execution of the debuggee has continued.

Please note: a debug adapter is not expected to send this event in response to a request that
implies that execution continues, e.g. `launch` or `continue`.

It is only necessary to send a `continued` event if there was no previous request that implied this.

Note: automatically generated code. Do not edit manually.

## Class: ExitedEvent

**Description:** The event indicates that the debuggee has exited and returns its exit code.

Note: automatically generated code. Do not edit manually.

## Class: TerminatedEvent

**Description:** The event indicates that debugging of the debuggee has terminated. This does **not** mean that the
debuggee itself has exited.

Note: automatically generated code. Do not edit manually.

## Class: ThreadEvent

**Description:** The event indicates that a thread has started or exited.

Note: automatically generated code. Do not edit manually.

## Class: OutputEvent

**Description:** The event indicates that the target has produced some output.

Note: automatically generated code. Do not edit manually.

## Class: BreakpointEvent

**Description:** The event indicates that some information about a breakpoint has changed.

Note: automatically generated code. Do not edit manually.

## Class: ModuleEvent

**Description:** The event indicates that some information about a module has changed.

Note: automatically generated code. Do not edit manually.

## Class: LoadedSourceEvent

**Description:** The event indicates that some source has been added, changed, or removed from the set of all loaded
sources.

Note: automatically generated code. Do not edit manually.

## Class: ProcessEvent

**Description:** The event indicates that the debugger has begun debugging a new process. Either one that it has
launched, or one that it has attached to.

Note: automatically generated code. Do not edit manually.

## Class: CapabilitiesEvent

**Description:** The event indicates that one or more capabilities have changed.

Since the capabilities are dependent on the client and its UI, it might not be possible to change
that at random times (or too late).

Consequently this event has a hint characteristic: a client can only be expected to make a 'best
effort' in honoring individual capabilities but there are no guarantees.

Only changed capabilities need to be included, all other capabilities keep their values.

Note: automatically generated code. Do not edit manually.

## Class: ProgressStartEvent

**Description:** The event signals that a long running operation is about to start and provides additional
information for the client to set up a corresponding progress and cancellation UI.

The client is free to delay the showing of the UI in order to reduce flicker.

This event should only be sent if the corresponding capability `supportsProgressReporting` is true.

Note: automatically generated code. Do not edit manually.

## Class: ProgressUpdateEvent

**Description:** The event signals that the progress reporting needs to be updated with a new message and/or
percentage.

The client does not have to update the UI immediately, but the clients needs to keep track of the
message and/or percentage values.

This event should only be sent if the corresponding capability `supportsProgressReporting` is true.

Note: automatically generated code. Do not edit manually.

## Class: ProgressEndEvent

**Description:** The event signals the end of the progress reporting with a final message.

This event should only be sent if the corresponding capability `supportsProgressReporting` is true.

Note: automatically generated code. Do not edit manually.

## Class: InvalidatedEvent

**Description:** This event signals that some state in the debug adapter has changed and requires that the client
needs to re-render the data snapshot previously requested.

Debug adapters do not have to emit this event for runtime changes like stopped or thread events
because in that case the client refetches the new state anyway. But the event can be used for
example to refresh the UI after rendering formatting has changed in the debug adapter.

This event should only be sent if the corresponding capability `supportsInvalidatedEvent` is true.

Note: automatically generated code. Do not edit manually.

## Class: MemoryEvent

**Description:** This event indicates that some memory range has been updated. It should only be sent if the
corresponding capability `supportsMemoryEvent` is true.

Clients typically react to the event by re-issuing a `readMemory` request if they show the memory
identified by the `memoryReference` and if the updated memory range overlaps the displayed range.
Clients should not make assumptions how individual memory references relate to each other, so they
should not assume that they are part of a single continuous address range and might overlap.

Debug adapters can use this event to indicate that the contents of a memory range has changed due to
some other request like `setVariable` or `setExpression`. Debug adapters are not expected to emit
this event for each and every memory change of a running program, because that information is
typically not available from debuggers and it would flood clients with too many events.

Note: automatically generated code. Do not edit manually.

## Class: RunInTerminalRequest

**Description:** This request is sent from the debug adapter to the client to run a command in a terminal.

This is typically used to launch the debuggee in a terminal provided by the client.

This request should only be called if the corresponding client capability
`supportsRunInTerminalRequest` is true.

Client implementations of `runInTerminal` are free to run the command however they choose including
issuing the command to a command line interpreter (aka 'shell'). Argument strings passed to the
`runInTerminal` request must arrive verbatim in the command to be run. As a consequence, clients
which use a shell are responsible for escaping any special shell characters in the argument strings
to prevent them from being interpreted (and modified) by the shell.

Some users may wish to take advantage of shell processing in the argument strings. For clients which
implement `runInTerminal` using an intermediary shell, the `argsCanBeInterpretedByShell` property
can be set to true. In this case the client is requested not to escape any special shell characters
in the argument strings.

Note: automatically generated code. Do not edit manually.

## Class: RunInTerminalRequestArguments

**Description:** Arguments for `runInTerminal` request.

Note: automatically generated code. Do not edit manually.

## Class: RunInTerminalResponse

**Description:** Response to `runInTerminal` request.

Note: automatically generated code. Do not edit manually.

## Class: StartDebuggingRequest

**Description:** This request is sent from the debug adapter to the client to start a new debug session of the same
type as the caller.

This request should only be sent if the corresponding client capability
`supportsStartDebuggingRequest` is true.

A client implementation of `startDebugging` should start a new debug session (of the same type as
the caller) in the same way that the caller's session was started. If the client supports
hierarchical debug sessions, the newly created session can be treated as a child of the caller
session.

Note: automatically generated code. Do not edit manually.

## Class: StartDebuggingRequestArguments

**Description:** Arguments for `startDebugging` request.

Note: automatically generated code. Do not edit manually.

## Class: StartDebuggingResponse

**Description:** Response to `startDebugging` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: InitializeRequest

**Description:** The `initialize` request is sent as the first request from the client to the debug adapter in order
to configure it with client capabilities and to retrieve capabilities from the debug adapter.

Until the debug adapter has responded with an `initialize` response, the client must not send any
additional requests or events to the debug adapter.

In addition the debug adapter is not allowed to send any requests or events to the client until it
has responded with an `initialize` response.

The `initialize` request may only be sent once.

Note: automatically generated code. Do not edit manually.

## Class: InitializeRequestArguments

**Description:** Arguments for `initialize` request.

Note: automatically generated code. Do not edit manually.

## Class: InitializeResponse

**Description:** Response to `initialize` request.

Note: automatically generated code. Do not edit manually.

## Class: ConfigurationDoneRequest

**Description:** This request indicates that the client has finished initialization of the debug adapter.

So it is the last request in the sequence of configuration requests (which was started by the
`initialized` event).

Clients should only call this request if the corresponding capability
`supportsConfigurationDoneRequest` is true.

Note: automatically generated code. Do not edit manually.

## Class: ConfigurationDoneArguments

**Description:** Arguments for `configurationDone` request.

Note: automatically generated code. Do not edit manually.

## Class: ConfigurationDoneResponse

**Description:** Response to `configurationDone` request. This is just an acknowledgement, so no body field is
required.

Note: automatically generated code. Do not edit manually.

## Class: LaunchRequest

**Description:** This launch request is sent from the client to the debug adapter to start the debuggee with or
without debugging (if `noDebug` is true).

Since launching is debugger/runtime specific, the arguments for this request are not part of this
specification.

Note: automatically generated code. Do not edit manually.

## Class: LaunchRequestArguments

**Description:** Arguments for `launch` request. Additional attributes are implementation specific.

Note: automatically generated code. Do not edit manually.

## Class: LaunchResponse

**Description:** Response to `launch` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: AttachRequest

**Description:** The `attach` request is sent from the client to the debug adapter to attach to a debuggee that is
already running.

Since attaching is debugger/runtime specific, the arguments for this request are not part of this
specification.

Note: automatically generated code. Do not edit manually.

## Class: AttachRequestArguments

**Description:** Arguments for `attach` request. Additional attributes are implementation specific.

Note: automatically generated code. Do not edit manually.

## Class: AttachResponse

**Description:** Response to `attach` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: RestartRequest

**Description:** Restarts a debug session. Clients should only call this request if the corresponding capability
`supportsRestartRequest` is true.

If the capability is missing or has the value false, a typical client emulates `restart` by
terminating the debug adapter first and then launching it anew.

Note: automatically generated code. Do not edit manually.

## Class: RestartArguments

**Description:** Arguments for `restart` request.

Note: automatically generated code. Do not edit manually.

## Class: RestartResponse

**Description:** Response to `restart` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: DisconnectRequest

**Description:** The `disconnect` request asks the debug adapter to disconnect from the debuggee (thus ending the
debug session) and then to shut down itself (the debug adapter).

In addition, the debug adapter must terminate the debuggee if it was started with the `launch`
request. If an `attach` request was used to connect to the debuggee, then the debug adapter must not
terminate the debuggee.

This implicit behavior of when to terminate the debuggee can be overridden with the
`terminateDebuggee` argument (which is only supported by a debug adapter if the corresponding
capability `supportTerminateDebuggee` is true).

Note: automatically generated code. Do not edit manually.

## Class: DisconnectArguments

**Description:** Arguments for `disconnect` request.

Note: automatically generated code. Do not edit manually.

## Class: DisconnectResponse

**Description:** Response to `disconnect` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: TerminateRequest

**Description:** The `terminate` request is sent from the client to the debug adapter in order to shut down the
debuggee gracefully. Clients should only call this request if the capability
`supportsTerminateRequest` is true.

Typically a debug adapter implements `terminate` by sending a software signal which the debuggee
intercepts in order to clean things up properly before terminating itself.

Please note that this request does not directly affect the state of the debug session: if the
debuggee decides to veto the graceful shutdown for any reason by not terminating itself, then the
debug session just continues.

Clients can surface the `terminate` request as an explicit command or they can integrate it into a
two stage Stop command that first sends `terminate` to request a graceful shutdown, and if that
fails uses `disconnect` for a forceful shutdown.

Note: automatically generated code. Do not edit manually.

## Class: TerminateArguments

**Description:** Arguments for `terminate` request.

Note: automatically generated code. Do not edit manually.

## Class: TerminateResponse

**Description:** Response to `terminate` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: BreakpointLocationsRequest

**Description:** The `breakpointLocations` request returns all possible locations for source breakpoints in a given
range.

Clients should only call this request if the corresponding capability
`supportsBreakpointLocationsRequest` is true.

Note: automatically generated code. Do not edit manually.

## Class: BreakpointLocationsArguments

**Description:** Arguments for `breakpointLocations` request.

Note: automatically generated code. Do not edit manually.

## Class: BreakpointLocationsResponse

**Description:** Response to `breakpointLocations` request.

Contains possible locations for source breakpoints.

Note: automatically generated code. Do not edit manually.

## Class: SetBreakpointsRequest

**Description:** Sets multiple breakpoints for a single source and clears all previous breakpoints in that source.

To clear all breakpoint for a source, specify an empty array.

When a breakpoint is hit, a `stopped` event (with reason `breakpoint`) is generated.

Note: automatically generated code. Do not edit manually.

## Class: SetBreakpointsArguments

**Description:** Arguments for `setBreakpoints` request.

Note: automatically generated code. Do not edit manually.

## Class: SetBreakpointsResponse

**Description:** Response to `setBreakpoints` request.

Returned is information about each breakpoint created by this request.

This includes the actual code location and whether the breakpoint could be verified.

The breakpoints returned are in the same order as the elements of the `breakpoints`

(or the deprecated `lines`) array in the arguments.

Note: automatically generated code. Do not edit manually.

## Class: SetFunctionBreakpointsRequest

**Description:** Replaces all existing function breakpoints with new function breakpoints.

To clear all function breakpoints, specify an empty array.

When a function breakpoint is hit, a `stopped` event (with reason `function breakpoint`) is
generated.

Clients should only call this request if the corresponding capability `supportsFunctionBreakpoints`
is true.

Note: automatically generated code. Do not edit manually.

## Class: SetFunctionBreakpointsArguments

**Description:** Arguments for `setFunctionBreakpoints` request.

Note: automatically generated code. Do not edit manually.

## Class: SetFunctionBreakpointsResponse

**Description:** Response to `setFunctionBreakpoints` request.

Returned is information about each breakpoint created by this request.

Note: automatically generated code. Do not edit manually.

## Class: SetExceptionBreakpointsRequest

**Description:** The request configures the debugger's response to thrown exceptions.

If an exception is configured to break, a `stopped` event is fired (with reason `exception`).

Clients should only call this request if the corresponding capability `exceptionBreakpointFilters`
returns one or more filters.

Note: automatically generated code. Do not edit manually.

## Class: SetExceptionBreakpointsArguments

**Description:** Arguments for `setExceptionBreakpoints` request.

Note: automatically generated code. Do not edit manually.

## Class: SetExceptionBreakpointsResponse

**Description:** Response to `setExceptionBreakpoints` request.

The response contains an array of `Breakpoint` objects with information about each exception
breakpoint or filter. The `Breakpoint` objects are in the same order as the elements of the
`filters`, `filterOptions`, `exceptionOptions` arrays given as arguments. If both `filters` and
`filterOptions` are given, the returned array must start with `filters` information first, followed
by `filterOptions` information.

The `verified` property of a `Breakpoint` object signals whether the exception breakpoint or filter
could be successfully created and whether the condition is valid. In case of an error the `message`
property explains the problem. The `id` property can be used to introduce a unique ID for the
exception breakpoint or filter so that it can be updated subsequently by sending breakpoint events.

For backward compatibility both the `breakpoints` array and the enclosing `body` are optional. If
these elements are missing a client is not able to show problems for individual exception
breakpoints or filters.

Note: automatically generated code. Do not edit manually.

## Class: DataBreakpointInfoRequest

**Description:** Obtains information on a possible data breakpoint that could be set on an expression or variable.

Clients should only call this request if the corresponding capability `supportsDataBreakpoints` is
true.

Note: automatically generated code. Do not edit manually.

## Class: DataBreakpointInfoArguments

**Description:** Arguments for `dataBreakpointInfo` request.

Note: automatically generated code. Do not edit manually.

## Class: DataBreakpointInfoResponse

**Description:** Response to `dataBreakpointInfo` request.

Note: automatically generated code. Do not edit manually.

## Class: SetDataBreakpointsRequest

**Description:** Replaces all existing data breakpoints with new data breakpoints.

To clear all data breakpoints, specify an empty array.

When a data breakpoint is hit, a `stopped` event (with reason `data breakpoint`) is generated.

Clients should only call this request if the corresponding capability `supportsDataBreakpoints` is
true.

Note: automatically generated code. Do not edit manually.

## Class: SetDataBreakpointsArguments

**Description:** Arguments for `setDataBreakpoints` request.

Note: automatically generated code. Do not edit manually.

## Class: SetDataBreakpointsResponse

**Description:** Response to `setDataBreakpoints` request.

Returned is information about each breakpoint created by this request.

Note: automatically generated code. Do not edit manually.

## Class: SetInstructionBreakpointsRequest

**Description:** Replaces all existing instruction breakpoints. Typically, instruction breakpoints would be set from
a disassembly window.

To clear all instruction breakpoints, specify an empty array.

When an instruction breakpoint is hit, a `stopped` event (with reason `instruction breakpoint`) is
generated.

Clients should only call this request if the corresponding capability
`supportsInstructionBreakpoints` is true.

Note: automatically generated code. Do not edit manually.

## Class: SetInstructionBreakpointsArguments

**Description:** Arguments for `setInstructionBreakpoints` request

Note: automatically generated code. Do not edit manually.

## Class: SetInstructionBreakpointsResponse

**Description:** Response to `setInstructionBreakpoints` request

Note: automatically generated code. Do not edit manually.

## Class: ContinueRequest

**Description:** The request resumes execution of all threads. If the debug adapter supports single thread execution
(see capability `supportsSingleThreadExecutionRequests`), setting the `singleThread` argument to
true resumes only the specified thread. If not all threads were resumed, the `allThreadsContinued`
attribute of the response should be set to false.

Note: automatically generated code. Do not edit manually.

## Class: ContinueArguments

**Description:** Arguments for `continue` request.

Note: automatically generated code. Do not edit manually.

## Class: ContinueResponse

**Description:** Response to `continue` request.

Note: automatically generated code. Do not edit manually.

## Class: NextRequest

**Description:** The request executes one step (in the given granularity) for the specified thread and allows all
other threads to run freely by resuming them.

If the debug adapter supports single thread execution (see capability
`supportsSingleThreadExecutionRequests`), setting the `singleThread` argument to true prevents other
suspended threads from resuming.

The debug adapter first sends the response and then a `stopped` event (with reason `step`) after the
step has completed.

Note: automatically generated code. Do not edit manually.

## Class: NextArguments

**Description:** Arguments for `next` request.

Note: automatically generated code. Do not edit manually.

## Class: NextResponse

**Description:** Response to `next` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: StepInRequest

**Description:** The request resumes the given thread to step into a function/method and allows all other threads to
run freely by resuming them.

If the debug adapter supports single thread execution (see capability
`supportsSingleThreadExecutionRequests`), setting the `singleThread` argument to true prevents other
suspended threads from resuming.

If the request cannot step into a target, `stepIn` behaves like the `next` request.

The debug adapter first sends the response and then a `stopped` event (with reason `step`) after the
step has completed.

If there are multiple function/method calls (or other targets) on the source line,

the argument `targetId` can be used to control into which target the `stepIn` should occur.

The list of possible targets for a given source line can be retrieved via the `stepInTargets`
request.

Note: automatically generated code. Do not edit manually.

## Class: StepInArguments

**Description:** Arguments for `stepIn` request.

Note: automatically generated code. Do not edit manually.

## Class: StepInResponse

**Description:** Response to `stepIn` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: StepOutRequest

**Description:** The request resumes the given thread to step out (return) from a function/method and allows all
other threads to run freely by resuming them.

If the debug adapter supports single thread execution (see capability
`supportsSingleThreadExecutionRequests`), setting the `singleThread` argument to true prevents other
suspended threads from resuming.

The debug adapter first sends the response and then a `stopped` event (with reason `step`) after the
step has completed.

Note: automatically generated code. Do not edit manually.

## Class: StepOutArguments

**Description:** Arguments for `stepOut` request.

Note: automatically generated code. Do not edit manually.

## Class: StepOutResponse

**Description:** Response to `stepOut` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: StepBackRequest

**Description:** The request executes one backward step (in the given granularity) for the specified thread and
allows all other threads to run backward freely by resuming them.

If the debug adapter supports single thread execution (see capability
`supportsSingleThreadExecutionRequests`), setting the `singleThread` argument to true prevents other
suspended threads from resuming.

The debug adapter first sends the response and then a `stopped` event (with reason `step`) after the
step has completed.

Clients should only call this request if the corresponding capability `supportsStepBack` is true.

Note: automatically generated code. Do not edit manually.

## Class: StepBackArguments

**Description:** Arguments for `stepBack` request.

Note: automatically generated code. Do not edit manually.

## Class: StepBackResponse

**Description:** Response to `stepBack` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: ReverseContinueRequest

**Description:** The request resumes backward execution of all threads. If the debug adapter supports single thread
execution (see capability `supportsSingleThreadExecutionRequests`), setting the `singleThread`
argument to true resumes only the specified thread. If not all threads were resumed, the
`allThreadsContinued` attribute of the response should be set to false.

Clients should only call this request if the corresponding capability `supportsStepBack` is true.

Note: automatically generated code. Do not edit manually.

## Class: ReverseContinueArguments

**Description:** Arguments for `reverseContinue` request.

Note: automatically generated code. Do not edit manually.

## Class: ReverseContinueResponse

**Description:** Response to `reverseContinue` request. This is just an acknowledgement, so no body field is
required.

Note: automatically generated code. Do not edit manually.

## Class: RestartFrameRequest

**Description:** The request restarts execution of the specified stack frame.

The debug adapter first sends the response and then a `stopped` event (with reason `restart`) after
the restart has completed.

Clients should only call this request if the corresponding capability `supportsRestartFrame` is
true.

Note: automatically generated code. Do not edit manually.

## Class: RestartFrameArguments

**Description:** Arguments for `restartFrame` request.

Note: automatically generated code. Do not edit manually.

## Class: RestartFrameResponse

**Description:** Response to `restartFrame` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: GotoRequest

**Description:** The request sets the location where the debuggee will continue to run.

This makes it possible to skip the execution of code or to execute code again.

The code between the current location and the goto target is not executed but skipped.

The debug adapter first sends the response and then a `stopped` event with reason `goto`.

Clients should only call this request if the corresponding capability `supportsGotoTargetsRequest`
is true (because only then goto targets exist that can be passed as arguments).

Note: automatically generated code. Do not edit manually.

## Class: GotoArguments

**Description:** Arguments for `goto` request.

Note: automatically generated code. Do not edit manually.

## Class: GotoResponse

**Description:** Response to `goto` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: PauseRequest

**Description:** The request suspends the debuggee.

The debug adapter first sends the response and then a `stopped` event (with reason `pause`) after
the thread has been paused successfully.

Note: automatically generated code. Do not edit manually.

## Class: PauseArguments

**Description:** Arguments for `pause` request.

Note: automatically generated code. Do not edit manually.

## Class: PauseResponse

**Description:** Response to `pause` request. This is just an acknowledgement, so no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: StackTraceRequest

**Description:** The request returns a stacktrace from the current execution state of a given thread.

A client can request all stack frames by omitting the startFrame and levels arguments. For
performance-conscious clients and if the corresponding capability `supportsDelayedStackTraceLoading`
is true, stack frames can be retrieved in a piecemeal way with the `startFrame` and `levels`
arguments. The response of the `stackTrace` request may contain a `totalFrames` property that hints
at the total number of frames in the stack. If a client needs this total number upfront, it can
issue a request for a single (first) frame and depending on the value of `totalFrames` decide how to
proceed. In any case a client should be prepared to receive fewer frames than requested, which is an
indication that the end of the stack has been reached.

Note: automatically generated code. Do not edit manually.

## Class: StackTraceArguments

**Description:** Arguments for `stackTrace` request.

Note: automatically generated code. Do not edit manually.

## Class: StackTraceResponse

**Description:** Response to `stackTrace` request.

Note: automatically generated code. Do not edit manually.

## Class: ScopesRequest

**Description:** The request returns the variable scopes for a given stack frame ID.

Note: automatically generated code. Do not edit manually.

## Class: ScopesArguments

**Description:** Arguments for `scopes` request.

Note: automatically generated code. Do not edit manually.

## Class: ScopesResponse

**Description:** Response to `scopes` request.

Note: automatically generated code. Do not edit manually.

## Class: VariablesRequest

**Description:** Retrieves all child variables for the given variable reference.

A filter can be used to limit the fetched children to either named or indexed children.

Note: automatically generated code. Do not edit manually.

## Class: VariablesArguments

**Description:** Arguments for `variables` request.

Note: automatically generated code. Do not edit manually.

## Class: VariablesResponse

**Description:** Response to `variables` request.

Note: automatically generated code. Do not edit manually.

## Class: SetVariableRequest

**Description:** Set the variable with the given name in the variable container to a new value. Clients should only
call this request if the corresponding capability `supportsSetVariable` is true.

If a debug adapter implements both `setVariable` and `setExpression`, a client will only use
`setExpression` if the variable has an `evaluateName` property.

Note: automatically generated code. Do not edit manually.

## Class: SetVariableArguments

**Description:** Arguments for `setVariable` request.

Note: automatically generated code. Do not edit manually.

## Class: SetVariableResponse

**Description:** Response to `setVariable` request.

Note: automatically generated code. Do not edit manually.

## Class: SourceRequest

**Description:** The request retrieves the source code for a given source reference.

Note: automatically generated code. Do not edit manually.

## Class: SourceArguments

**Description:** Arguments for `source` request.

Note: automatically generated code. Do not edit manually.

## Class: SourceResponse

**Description:** Response to `source` request.

Note: automatically generated code. Do not edit manually.

## Class: ThreadsRequest

**Description:** The request retrieves a list of all threads.

Note: automatically generated code. Do not edit manually.

## Class: ThreadsResponse

**Description:** Response to `threads` request.

Note: automatically generated code. Do not edit manually.

## Class: TerminateThreadsRequest

**Description:** The request terminates the threads with the given ids.

Clients should only call this request if the corresponding capability
`supportsTerminateThreadsRequest` is true.

Note: automatically generated code. Do not edit manually.

## Class: TerminateThreadsArguments

**Description:** Arguments for `terminateThreads` request.

Note: automatically generated code. Do not edit manually.

## Class: TerminateThreadsResponse

**Description:** Response to `terminateThreads` request. This is just an acknowledgement, no body field is required.

Note: automatically generated code. Do not edit manually.

## Class: ModulesRequest

**Description:** Modules can be retrieved from the debug adapter with this request which can either return all
modules or a range of modules to support paging.

Clients should only call this request if the corresponding capability `supportsModulesRequest` is
true.

Note: automatically generated code. Do not edit manually.

## Class: ModulesArguments

**Description:** Arguments for `modules` request.

Note: automatically generated code. Do not edit manually.

## Class: ModulesResponse

**Description:** Response to `modules` request.

Note: automatically generated code. Do not edit manually.

## Class: LoadedSourcesRequest

**Description:** Retrieves the set of all sources currently loaded by the debugged process.

Clients should only call this request if the corresponding capability `supportsLoadedSourcesRequest`
is true.

Note: automatically generated code. Do not edit manually.

## Class: LoadedSourcesArguments

**Description:** Arguments for `loadedSources` request.

Note: automatically generated code. Do not edit manually.

## Class: LoadedSourcesResponse

**Description:** Response to `loadedSources` request.

Note: automatically generated code. Do not edit manually.

## Class: EvaluateRequest

**Description:** Evaluates the given expression in the context of the topmost stack frame.

The expression has access to any variables and arguments that are in scope.

Note: automatically generated code. Do not edit manually.

## Class: EvaluateArguments

**Description:** Arguments for `evaluate` request.

Note: automatically generated code. Do not edit manually.

## Class: EvaluateResponse

**Description:** Response to `evaluate` request.

Note: automatically generated code. Do not edit manually.

## Class: SetExpressionRequest

**Description:** Evaluates the given `value` expression and assigns it to the `expression` which must be a modifiable
l-value.

The expressions have access to any variables and arguments that are in scope of the specified frame.

Clients should only call this request if the corresponding capability `supportsSetExpression` is
true.

If a debug adapter implements both `setExpression` and `setVariable`, a client uses `setExpression`
if the variable has an `evaluateName` property.

Note: automatically generated code. Do not edit manually.

## Class: SetExpressionArguments

**Description:** Arguments for `setExpression` request.

Note: automatically generated code. Do not edit manually.

## Class: SetExpressionResponse

**Description:** Response to `setExpression` request.

Note: automatically generated code. Do not edit manually.

## Class: StepInTargetsRequest

**Description:** This request retrieves the possible step-in targets for the specified stack frame.

These targets can be used in the `stepIn` request.

Clients should only call this request if the corresponding capability `supportsStepInTargetsRequest`
is true.

Note: automatically generated code. Do not edit manually.

## Class: StepInTargetsArguments

**Description:** Arguments for `stepInTargets` request.

Note: automatically generated code. Do not edit manually.

## Class: StepInTargetsResponse

**Description:** Response to `stepInTargets` request.

Note: automatically generated code. Do not edit manually.

## Class: GotoTargetsRequest

**Description:** This request retrieves the possible goto targets for the specified source location.

These targets can be used in the `goto` request.

Clients should only call this request if the corresponding capability `supportsGotoTargetsRequest`
is true.

Note: automatically generated code. Do not edit manually.

## Class: GotoTargetsArguments

**Description:** Arguments for `gotoTargets` request.

Note: automatically generated code. Do not edit manually.

## Class: GotoTargetsResponse

**Description:** Response to `gotoTargets` request.

Note: automatically generated code. Do not edit manually.

## Class: CompletionsRequest

**Description:** Returns a list of possible completions for a given caret position and text.

Clients should only call this request if the corresponding capability `supportsCompletionsRequest`
is true.

Note: automatically generated code. Do not edit manually.

## Class: CompletionsArguments

**Description:** Arguments for `completions` request.

Note: automatically generated code. Do not edit manually.

## Class: CompletionsResponse

**Description:** Response to `completions` request.

Note: automatically generated code. Do not edit manually.

## Class: ExceptionInfoRequest

**Description:** Retrieves the details of the exception that caused this event to be raised.

Clients should only call this request if the corresponding capability `supportsExceptionInfoRequest`
is true.

Note: automatically generated code. Do not edit manually.

## Class: ExceptionInfoArguments

**Description:** Arguments for `exceptionInfo` request.

Note: automatically generated code. Do not edit manually.

## Class: ExceptionInfoResponse

**Description:** Response to `exceptionInfo` request.

Note: automatically generated code. Do not edit manually.

## Class: ReadMemoryRequest

**Description:** Reads bytes from memory at the provided location.

Clients should only call this request if the corresponding capability `supportsReadMemoryRequest` is
true.

Note: automatically generated code. Do not edit manually.

## Class: ReadMemoryArguments

**Description:** Arguments for `readMemory` request.

Note: automatically generated code. Do not edit manually.

## Class: ReadMemoryResponse

**Description:** Response to `readMemory` request.

Note: automatically generated code. Do not edit manually.

## Class: WriteMemoryRequest

**Description:** Writes bytes to memory at the provided location.

Clients should only call this request if the corresponding capability `supportsWriteMemoryRequest`
is true.

Note: automatically generated code. Do not edit manually.

## Class: WriteMemoryArguments

**Description:** Arguments for `writeMemory` request.

Note: automatically generated code. Do not edit manually.

## Class: WriteMemoryResponse

**Description:** Response to `writeMemory` request.

Note: automatically generated code. Do not edit manually.

## Class: DisassembleRequest

**Description:** Disassembles code stored at the provided location.

Clients should only call this request if the corresponding capability `supportsDisassembleRequest`
is true.

Note: automatically generated code. Do not edit manually.

## Class: DisassembleArguments

**Description:** Arguments for `disassemble` request.

Note: automatically generated code. Do not edit manually.

## Class: DisassembleResponse

**Description:** Response to `disassemble` request.

Note: automatically generated code. Do not edit manually.

## Class: Capabilities

**Description:** Information about the capabilities of a debug adapter.

Note: automatically generated code. Do not edit manually.

## Class: ExceptionBreakpointsFilter

**Description:** An `ExceptionBreakpointsFilter` is shown in the UI as an filter option for configuring how
exceptions are dealt with.

Note: automatically generated code. Do not edit manually.

## Class: Message

**Description:** A structured message object. Used to return errors from requests.

Note: automatically generated code. Do not edit manually.

## Class: Module

**Description:** A Module object represents a row in the modules view.

The `id` attribute identifies a module in the modules view and is used in a `module` event for
identifying a module for adding, updating or deleting.

The `name` attribute is used to minimally render the module in the UI.


Additional attributes can be added to the module. They show up in the module view if they have a
corresponding `ColumnDescriptor`.


To avoid an unnecessary proliferation of additional attributes with similar semantics but different
names, we recommend to re-use attributes from the 'recommended' list below first, and only introduce
new attributes if nothing appropriate could be found.

Note: automatically generated code. Do not edit manually.

## Class: ColumnDescriptor

**Description:** A `ColumnDescriptor` specifies what module attribute to show in a column of the modules view, how to
format it,

and what the column's label should be.

It is only used if the underlying UI actually supports this level of customization.

Note: automatically generated code. Do not edit manually.

## Class: Thread

**Description:** A Thread

Note: automatically generated code. Do not edit manually.

## Class: Source

**Description:** A `Source` is a descriptor for source code.

It is returned from the debug adapter as part of a `StackFrame` and it is used by clients when
specifying breakpoints.

Note: automatically generated code. Do not edit manually.

## Class: StackFrame

**Description:** A Stackframe contains the source location.

Note: automatically generated code. Do not edit manually.

## Class: Scope

**Description:** A `Scope` is a named container for variables. Optionally a scope can map to a source or a range
within a source.

Note: automatically generated code. Do not edit manually.

## Class: Variable

**Description:** A Variable is a name/value pair.

The `type` attribute is shown if space permits or when hovering over the variable's name.

The `kind` attribute is used to render additional properties of the variable, e.g. different icons
can be used to indicate that a variable is public or private.

If the value is structured (has children), a handle is provided to retrieve the children with the
`variables` request.

If the number of named or indexed children is large, the numbers should be returned via the
`namedVariables` and `indexedVariables` attributes.

The client can use this information to present the children in a paged UI and fetch them in chunks.

Note: automatically generated code. Do not edit manually.

## Class: VariablePresentationHint

**Description:** Properties of a variable that can be used to determine how to render the variable in the UI.

Note: automatically generated code. Do not edit manually.

## Class: BreakpointLocation

**Description:** Properties of a breakpoint location returned from the `breakpointLocations` request.

Note: automatically generated code. Do not edit manually.

## Class: SourceBreakpoint

**Description:** Properties of a breakpoint or logpoint passed to the `setBreakpoints` request.

Note: automatically generated code. Do not edit manually.

## Class: FunctionBreakpoint

**Description:** Properties of a breakpoint passed to the `setFunctionBreakpoints` request.

Note: automatically generated code. Do not edit manually.

## Class: DataBreakpointAccessType

**Description:** This enumeration defines all possible access types for data breakpoints.

Note: automatically generated code. Do not edit manually.

## Class: DataBreakpoint

**Description:** Properties of a data breakpoint passed to the `setDataBreakpoints` request.

Note: automatically generated code. Do not edit manually.

## Class: InstructionBreakpoint

**Description:** Properties of a breakpoint passed to the `setInstructionBreakpoints` request

Note: automatically generated code. Do not edit manually.

## Class: Breakpoint

**Description:** Information about a breakpoint created in `setBreakpoints`, `setFunctionBreakpoints`,
`setInstructionBreakpoints`, or `setDataBreakpoints` requests.

Note: automatically generated code. Do not edit manually.

## Class: SteppingGranularity

**Description:** The granularity of one 'step' in the stepping requests `next`, `stepIn`, `stepOut`, and `stepBack`.

Note: automatically generated code. Do not edit manually.

## Class: StepInTarget

**Description:** A `StepInTarget` can be used in the `stepIn` request and determines into which single target the
`stepIn` request should step.

Note: automatically generated code. Do not edit manually.

## Class: GotoTarget

**Description:** A `GotoTarget` describes a code location that can be used as a target in the `goto` request.

The possible goto targets can be determined via the `gotoTargets` request.

Note: automatically generated code. Do not edit manually.

## Class: CompletionItem

**Description:** `CompletionItems` are the suggestions returned from the `completions` request.

Note: automatically generated code. Do not edit manually.

## Class: CompletionItemType

**Description:** Some predefined types for the CompletionItem. Please note that not all clients have specific icons
for all of them.

Note: automatically generated code. Do not edit manually.

## Class: ChecksumAlgorithm

**Description:** Names of checksum algorithms that may be supported by a debug adapter.

Note: automatically generated code. Do not edit manually.

## Class: Checksum

**Description:** The checksum of an item calculated by the specified algorithm.

Note: automatically generated code. Do not edit manually.

## Class: ValueFormat

**Description:** Provides formatting information for a value.

Note: automatically generated code. Do not edit manually.

## Class: StackFrameFormat

**Description:** Provides formatting information for a stack frame.

Note: automatically generated code. Do not edit manually.

## Class: ExceptionFilterOptions

**Description:** An `ExceptionFilterOptions` is used to specify an exception filter together with a condition for the
`setExceptionBreakpoints` request.

Note: automatically generated code. Do not edit manually.

## Class: ExceptionOptions

**Description:** An `ExceptionOptions` assigns configuration options to a set of exceptions.

Note: automatically generated code. Do not edit manually.

## Class: ExceptionBreakMode

**Description:** This enumeration defines all possible conditions when a thrown exception should result in a break.

never: never breaks,

always: always breaks,

unhandled: breaks when exception unhandled,

userUnhandled: breaks if the exception is not handled by user code.

Note: automatically generated code. Do not edit manually.

## Class: ExceptionPathSegment

**Description:** An `ExceptionPathSegment` represents a segment in a path that is used to match leafs or nodes in a
tree of exceptions.

If a segment consists of more than one name, it matches the names provided if `negate` is false or
missing, or it matches anything except the names provided if `negate` is true.

Note: automatically generated code. Do not edit manually.

## Class: ExceptionDetails

**Description:** Detailed information about an exception that has occurred.

Note: automatically generated code. Do not edit manually.

## Class: DisassembledInstruction

**Description:** Represents a single disassembled instruction.

Note: automatically generated code. Do not edit manually.

## Class: InvalidatedAreas

**Description:** Logical areas that can be invalidated by the `invalidated` event.

Note: automatically generated code. Do not edit manually.

## Class: SetDebuggerPropertyRequest

**Description:** The request can be used to enable or disable debugger features.

Note: automatically generated code. Do not edit manually.

## Class: SetDebuggerPropertyArguments

**Description:** Arguments for 'setDebuggerProperty' request.

Note: automatically generated code. Do not edit manually.

## Class: SetDebuggerPropertyResponse

**Description:** Response to 'setDebuggerProperty' request. This is just an acknowledgement, so no body field is
required.

Note: automatically generated code. Do not edit manually.

## Class: PydevdInputRequestedEvent

**Description:** The event indicates input was requested by debuggee.

Note: automatically generated code. Do not edit manually.

## Class: SetPydevdSourceMapRequest

**Description:** Sets multiple PydevdSourceMap for a single source and clears all previous PydevdSourceMap in that
source.

i.e.: Maps paths and lines in a 1:N mapping (use case: map a single file in the IDE to multiple
IPython cells).

To clear all PydevdSourceMap for a source, specify an empty array.

Interaction with breakpoints: When a new mapping is sent, breakpoints that match the source (or
previously matched a source) are reapplied.

Interaction with launch pathMapping: both mappings are independent. This mapping is applied after
the launch pathMapping.

Note: automatically generated code. Do not edit manually.

## Class: SetPydevdSourceMapArguments

**Description:** Arguments for 'setPydevdSourceMap' request.

Note: automatically generated code. Do not edit manually.

## Class: SetPydevdSourceMapResponse

**Description:** Response to 'setPydevdSourceMap' request. This is just an acknowledgement, so no body field is
required.

Note: automatically generated code. Do not edit manually.

## Class: PydevdSourceMap

**Description:** Information that allows mapping a local line to a remote source/line.

Note: automatically generated code. Do not edit manually.

## Class: PydevdSystemInfoRequest

**Description:** The request can be used retrieve system information, python version, etc.

Note: automatically generated code. Do not edit manually.

## Class: PydevdSystemInfoArguments

**Description:** Arguments for 'pydevdSystemInfo' request.

Note: automatically generated code. Do not edit manually.

## Class: PydevdSystemInfoResponse

**Description:** Response to 'pydevdSystemInfo' request.

Note: automatically generated code. Do not edit manually.

## Class: PydevdPythonInfo

**Description:** This object contains python version and implementation details.

Note: automatically generated code. Do not edit manually.

## Class: PydevdPythonImplementationInfo

**Description:** This object contains python implementation details.

Note: automatically generated code. Do not edit manually.

## Class: PydevdPlatformInfo

**Description:** This object contains python version and implementation details.

Note: automatically generated code. Do not edit manually.

## Class: PydevdProcessInfo

**Description:** This object contains python process details.

Note: automatically generated code. Do not edit manually.

## Class: PydevdInfo

**Description:** This object contains details on pydevd.

Note: automatically generated code. Do not edit manually.

## Class: PydevdAuthorizeRequest

**Description:** A request to authorize the ide to start accepting commands.

Note: automatically generated code. Do not edit manually.

## Class: PydevdAuthorizeArguments

**Description:** Arguments for 'pydevdAuthorize' request.

Note: automatically generated code. Do not edit manually.

## Class: PydevdAuthorizeResponse

**Description:** Response to 'pydevdAuthorize' request.

Note: automatically generated code. Do not edit manually.

## Class: ErrorResponseBody

**Description:** "body" of ErrorResponse

Note: automatically generated code. Do not edit manually.

## Class: StoppedEventBody

**Description:** "body" of StoppedEvent

Note: automatically generated code. Do not edit manually.

## Class: ContinuedEventBody

**Description:** "body" of ContinuedEvent

Note: automatically generated code. Do not edit manually.

## Class: ExitedEventBody

**Description:** "body" of ExitedEvent

Note: automatically generated code. Do not edit manually.

## Class: TerminatedEventBody

**Description:** "body" of TerminatedEvent

Note: automatically generated code. Do not edit manually.

## Class: ThreadEventBody

**Description:** "body" of ThreadEvent

Note: automatically generated code. Do not edit manually.

## Class: OutputEventBody

**Description:** "body" of OutputEvent

Note: automatically generated code. Do not edit manually.

## Class: BreakpointEventBody

**Description:** "body" of BreakpointEvent

Note: automatically generated code. Do not edit manually.

## Class: ModuleEventBody

**Description:** "body" of ModuleEvent

Note: automatically generated code. Do not edit manually.

## Class: LoadedSourceEventBody

**Description:** "body" of LoadedSourceEvent

Note: automatically generated code. Do not edit manually.

## Class: ProcessEventBody

**Description:** "body" of ProcessEvent

Note: automatically generated code. Do not edit manually.

## Class: CapabilitiesEventBody

**Description:** "body" of CapabilitiesEvent

Note: automatically generated code. Do not edit manually.

## Class: ProgressStartEventBody

**Description:** "body" of ProgressStartEvent

Note: automatically generated code. Do not edit manually.

## Class: ProgressUpdateEventBody

**Description:** "body" of ProgressUpdateEvent

Note: automatically generated code. Do not edit manually.

## Class: ProgressEndEventBody

**Description:** "body" of ProgressEndEvent

Note: automatically generated code. Do not edit manually.

## Class: InvalidatedEventBody

**Description:** "body" of InvalidatedEvent

Note: automatically generated code. Do not edit manually.

## Class: MemoryEventBody

**Description:** "body" of MemoryEvent

Note: automatically generated code. Do not edit manually.

## Class: RunInTerminalRequestArgumentsEnv

**Description:** "env" of RunInTerminalRequestArguments

Note: automatically generated code. Do not edit manually.

## Class: RunInTerminalResponseBody

**Description:** "body" of RunInTerminalResponse

Note: automatically generated code. Do not edit manually.

## Class: StartDebuggingRequestArgumentsConfiguration

**Description:** "configuration" of StartDebuggingRequestArguments

Note: automatically generated code. Do not edit manually.

## Class: BreakpointLocationsResponseBody

**Description:** "body" of BreakpointLocationsResponse

Note: automatically generated code. Do not edit manually.

## Class: SetBreakpointsResponseBody

**Description:** "body" of SetBreakpointsResponse

Note: automatically generated code. Do not edit manually.

## Class: SetFunctionBreakpointsResponseBody

**Description:** "body" of SetFunctionBreakpointsResponse

Note: automatically generated code. Do not edit manually.

## Class: SetExceptionBreakpointsResponseBody

**Description:** "body" of SetExceptionBreakpointsResponse

Note: automatically generated code. Do not edit manually.

## Class: DataBreakpointInfoResponseBody

**Description:** "body" of DataBreakpointInfoResponse

Note: automatically generated code. Do not edit manually.

## Class: SetDataBreakpointsResponseBody

**Description:** "body" of SetDataBreakpointsResponse

Note: automatically generated code. Do not edit manually.

## Class: SetInstructionBreakpointsResponseBody

**Description:** "body" of SetInstructionBreakpointsResponse

Note: automatically generated code. Do not edit manually.

## Class: ContinueResponseBody

**Description:** "body" of ContinueResponse

Note: automatically generated code. Do not edit manually.

## Class: StackTraceResponseBody

**Description:** "body" of StackTraceResponse

Note: automatically generated code. Do not edit manually.

## Class: ScopesResponseBody

**Description:** "body" of ScopesResponse

Note: automatically generated code. Do not edit manually.

## Class: VariablesResponseBody

**Description:** "body" of VariablesResponse

Note: automatically generated code. Do not edit manually.

## Class: SetVariableResponseBody

**Description:** "body" of SetVariableResponse

Note: automatically generated code. Do not edit manually.

## Class: SourceResponseBody

**Description:** "body" of SourceResponse

Note: automatically generated code. Do not edit manually.

## Class: ThreadsResponseBody

**Description:** "body" of ThreadsResponse

Note: automatically generated code. Do not edit manually.

## Class: ModulesResponseBody

**Description:** "body" of ModulesResponse

Note: automatically generated code. Do not edit manually.

## Class: LoadedSourcesResponseBody

**Description:** "body" of LoadedSourcesResponse

Note: automatically generated code. Do not edit manually.

## Class: EvaluateResponseBody

**Description:** "body" of EvaluateResponse

Note: automatically generated code. Do not edit manually.

## Class: SetExpressionResponseBody

**Description:** "body" of SetExpressionResponse

Note: automatically generated code. Do not edit manually.

## Class: StepInTargetsResponseBody

**Description:** "body" of StepInTargetsResponse

Note: automatically generated code. Do not edit manually.

## Class: GotoTargetsResponseBody

**Description:** "body" of GotoTargetsResponse

Note: automatically generated code. Do not edit manually.

## Class: CompletionsResponseBody

**Description:** "body" of CompletionsResponse

Note: automatically generated code. Do not edit manually.

## Class: ExceptionInfoResponseBody

**Description:** "body" of ExceptionInfoResponse

Note: automatically generated code. Do not edit manually.

## Class: ReadMemoryResponseBody

**Description:** "body" of ReadMemoryResponse

Note: automatically generated code. Do not edit manually.

## Class: WriteMemoryResponseBody

**Description:** "body" of WriteMemoryResponse

Note: automatically generated code. Do not edit manually.

## Class: DisassembleResponseBody

**Description:** "body" of DisassembleResponse

Note: automatically generated code. Do not edit manually.

## Class: MessageVariables

**Description:** "variables" of Message

Note: automatically generated code. Do not edit manually.

## Class: PydevdSystemInfoResponseBody

**Description:** "body" of PydevdSystemInfoResponse

Note: automatically generated code. Do not edit manually.

## Class: PydevdAuthorizeResponseBody

**Description:** "body" of PydevdAuthorizeResponse

Note: automatically generated code. Do not edit manually.

### Function: __init__(self, type, seq, update_ids_from_dap)

**Description:** :param string type: Message type.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, command, seq, arguments, update_ids_from_dap)

**Description:** :param string type:
:param string command: The command to execute.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] arguments: Object containing arguments for the command.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, event, seq, body, update_ids_from_dap)

**Description:** :param string type:
:param string event: Type of event.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Event-specific information.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param ErrorResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, seq, arguments, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param CancelArguments arguments:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, requestId, progressId, update_ids_from_dap)

**Description:** :param integer requestId: The ID (attribute `seq`) of the request to cancel. If missing no request is cancelled.
Both a `requestId` and a `progressId` can be specified in one request.
:param string progressId: The ID (attribute `progressId`) of the progress to cancel. If missing no progress is cancelled.
Both a `requestId` and a `progressId` can be specified in one request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, seq, body, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Event-specific information.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param StoppedEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param ContinuedEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param ExitedEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, seq, body, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param TerminatedEventBody body:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param ThreadEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param OutputEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param BreakpointEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param ModuleEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param LoadedSourceEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param ProcessEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param CapabilitiesEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param ProgressStartEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param ProgressUpdateEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param ProgressEndEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param InvalidatedEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, body, seq, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param MemoryEventBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param RunInTerminalRequestArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, cwd, args, kind, title, env, argsCanBeInterpretedByShell, update_ids_from_dap)

**Description:** :param string cwd: Working directory for the command. For non-empty, valid paths this typically results in execution of a change directory command.
:param array args: List of arguments. The first argument is the command to run.
:param string kind: What kind of terminal to launch. Defaults to `integrated` if not specified.
:param string title: Title of the terminal.
:param RunInTerminalRequestArgumentsEnv env: Environment key-value pairs that are added to or removed from the default environment.
:param boolean argsCanBeInterpretedByShell: This property should only be set if the corresponding capability `supportsArgsCanBeInterpretedByShell` is true. If the client uses an intermediary shell to launch the application, then the client must not attempt to escape characters with special meanings for the shell. The user is fully responsible for escaping as needed and that arguments using special characters may not be portable across shells.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param RunInTerminalResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param StartDebuggingRequestArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, configuration, request, update_ids_from_dap)

**Description:** :param StartDebuggingRequestArgumentsConfiguration configuration: Arguments passed to the new debug session. The arguments must only contain properties understood by the `launch` or `attach` requests of the debug adapter and they must not contain any client-specific properties (e.g. `type`) or client-specific features (e.g. substitutable 'variables').
:param string request: Indicates whether the new debug session should be started with a `launch` or `attach` request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param InitializeRequestArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, adapterID, clientID, clientName, locale, linesStartAt1, columnsStartAt1, pathFormat, supportsVariableType, supportsVariablePaging, supportsRunInTerminalRequest, supportsMemoryReferences, supportsProgressReporting, supportsInvalidatedEvent, supportsMemoryEvent, supportsArgsCanBeInterpretedByShell, supportsStartDebuggingRequest, update_ids_from_dap)

**Description:** :param string adapterID: The ID of the debug adapter.
:param string clientID: The ID of the client using this adapter.
:param string clientName: The human-readable name of the client using this adapter.
:param string locale: The ISO-639 locale of the client using this adapter, e.g. en-US or de-CH.
:param boolean linesStartAt1: If true all line numbers are 1-based (default).
:param boolean columnsStartAt1: If true all column numbers are 1-based (default).
:param string pathFormat: Determines in what format paths are specified. The default is `path`, which is the native format.
:param boolean supportsVariableType: Client supports the `type` attribute for variables.
:param boolean supportsVariablePaging: Client supports the paging of variables.
:param boolean supportsRunInTerminalRequest: Client supports the `runInTerminal` request.
:param boolean supportsMemoryReferences: Client supports memory references.
:param boolean supportsProgressReporting: Client supports progress reporting.
:param boolean supportsInvalidatedEvent: Client supports the `invalidated` event.
:param boolean supportsMemoryEvent: Client supports the `memory` event.
:param boolean supportsArgsCanBeInterpretedByShell: Client supports the `argsCanBeInterpretedByShell` attribute on the `runInTerminal` request.
:param boolean supportsStartDebuggingRequest: Client supports the `startDebugging` request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param Capabilities body: The capabilities of this debug adapter.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, seq, arguments, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param ConfigurationDoneArguments arguments:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, update_ids_from_dap)

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param LaunchRequestArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, noDebug, __restart, update_ids_from_dap)

**Description:** :param boolean noDebug: If true, the launch request should launch the program without enabling debugging.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] __restart: Arbitrary data from the previous, restarted session.
The data is sent as the `restart` attribute of the `terminated` event.
The client should leave the data intact.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param AttachRequestArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, __restart, update_ids_from_dap)

**Description:** :param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] __restart: Arbitrary data from the previous, restarted session.
The data is sent as the `restart` attribute of the `terminated` event.
The client should leave the data intact.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, seq, arguments, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param RestartArguments arguments:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, update_ids_from_dap)

**Description:** :param TypeNA arguments: The latest version of the `launch` or `attach` configuration.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, seq, arguments, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param DisconnectArguments arguments:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, restart, terminateDebuggee, suspendDebuggee, update_ids_from_dap)

**Description:** :param boolean restart: A value of true indicates that this `disconnect` request is part of a restart sequence.
:param boolean terminateDebuggee: Indicates whether the debuggee should be terminated when the debugger is disconnected.
If unspecified, the debug adapter is free to do whatever it thinks is best.
The attribute is only honored by a debug adapter if the corresponding capability `supportTerminateDebuggee` is true.
:param boolean suspendDebuggee: Indicates whether the debuggee should stay suspended when the debugger is disconnected.
If unspecified, the debuggee should resume execution.
The attribute is only honored by a debug adapter if the corresponding capability `supportSuspendDebuggee` is true.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, seq, arguments, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param TerminateArguments arguments:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, restart, update_ids_from_dap)

**Description:** :param boolean restart: A value of true indicates that this `terminate` request is part of a restart sequence.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, seq, arguments, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param BreakpointLocationsArguments arguments:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, source, line, column, endLine, endColumn, update_ids_from_dap)

**Description:** :param Source source: The source location of the breakpoints; either `source.path` or `source.sourceReference` must be specified.
:param integer line: Start line of range to search possible breakpoint locations in. If only the line is specified, the request returns all possible locations in that line.
:param integer column: Start position within `line` to search possible breakpoint locations in. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based. If no column is given, the first position in the start line is assumed.
:param integer endLine: End line of range to search possible breakpoint locations in. If no end line is given, then the end line is assumed to be the start line.
:param integer endColumn: End position within `endLine` to search possible breakpoint locations in. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based. If no end column is given, the last position in the end line is assumed.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param BreakpointLocationsResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param SetBreakpointsArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, source, breakpoints, lines, sourceModified, update_ids_from_dap)

**Description:** :param Source source: The source location of the breakpoints; either `source.path` or `source.sourceReference` must be specified.
:param array breakpoints: The code locations of the breakpoints.
:param array lines: Deprecated: The code locations of the breakpoints.
:param boolean sourceModified: A value of true indicates that the underlying source has been modified which results in new breakpoint locations.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param SetBreakpointsResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param SetFunctionBreakpointsArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, breakpoints, update_ids_from_dap)

**Description:** :param array breakpoints: The function names of the breakpoints.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param SetFunctionBreakpointsResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param SetExceptionBreakpointsArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, filters, filterOptions, exceptionOptions, update_ids_from_dap)

**Description:** :param array filters: Set of exception filters specified by their ID. The set of all possible exception filters is defined by the `exceptionBreakpointFilters` capability. The `filter` and `filterOptions` sets are additive.
:param array filterOptions: Set of exception filters and their options. The set of all possible exception filters is defined by the `exceptionBreakpointFilters` capability. This attribute is only honored by a debug adapter if the corresponding capability `supportsExceptionFilterOptions` is true. The `filter` and `filterOptions` sets are additive.
:param array exceptionOptions: Configuration options for selected exceptions.
The attribute is only honored by a debug adapter if the corresponding capability `supportsExceptionOptions` is true.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param SetExceptionBreakpointsResponseBody body:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param DataBreakpointInfoArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, name, variablesReference, frameId, update_ids_from_dap)

**Description:** :param string name: The name of the variable's child to obtain data breakpoint information for.
If `variablesReference` isn't specified, this can be an expression.
:param integer variablesReference: Reference to the variable container if the data breakpoint is requested for a child of the container. The `variablesReference` must have been obtained in the current suspended state. See 'Lifetime of Object References' in the Overview section for details.
:param integer frameId: When `name` is an expression, evaluate it in the scope of this stack frame. If not specified, the expression is evaluated in the global scope. When `variablesReference` is specified, this property has no effect.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param DataBreakpointInfoResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param SetDataBreakpointsArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, breakpoints, update_ids_from_dap)

**Description:** :param array breakpoints: The contents of this array replaces all existing data breakpoints. An empty array clears all data breakpoints.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param SetDataBreakpointsResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param SetInstructionBreakpointsArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, breakpoints, update_ids_from_dap)

**Description:** :param array breakpoints: The instruction references of the breakpoints

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param SetInstructionBreakpointsResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param ContinueArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, threadId, singleThread, update_ids_from_dap)

**Description:** :param integer threadId: Specifies the active thread. If the debug adapter supports single thread execution (see `supportsSingleThreadExecutionRequests`) and the argument `singleThread` is true, only the thread with this ID is resumed.
:param boolean singleThread: If this flag is true, execution is resumed only for the thread with given `threadId`.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param ContinueResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param NextArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, threadId, singleThread, granularity, update_ids_from_dap)

**Description:** :param integer threadId: Specifies the thread for which to resume execution for one step (of the given granularity).
:param boolean singleThread: If this flag is true, all other suspended threads are not resumed.
:param SteppingGranularity granularity: Stepping granularity. If no granularity is specified, a granularity of `statement` is assumed.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param StepInArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, threadId, singleThread, targetId, granularity, update_ids_from_dap)

**Description:** :param integer threadId: Specifies the thread for which to resume execution for one step-into (of the given granularity).
:param boolean singleThread: If this flag is true, all other suspended threads are not resumed.
:param integer targetId: Id of the target to step into.
:param SteppingGranularity granularity: Stepping granularity. If no granularity is specified, a granularity of `statement` is assumed.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param StepOutArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, threadId, singleThread, granularity, update_ids_from_dap)

**Description:** :param integer threadId: Specifies the thread for which to resume execution for one step-out (of the given granularity).
:param boolean singleThread: If this flag is true, all other suspended threads are not resumed.
:param SteppingGranularity granularity: Stepping granularity. If no granularity is specified, a granularity of `statement` is assumed.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param StepBackArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, threadId, singleThread, granularity, update_ids_from_dap)

**Description:** :param integer threadId: Specifies the thread for which to resume execution for one step backwards (of the given granularity).
:param boolean singleThread: If this flag is true, all other suspended threads are not resumed.
:param SteppingGranularity granularity: Stepping granularity to step. If no granularity is specified, a granularity of `statement` is assumed.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param ReverseContinueArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, threadId, singleThread, update_ids_from_dap)

**Description:** :param integer threadId: Specifies the active thread. If the debug adapter supports single thread execution (see `supportsSingleThreadExecutionRequests`) and the `singleThread` argument is true, only the thread with this ID is resumed.
:param boolean singleThread: If this flag is true, backward execution is resumed only for the thread with given `threadId`.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param RestartFrameArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, frameId, update_ids_from_dap)

**Description:** :param integer frameId: Restart the stack frame identified by `frameId`. The `frameId` must have been obtained in the current suspended state. See 'Lifetime of Object References' in the Overview section for details.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param GotoArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, threadId, targetId, update_ids_from_dap)

**Description:** :param integer threadId: Set the goto target for this thread.
:param integer targetId: The location where the debuggee will continue to run.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param PauseArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, threadId, update_ids_from_dap)

**Description:** :param integer threadId: Pause execution for this thread.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param StackTraceArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, threadId, startFrame, levels, format, update_ids_from_dap)

**Description:** :param integer threadId: Retrieve the stacktrace for this thread.
:param integer startFrame: The index of the first frame to return; if omitted frames start at 0.
:param integer levels: The maximum number of frames to return. If levels is not specified or 0, all frames are returned.
:param StackFrameFormat format: Specifies details on how to format the stack frames.
The attribute is only honored by a debug adapter if the corresponding capability `supportsValueFormattingOptions` is true.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param StackTraceResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param ScopesArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, frameId, update_ids_from_dap)

**Description:** :param integer frameId: Retrieve the scopes for the stack frame identified by `frameId`. The `frameId` must have been obtained in the current suspended state. See 'Lifetime of Object References' in the Overview section for details.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param ScopesResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param VariablesArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, variablesReference, filter, start, count, format, update_ids_from_dap)

**Description:** :param integer variablesReference: The variable for which to retrieve its children. The `variablesReference` must have been obtained in the current suspended state. See 'Lifetime of Object References' in the Overview section for details.
:param string filter: Filter to limit the child variables to either named or indexed. If omitted, both types are fetched.
:param integer start: The index of the first variable to return; if omitted children start at 0.
The attribute is only honored by a debug adapter if the corresponding capability `supportsVariablePaging` is true.
:param integer count: The number of variables to return. If count is missing or 0, all variables are returned.
The attribute is only honored by a debug adapter if the corresponding capability `supportsVariablePaging` is true.
:param ValueFormat format: Specifies details on how to format the Variable values.
The attribute is only honored by a debug adapter if the corresponding capability `supportsValueFormattingOptions` is true.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param VariablesResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param SetVariableArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, variablesReference, name, value, format, update_ids_from_dap)

**Description:** :param integer variablesReference: The reference of the variable container. The `variablesReference` must have been obtained in the current suspended state. See 'Lifetime of Object References' in the Overview section for details.
:param string name: The name of the variable in the container.
:param string value: The value of the variable.
:param ValueFormat format: Specifies details on how to format the response value.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param SetVariableResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param SourceArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, sourceReference, source, update_ids_from_dap)

**Description:** :param integer sourceReference: The reference to the source. This is the same as `source.sourceReference`.
This is provided for backward compatibility since old clients do not understand the `source` attribute.
:param Source source: Specifies the source content to load. Either `source.path` or `source.sourceReference` must be specified.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param SourceResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, seq, arguments, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] arguments: Object containing arguments for the command.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param ThreadsResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param TerminateThreadsArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, threadIds, update_ids_from_dap)

**Description:** :param array threadIds: Ids of threads to be terminated.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, seq, arguments, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param ModulesArguments arguments:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, startModule, moduleCount, update_ids_from_dap)

**Description:** :param integer startModule: The index of the first module to return; if omitted modules start at 0.
:param integer moduleCount: The number of modules to return. If `moduleCount` is not specified or 0, all modules are returned.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param ModulesResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, seq, arguments, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param LoadedSourcesArguments arguments:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, update_ids_from_dap)

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param LoadedSourcesResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param EvaluateArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, expression, frameId, context, format, update_ids_from_dap)

**Description:** :param string expression: The expression to evaluate.
:param integer frameId: Evaluate the expression in the scope of this stack frame. If not specified, the expression is evaluated in the global scope.
:param string context: The context in which the evaluate request is used.
:param ValueFormat format: Specifies details on how to format the result.
The attribute is only honored by a debug adapter if the corresponding capability `supportsValueFormattingOptions` is true.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param EvaluateResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param SetExpressionArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, expression, value, frameId, format, update_ids_from_dap)

**Description:** :param string expression: The l-value expression to assign to.
:param string value: The value expression to assign to the l-value expression.
:param integer frameId: Evaluate the expressions in the scope of this stack frame. If not specified, the expressions are evaluated in the global scope.
:param ValueFormat format: Specifies how the resulting value should be formatted.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param SetExpressionResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param StepInTargetsArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, frameId, update_ids_from_dap)

**Description:** :param integer frameId: The stack frame for which to retrieve the possible step-in targets.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param StepInTargetsResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param GotoTargetsArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, source, line, column, update_ids_from_dap)

**Description:** :param Source source: The source location for which the goto targets are determined.
:param integer line: The line location for which the goto targets are determined.
:param integer column: The position within `line` for which the goto targets are determined. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param GotoTargetsResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param CompletionsArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, text, column, frameId, line, update_ids_from_dap)

**Description:** :param string text: One or more source lines. Typically this is the text users have typed into the debug console before they asked for completion.
:param integer column: The position within `text` for which to determine the completion proposals. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based.
:param integer frameId: Returns completions in the scope of this stack frame. If not specified, the completions are returned for the global scope.
:param integer line: A line for which to determine the completion proposals. If missing the first line of the text is assumed.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param CompletionsResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param ExceptionInfoArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, threadId, update_ids_from_dap)

**Description:** :param integer threadId: Thread for which exception information should be retrieved.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param ExceptionInfoResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param ReadMemoryArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, memoryReference, count, offset, update_ids_from_dap)

**Description:** :param string memoryReference: Memory reference to the base location from which data should be read.
:param integer count: Number of bytes to read at the specified location and offset.
:param integer offset: Offset (in bytes) to be applied to the reference location before reading data. Can be negative.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ReadMemoryResponseBody body:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param WriteMemoryArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, memoryReference, data, offset, allowPartial, update_ids_from_dap)

**Description:** :param string memoryReference: Memory reference to the base location to which data should be written.
:param string data: Bytes to write, encoded using base64.
:param integer offset: Offset (in bytes) to be applied to the reference location before writing data. Can be negative.
:param boolean allowPartial: Property to control partial writes. If true, the debug adapter should attempt to write memory even if the entire memory region is not writable. In such a case the debug adapter should stop after hitting the first byte of memory that cannot be written and return the number of bytes written in the response via the `offset` and `bytesWritten` properties.
If false or missing, a debug adapter should attempt to verify the region is writable before writing, and fail the response if it is not.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param WriteMemoryResponseBody body:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param DisassembleArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, memoryReference, instructionCount, offset, instructionOffset, resolveSymbols, update_ids_from_dap)

**Description:** :param string memoryReference: Memory reference to the base location containing the instructions to disassemble.
:param integer instructionCount: Number of instructions to disassemble starting at the specified location and offset.
An adapter must return exactly this number of instructions - any unavailable instructions should be replaced with an implementation-defined 'invalid instruction' value.
:param integer offset: Offset (in bytes) to be applied to the reference location before disassembling. Can be negative.
:param integer instructionOffset: Offset (in instructions) to be applied after the byte offset (if any) before disassembling. Can be negative.
:param boolean resolveSymbols: If true, the adapter should attempt to resolve memory addresses and other values to symbolic names.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param DisassembleResponseBody body:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, supportsConfigurationDoneRequest, supportsFunctionBreakpoints, supportsConditionalBreakpoints, supportsHitConditionalBreakpoints, supportsEvaluateForHovers, exceptionBreakpointFilters, supportsStepBack, supportsSetVariable, supportsRestartFrame, supportsGotoTargetsRequest, supportsStepInTargetsRequest, supportsCompletionsRequest, completionTriggerCharacters, supportsModulesRequest, additionalModuleColumns, supportedChecksumAlgorithms, supportsRestartRequest, supportsExceptionOptions, supportsValueFormattingOptions, supportsExceptionInfoRequest, supportTerminateDebuggee, supportSuspendDebuggee, supportsDelayedStackTraceLoading, supportsLoadedSourcesRequest, supportsLogPoints, supportsTerminateThreadsRequest, supportsSetExpression, supportsTerminateRequest, supportsDataBreakpoints, supportsReadMemoryRequest, supportsWriteMemoryRequest, supportsDisassembleRequest, supportsCancelRequest, supportsBreakpointLocationsRequest, supportsClipboardContext, supportsSteppingGranularity, supportsInstructionBreakpoints, supportsExceptionFilterOptions, supportsSingleThreadExecutionRequests, update_ids_from_dap)

**Description:** :param boolean supportsConfigurationDoneRequest: The debug adapter supports the `configurationDone` request.
:param boolean supportsFunctionBreakpoints: The debug adapter supports function breakpoints.
:param boolean supportsConditionalBreakpoints: The debug adapter supports conditional breakpoints.
:param boolean supportsHitConditionalBreakpoints: The debug adapter supports breakpoints that break execution after a specified number of hits.
:param boolean supportsEvaluateForHovers: The debug adapter supports a (side effect free) `evaluate` request for data hovers.
:param array exceptionBreakpointFilters: Available exception filter options for the `setExceptionBreakpoints` request.
:param boolean supportsStepBack: The debug adapter supports stepping back via the `stepBack` and `reverseContinue` requests.
:param boolean supportsSetVariable: The debug adapter supports setting a variable to a value.
:param boolean supportsRestartFrame: The debug adapter supports restarting a frame.
:param boolean supportsGotoTargetsRequest: The debug adapter supports the `gotoTargets` request.
:param boolean supportsStepInTargetsRequest: The debug adapter supports the `stepInTargets` request.
:param boolean supportsCompletionsRequest: The debug adapter supports the `completions` request.
:param array completionTriggerCharacters: The set of characters that should trigger completion in a REPL. If not specified, the UI should assume the `.` character.
:param boolean supportsModulesRequest: The debug adapter supports the `modules` request.
:param array additionalModuleColumns: The set of additional module information exposed by the debug adapter.
:param array supportedChecksumAlgorithms: Checksum algorithms supported by the debug adapter.
:param boolean supportsRestartRequest: The debug adapter supports the `restart` request. In this case a client should not implement `restart` by terminating and relaunching the adapter but by calling the `restart` request.
:param boolean supportsExceptionOptions: The debug adapter supports `exceptionOptions` on the `setExceptionBreakpoints` request.
:param boolean supportsValueFormattingOptions: The debug adapter supports a `format` attribute on the `stackTrace`, `variables`, and `evaluate` requests.
:param boolean supportsExceptionInfoRequest: The debug adapter supports the `exceptionInfo` request.
:param boolean supportTerminateDebuggee: The debug adapter supports the `terminateDebuggee` attribute on the `disconnect` request.
:param boolean supportSuspendDebuggee: The debug adapter supports the `suspendDebuggee` attribute on the `disconnect` request.
:param boolean supportsDelayedStackTraceLoading: The debug adapter supports the delayed loading of parts of the stack, which requires that both the `startFrame` and `levels` arguments and the `totalFrames` result of the `stackTrace` request are supported.
:param boolean supportsLoadedSourcesRequest: The debug adapter supports the `loadedSources` request.
:param boolean supportsLogPoints: The debug adapter supports log points by interpreting the `logMessage` attribute of the `SourceBreakpoint`.
:param boolean supportsTerminateThreadsRequest: The debug adapter supports the `terminateThreads` request.
:param boolean supportsSetExpression: The debug adapter supports the `setExpression` request.
:param boolean supportsTerminateRequest: The debug adapter supports the `terminate` request.
:param boolean supportsDataBreakpoints: The debug adapter supports data breakpoints.
:param boolean supportsReadMemoryRequest: The debug adapter supports the `readMemory` request.
:param boolean supportsWriteMemoryRequest: The debug adapter supports the `writeMemory` request.
:param boolean supportsDisassembleRequest: The debug adapter supports the `disassemble` request.
:param boolean supportsCancelRequest: The debug adapter supports the `cancel` request.
:param boolean supportsBreakpointLocationsRequest: The debug adapter supports the `breakpointLocations` request.
:param boolean supportsClipboardContext: The debug adapter supports the `clipboard` context value in the `evaluate` request.
:param boolean supportsSteppingGranularity: The debug adapter supports stepping granularities (argument `granularity`) for the stepping requests.
:param boolean supportsInstructionBreakpoints: The debug adapter supports adding breakpoints based on instruction references.
:param boolean supportsExceptionFilterOptions: The debug adapter supports `filterOptions` as an argument on the `setExceptionBreakpoints` request.
:param boolean supportsSingleThreadExecutionRequests: The debug adapter supports the `singleThread` property on the execution requests (`continue`, `next`, `stepIn`, `stepOut`, `reverseContinue`, `stepBack`).

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, filter, label, description, default, supportsCondition, conditionDescription, update_ids_from_dap)

**Description:** :param string filter: The internal ID of the filter option. This value is passed to the `setExceptionBreakpoints` request.
:param string label: The name of the filter option. This is shown in the UI.
:param string description: A help text providing additional information about the exception filter. This string is typically shown as a hover and can be translated.
:param boolean default: Initial value of the filter option. If not specified a value false is assumed.
:param boolean supportsCondition: Controls whether a condition can be specified for this filter option. If false or missing, a condition can not be set.
:param string conditionDescription: A help text providing information about the condition. This string is shown as the placeholder text for a text box and can be translated.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, id, format, variables, sendTelemetry, showUser, url, urlLabel, update_ids_from_dap)

**Description:** :param integer id: Unique (within a debug adapter implementation) identifier for the message. The purpose of these error IDs is to help extension authors that have the requirement that every user visible error message needs a corresponding error number, so that users or customer support can find information about the specific error more easily.
:param string format: A format string for the message. Embedded variables have the form `{name}`.
If variable name starts with an underscore character, the variable does not contain user data (PII) and can be safely used for telemetry purposes.
:param MessageVariables variables: An object used as a dictionary for looking up the variables in the format string.
:param boolean sendTelemetry: If true send to telemetry.
:param boolean showUser: If true show user.
:param string url: A url where additional information about this message can be found.
:param string urlLabel: A label that is presented to the user as the UI for opening the url.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, id, name, path, isOptimized, isUserCode, version, symbolStatus, symbolFilePath, dateTimeStamp, addressRange, update_ids_from_dap)

**Description:** :param ['integer', 'string'] id: Unique identifier for the module.
:param string name: A name of the module.
:param string path: Logical full path to the module. The exact definition is implementation defined, but usually this would be a full path to the on-disk file for the module.
:param boolean isOptimized: True if the module is optimized.
:param boolean isUserCode: True if the module is considered 'user code' by a debugger that supports 'Just My Code'.
:param string version: Version of Module.
:param string symbolStatus: User-understandable description of if symbols were found for the module (ex: 'Symbols Loaded', 'Symbols not found', etc.)
:param string symbolFilePath: Logical full path to the symbol file. The exact definition is implementation defined.
:param string dateTimeStamp: Module created or modified, encoded as a RFC 3339 timestamp.
:param string addressRange: Address range covered by this module.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, attributeName, label, format, type, width, update_ids_from_dap)

**Description:** :param string attributeName: Name of the attribute rendered in this column.
:param string label: Header UI label of column.
:param string format: Format to use for the rendered values in this column. TBD how the format strings looks like.
:param string type: Datatype of values in this column. Defaults to `string` if not specified.
:param integer width: Width of this column in characters (hint only).

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, id, name, update_ids_from_dap)

**Description:** :param integer id: Unique identifier for the thread.
:param string name: The name of the thread.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, name, path, sourceReference, presentationHint, origin, sources, adapterData, checksums, update_ids_from_dap)

**Description:** :param string name: The short name of the source. Every source returned from the debug adapter has a name.
When sending a source to the debug adapter this name is optional.
:param string path: The path of the source to be shown in the UI.
It is only used to locate and load the content of the source if no `sourceReference` is specified (or its value is 0).
:param integer sourceReference: If the value > 0 the contents of the source must be retrieved through the `source` request (even if a path is specified).
Since a `sourceReference` is only valid for a session, it can not be used to persist a source.
The value should be less than or equal to 2147483647 (2^31-1).
:param string presentationHint: A hint for how to present the source in the UI.
A value of `deemphasize` can be used to indicate that the source is not available or that it is skipped on stepping.
:param string origin: The origin of this source. For example, 'internal module', 'inlined content from source map', etc.
:param array sources: A list of sources that are related to this source. These may be the source that generated this source.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] adapterData: Additional data that a debug adapter might want to loop through the client.
The client should leave the data intact and persist it across sessions. The client should not interpret the data.
:param array checksums: The checksums associated with this file.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, id, name, line, column, source, endLine, endColumn, canRestart, instructionPointerReference, moduleId, presentationHint, update_ids_from_dap)

**Description:** :param integer id: An identifier for the stack frame. It must be unique across all threads.
This id can be used to retrieve the scopes of the frame with the `scopes` request or to restart the execution of a stack frame.
:param string name: The name of the stack frame, typically a method name.
:param integer line: The line within the source of the frame. If the source attribute is missing or doesn't exist, `line` is 0 and should be ignored by the client.
:param integer column: Start position of the range covered by the stack frame. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based. If attribute `source` is missing or doesn't exist, `column` is 0 and should be ignored by the client.
:param Source source: The source of the frame.
:param integer endLine: The end line of the range covered by the stack frame.
:param integer endColumn: End position of the range covered by the stack frame. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based.
:param boolean canRestart: Indicates whether this frame can be restarted with the `restart` request. Clients should only use this if the debug adapter supports the `restart` request and the corresponding capability `supportsRestartRequest` is true. If a debug adapter has this capability, then `canRestart` defaults to `true` if the property is absent.
:param string instructionPointerReference: A memory reference for the current instruction pointer in this frame.
:param ['integer', 'string'] moduleId: The module associated with this frame, if any.
:param string presentationHint: A hint for how to present this frame in the UI.
A value of `label` can be used to indicate that the frame is an artificial frame that is used as a visual label or separator. A value of `subtle` can be used to change the appearance of a frame in a 'subtle' way.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, name, variablesReference, expensive, presentationHint, namedVariables, indexedVariables, source, line, column, endLine, endColumn, update_ids_from_dap)

**Description:** :param string name: Name of the scope such as 'Arguments', 'Locals', or 'Registers'. This string is shown in the UI as is and can be translated.
:param integer variablesReference: The variables of this scope can be retrieved by passing the value of `variablesReference` to the `variables` request as long as execution remains suspended. See 'Lifetime of Object References' in the Overview section for details.
:param boolean expensive: If true, the number of variables in this scope is large or expensive to retrieve.
:param string presentationHint: A hint for how to present this scope in the UI. If this attribute is missing, the scope is shown with a generic UI.
:param integer namedVariables: The number of named variables in this scope.
The client can use this information to present the variables in a paged UI and fetch them in chunks.
:param integer indexedVariables: The number of indexed variables in this scope.
The client can use this information to present the variables in a paged UI and fetch them in chunks.
:param Source source: The source for this scope.
:param integer line: The start line of the range covered by this scope.
:param integer column: Start position of the range covered by the scope. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based.
:param integer endLine: The end line of the range covered by this scope.
:param integer endColumn: End position of the range covered by the scope. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, name, value, variablesReference, type, presentationHint, evaluateName, namedVariables, indexedVariables, memoryReference, update_ids_from_dap)

**Description:** :param string name: The variable's name.
:param string value: The variable's value.
This can be a multi-line text, e.g. for a function the body of a function.
For structured variables (which do not have a simple value), it is recommended to provide a one-line representation of the structured object. This helps to identify the structured object in the collapsed state when its children are not yet visible.
An empty string can be used if no value should be shown in the UI.
:param integer variablesReference: If `variablesReference` is > 0, the variable is structured and its children can be retrieved by passing `variablesReference` to the `variables` request as long as execution remains suspended. See 'Lifetime of Object References' in the Overview section for details.
:param string type: The type of the variable's value. Typically shown in the UI when hovering over the value.
This attribute should only be returned by a debug adapter if the corresponding capability `supportsVariableType` is true.
:param VariablePresentationHint presentationHint: Properties of a variable that can be used to determine how to render the variable in the UI.
:param string evaluateName: The evaluatable name of this variable which can be passed to the `evaluate` request to fetch the variable's value.
:param integer namedVariables: The number of named child variables.
The client can use this information to present the children in a paged UI and fetch them in chunks.
:param integer indexedVariables: The number of indexed child variables.
The client can use this information to present the children in a paged UI and fetch them in chunks.
:param string memoryReference: A memory reference associated with this variable.
For pointer type variables, this is generally a reference to the memory address contained in the pointer.
For executable data, this reference may later be used in a `disassemble` request.
This attribute may be returned by a debug adapter if corresponding capability `supportsMemoryReferences` is true.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, kind, attributes, visibility, lazy, update_ids_from_dap)

**Description:** :param string kind: The kind of variable. Before introducing additional values, try to use the listed values.
:param array attributes: Set of attributes represented as an array of strings. Before introducing additional values, try to use the listed values.
:param string visibility: Visibility of variable. Before introducing additional values, try to use the listed values.
:param boolean lazy: If true, clients can present the variable with a UI that supports a specific gesture to trigger its evaluation.
This mechanism can be used for properties that require executing code when retrieving their value and where the code execution can be expensive and/or produce side-effects. A typical example are properties based on a getter function.
Please note that in addition to the `lazy` flag, the variable's `variablesReference` is expected to refer to a variable that will provide the value through another `variable` request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, line, column, endLine, endColumn, update_ids_from_dap)

**Description:** :param integer line: Start line of breakpoint location.
:param integer column: The start position of a breakpoint location. Position is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based.
:param integer endLine: The end line of breakpoint location if the location covers a range.
:param integer endColumn: The end position of a breakpoint location (if the location covers a range). Position is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, line, column, condition, hitCondition, logMessage, update_ids_from_dap)

**Description:** :param integer line: The source line of the breakpoint or logpoint.
:param integer column: Start position within source line of the breakpoint or logpoint. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based.
:param string condition: The expression for conditional breakpoints.
It is only honored by a debug adapter if the corresponding capability `supportsConditionalBreakpoints` is true.
:param string hitCondition: The expression that controls how many hits of the breakpoint are ignored.
The debug adapter is expected to interpret the expression as needed.
The attribute is only honored by a debug adapter if the corresponding capability `supportsHitConditionalBreakpoints` is true.
If both this property and `condition` are specified, `hitCondition` should be evaluated only if the `condition` is met, and the debug adapter should stop only if both conditions are met.
:param string logMessage: If this attribute exists and is non-empty, the debug adapter must not 'break' (stop)
but log the message instead. Expressions within `{}` are interpolated.
The attribute is only honored by a debug adapter if the corresponding capability `supportsLogPoints` is true.
If either `hitCondition` or `condition` is specified, then the message should only be logged if those conditions are met.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, name, condition, hitCondition, update_ids_from_dap)

**Description:** :param string name: The name of the function.
:param string condition: An expression for conditional breakpoints.
It is only honored by a debug adapter if the corresponding capability `supportsConditionalBreakpoints` is true.
:param string hitCondition: An expression that controls how many hits of the breakpoint are ignored.
The debug adapter is expected to interpret the expression as needed.
The attribute is only honored by a debug adapter if the corresponding capability `supportsHitConditionalBreakpoints` is true.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, update_ids_from_dap)

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, dataId, accessType, condition, hitCondition, update_ids_from_dap)

**Description:** :param string dataId: An id representing the data. This id is returned from the `dataBreakpointInfo` request.
:param DataBreakpointAccessType accessType: The access type of the data.
:param string condition: An expression for conditional breakpoints.
:param string hitCondition: An expression that controls how many hits of the breakpoint are ignored.
The debug adapter is expected to interpret the expression as needed.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, instructionReference, offset, condition, hitCondition, update_ids_from_dap)

**Description:** :param string instructionReference: The instruction reference of the breakpoint.
This should be a memory or instruction pointer reference from an `EvaluateResponse`, `Variable`, `StackFrame`, `GotoTarget`, or `Breakpoint`.
:param integer offset: The offset from the instruction reference in bytes.
This can be negative.
:param string condition: An expression for conditional breakpoints.
It is only honored by a debug adapter if the corresponding capability `supportsConditionalBreakpoints` is true.
:param string hitCondition: An expression that controls how many hits of the breakpoint are ignored.
The debug adapter is expected to interpret the expression as needed.
The attribute is only honored by a debug adapter if the corresponding capability `supportsHitConditionalBreakpoints` is true.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, verified, id, message, source, line, column, endLine, endColumn, instructionReference, offset, reason, update_ids_from_dap)

**Description:** :param boolean verified: If true, the breakpoint could be set (but not necessarily at the desired location).
:param integer id: The identifier for the breakpoint. It is needed if breakpoint events are used to update or remove breakpoints.
:param string message: A message about the state of the breakpoint.
This is shown to the user and can be used to explain why a breakpoint could not be verified.
:param Source source: The source where the breakpoint is located.
:param integer line: The start line of the actual range covered by the breakpoint.
:param integer column: Start position of the source range covered by the breakpoint. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based.
:param integer endLine: The end line of the actual range covered by the breakpoint.
:param integer endColumn: End position of the source range covered by the breakpoint. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based.
If no end line is given, then the end column is assumed to be in the start line.
:param string instructionReference: A memory reference to where the breakpoint is set.
:param integer offset: The offset from the instruction reference.
This can be negative.
:param string reason: A machine-readable explanation of why a breakpoint may not be verified. If a breakpoint is verified or a specific reason is not known, the adapter should omit this property. Possible values include:

- `pending`: Indicates a breakpoint might be verified in the future, but the adapter cannot verify it in the current state.
 - `failed`: Indicates a breakpoint was not able to be verified, and the adapter does not believe it can be verified without intervention.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, update_ids_from_dap)

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, id, label, line, column, endLine, endColumn, update_ids_from_dap)

**Description:** :param integer id: Unique identifier for a step-in target.
:param string label: The name of the step-in target (shown in the UI).
:param integer line: The line of the step-in target.
:param integer column: Start position of the range covered by the step in target. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based.
:param integer endLine: The end line of the range covered by the step-in target.
:param integer endColumn: End position of the range covered by the step in target. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, id, label, line, column, endLine, endColumn, instructionPointerReference, update_ids_from_dap)

**Description:** :param integer id: Unique identifier for a goto target. This is used in the `goto` request.
:param string label: The name of the goto target (shown in the UI).
:param integer line: The line of the goto target.
:param integer column: The column of the goto target.
:param integer endLine: The end line of the range covered by the goto target.
:param integer endColumn: The end column of the range covered by the goto target.
:param string instructionPointerReference: A memory reference for the instruction pointer value represented by this target.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, label, text, sortText, detail, type, start, length, selectionStart, selectionLength, update_ids_from_dap)

**Description:** :param string label: The label of this completion item. By default this is also the text that is inserted when selecting this completion.
:param string text: If text is returned and not an empty string, then it is inserted instead of the label.
:param string sortText: A string that should be used when comparing this item with other items. If not returned or an empty string, the `label` is used instead.
:param string detail: A human-readable string with additional information about this item, like type or symbol information.
:param CompletionItemType type: The item's type. Typically the client uses this information to render the item in the UI with an icon.
:param integer start: Start position (within the `text` attribute of the `completions` request) where the completion text is added. The position is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based. If the start position is omitted the text is added at the location specified by the `column` attribute of the `completions` request.
:param integer length: Length determines how many characters are overwritten by the completion text and it is measured in UTF-16 code units. If missing the value 0 is assumed which results in the completion text being inserted.
:param integer selectionStart: Determines the start of the new selection after the text has been inserted (or replaced). `selectionStart` is measured in UTF-16 code units and must be in the range 0 and length of the completion text. If omitted the selection starts at the end of the completion text.
:param integer selectionLength: Determines the length of the new selection after the text has been inserted (or replaced) and it is measured in UTF-16 code units. The selection can not extend beyond the bounds of the completion text. If omitted the length is assumed to be 0.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, update_ids_from_dap)

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, update_ids_from_dap)

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, algorithm, checksum, update_ids_from_dap)

**Description:** :param ChecksumAlgorithm algorithm: The algorithm used to calculate this checksum.
:param string checksum: Value of the checksum, encoded as a hexadecimal value.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, hex, update_ids_from_dap)

**Description:** :param boolean hex: Display the value in hex.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, hex, parameters, parameterTypes, parameterNames, parameterValues, line, module, includeAll, update_ids_from_dap)

**Description:** :param boolean hex: Display the value in hex.
:param boolean parameters: Displays parameters for the stack frame.
:param boolean parameterTypes: Displays the types of parameters for the stack frame.
:param boolean parameterNames: Displays the names of parameters for the stack frame.
:param boolean parameterValues: Displays the values of parameters for the stack frame.
:param boolean line: Displays the line number of the stack frame.
:param boolean module: Displays the module of the stack frame.
:param boolean includeAll: Includes all stack frames, including those the debug adapter might otherwise hide.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, filterId, condition, update_ids_from_dap)

**Description:** :param string filterId: ID of an exception filter returned by the `exceptionBreakpointFilters` capability.
:param string condition: An expression for conditional exceptions.
The exception breaks into the debugger if the result of the condition is true.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, breakMode, path, update_ids_from_dap)

**Description:** :param ExceptionBreakMode breakMode: Condition when a thrown exception should result in a break.
:param array path: A path that selects a single or multiple exceptions in a tree. If `path` is missing, the whole tree is selected.
By convention the first segment of the path is a category that is used to group exceptions in the UI.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, update_ids_from_dap)

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, names, negate, update_ids_from_dap)

**Description:** :param array names: Depending on the value of `negate` the names that should match or not match.
:param boolean negate: If false or missing this segment matches the names provided, otherwise it matches anything except the names provided.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, message, typeName, fullTypeName, evaluateName, stackTrace, innerException, update_ids_from_dap)

**Description:** :param string message: Message contained in the exception.
:param string typeName: Short type name of the exception object.
:param string fullTypeName: Fully-qualified type name of the exception object.
:param string evaluateName: An expression that can be evaluated in the current scope to obtain the exception object.
:param string stackTrace: Stack trace at the time the exception was thrown.
:param array innerException: Details of the exception contained by this exception, if any.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, address, instruction, instructionBytes, symbol, location, line, column, endLine, endColumn, presentationHint, update_ids_from_dap)

**Description:** :param string address: The address of the instruction. Treated as a hex value if prefixed with `0x`, or as a decimal value otherwise.
:param string instruction: Text representing the instruction and its operands, in an implementation-defined format.
:param string instructionBytes: Raw bytes representing the instruction and its operands, in an implementation-defined format.
:param string symbol: Name of the symbol that corresponds with the location of this instruction, if any.
:param Source location: Source location that corresponds to this instruction, if any.
Should always be set (if available) on the first instruction returned,
but can be omitted afterwards if this instruction maps to the same source file as the previous instruction.
:param integer line: The line within the source location that corresponds to this instruction, if any.
:param integer column: The column within the line that corresponds to this instruction, if any.
:param integer endLine: The end line of the range that corresponds to this instruction, if any.
:param integer endColumn: The end column of the range that corresponds to this instruction, if any.
:param string presentationHint: A hint for how to present the instruction in the UI.

A value of `invalid` may be used to indicate this instruction is 'filler' and cannot be reached by the program. For example, unreadable memory addresses may be presented is 'invalid.'

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, update_ids_from_dap)

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param SetDebuggerPropertyArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, ideOS, dontTraceStartPatterns, dontTraceEndPatterns, skipSuspendOnBreakpointException, skipPrintBreakpointException, multiThreadsSingleNotification, update_ids_from_dap)

**Description:** :param ['string'] ideOS: OS where the ide is running. Supported values [Windows, Linux]
:param ['array'] dontTraceStartPatterns: Patterns to match with the start of the file paths. Matching paths will be added to a list of file where trace is ignored.
:param ['array'] dontTraceEndPatterns: Patterns to match with the end of the file paths. Matching paths will be added to a list of file where trace is ignored.
:param ['array'] skipSuspendOnBreakpointException: List of exceptions that should be skipped when doing condition evaluations.
:param ['array'] skipPrintBreakpointException: List of exceptions that should skip printing to stderr when doing condition evaluations.
:param ['boolean'] multiThreadsSingleNotification: If false then a notification is generated for each thread event. If true a single event is gnenerated, and all threads follow that behavior.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, seq, body, update_ids_from_dap)

**Description:** :param string type:
:param string event:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Event-specific information.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param SetPydevdSourceMapArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, source, pydevdSourceMaps, update_ids_from_dap)

**Description:** :param Source source: The source location of the PydevdSourceMap; 'source.path' must be specified (e.g.: for an ipython notebook this could be something as /home/notebook/note.py).
:param array pydevdSourceMaps: The PydevdSourceMaps to be set to the given source (provide an empty array to clear the source mappings for a given path).

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, seq, message, body, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Contains request result if success is true and error details if success is false.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, line, endLine, runtimeSource, runtimeLine, update_ids_from_dap)

**Description:** :param integer line: The local line to which the mapping should map to (e.g.: for an ipython notebook this would be the first line of the cell in the file).
:param integer endLine: The end line.
:param Source runtimeSource: The path that the user has remotely -- 'source.path' must be specified (e.g.: for an ipython notebook this could be something as '<ipython-input-1-4561234>')
:param integer runtimeLine: The remote line to which the mapping should map to (e.g.: for an ipython notebook this would be always 1 as it'd map the start of the cell).

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, seq, arguments, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param PydevdSystemInfoArguments arguments:

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, update_ids_from_dap)

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param PydevdSystemInfoResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, version, implementation, update_ids_from_dap)

**Description:** :param string version: Python version as a string in semver format: <major>.<minor>.<micro><releaselevel><serial>.
:param PydevdPythonImplementationInfo implementation: Python version as a string in this format <major>.<minor>.<micro><releaselevel><serial>.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, name, version, description, update_ids_from_dap)

**Description:** :param string name: Python implementation name.
:param string version: Python version as a string in semver format: <major>.<minor>.<micro><releaselevel><serial>.
:param string description: Optional description for this python implementation.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, name, update_ids_from_dap)

**Description:** :param string name: Name of the platform as returned by 'sys.platform'.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, pid, ppid, executable, bitness, update_ids_from_dap)

**Description:** :param integer pid: Process ID for the current process.
:param integer ppid: Parent Process ID for the current process.
:param string executable: Path to the executable as returned by 'sys.executable'.
:param integer bitness: Integer value indicating the bitness of the current process.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, usingCython, usingFrameEval, update_ids_from_dap)

**Description:** :param boolean usingCython: Specifies whether the cython native module is being used.
:param boolean usingFrameEval: Specifies whether the frame eval native module is being used.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, arguments, seq, update_ids_from_dap)

**Description:** :param string type:
:param string command:
:param PydevdAuthorizeArguments arguments:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, debugServerAccessToken, update_ids_from_dap)

**Description:** :param string debugServerAccessToken: The access token to access the debug server.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, request_seq, success, command, body, seq, message, update_ids_from_dap)

**Description:** :param string type:
:param integer request_seq: Sequence number of the corresponding request.
:param boolean success: Outcome of the request.
If true, the request was successful and the `body` attribute may contain the result of the request.
If the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).
:param string command: The command requested.
:param PydevdAuthorizeResponseBody body:
:param integer seq: Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.
:param string message: Contains the raw error in short form if `success` is false.
This raw error might be interpreted by the client and is not shown in the UI.
Some predefined values exist.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, error, update_ids_from_dap)

**Description:** :param Message error: A structured error message.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, reason, description, threadId, preserveFocusHint, text, allThreadsStopped, hitBreakpointIds, update_ids_from_dap)

**Description:** :param string reason: The reason for the event.
For backward compatibility this string is shown in the UI if the `description` attribute is missing (but it must not be translated).
:param string description: The full reason for the event, e.g. 'Paused on exception'. This string is shown in the UI as is and can be translated.
:param integer threadId: The thread which was stopped.
:param boolean preserveFocusHint: A value of true hints to the client that this event should not change the focus.
:param string text: Additional information. E.g. if reason is `exception`, text contains the exception name. This string is shown in the UI.
:param boolean allThreadsStopped: If `allThreadsStopped` is true, a debug adapter can announce that all threads have stopped.
- The client should use this information to enable that all threads can be expanded to access their stacktraces.
- If the attribute is missing or false, only the thread with the given `threadId` can be expanded.
:param array hitBreakpointIds: Ids of the breakpoints that triggered the event. In most cases there is only a single breakpoint but here are some examples for multiple breakpoints:
- Different types of breakpoints map to the same location.
- Multiple source breakpoints get collapsed to the same instruction by the compiler/runtime.
- Multiple function breakpoints with different function names map to the same location.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, threadId, allThreadsContinued, update_ids_from_dap)

**Description:** :param integer threadId: The thread which was continued.
:param boolean allThreadsContinued: If `allThreadsContinued` is true, a debug adapter can announce that all threads have continued.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, exitCode, update_ids_from_dap)

**Description:** :param integer exitCode: The exit code returned from the debuggee.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, restart, update_ids_from_dap)

**Description:** :param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] restart: A debug adapter may set `restart` to true (or to an arbitrary object) to request that the client restarts the session.
The value is not interpreted by the client and passed unmodified as an attribute `__restart` to the `launch` and `attach` requests.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, reason, threadId, update_ids_from_dap)

**Description:** :param string reason: The reason for the event.
:param integer threadId: The identifier of the thread.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, output, category, group, variablesReference, source, line, column, data, update_ids_from_dap)

**Description:** :param string output: The output to report.
:param string category: The output category. If not specified or if the category is not understood by the client, `console` is assumed.
:param string group: Support for keeping an output log organized by grouping related messages.
:param integer variablesReference: If an attribute `variablesReference` exists and its value is > 0, the output contains objects which can be retrieved by passing `variablesReference` to the `variables` request as long as execution remains suspended. See 'Lifetime of Object References' in the Overview section for details.
:param Source source: The source location where the output was produced.
:param integer line: The source location's line where the output was produced.
:param integer column: The position in `line` where the output was produced. It is measured in UTF-16 code units and the client capability `columnsStartAt1` determines whether it is 0- or 1-based.
:param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] data: Additional data to report. For the `telemetry` category the data is sent to telemetry, for the other categories the data is shown in JSON format.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, reason, breakpoint, update_ids_from_dap)

**Description:** :param string reason: The reason for the event.
:param Breakpoint breakpoint: The `id` attribute is used to find the target breakpoint, the other attributes are used as the new values.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, reason, module, update_ids_from_dap)

**Description:** :param string reason: The reason for the event.
:param Module module: The new, changed, or removed module. In case of `removed` only the module id is used.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, reason, source, update_ids_from_dap)

**Description:** :param string reason: The reason for the event.
:param Source source: The new, changed, or removed source.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, name, systemProcessId, isLocalProcess, startMethod, pointerSize, update_ids_from_dap)

**Description:** :param string name: The logical name of the process. This is usually the full path to process's executable file. Example: /home/example/myproj/program.js.
:param integer systemProcessId: The system process id of the debugged process. This property is missing for non-system processes.
:param boolean isLocalProcess: If true, the process is running on the same computer as the debug adapter.
:param string startMethod: Describes how the debug engine started debugging this process.
:param integer pointerSize: The size of a pointer or address for this process, in bits. This value may be used by clients when formatting addresses for display.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, capabilities, update_ids_from_dap)

**Description:** :param Capabilities capabilities: The set of updated capabilities.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, progressId, title, requestId, cancellable, message, percentage, update_ids_from_dap)

**Description:** :param string progressId: An ID that can be used in subsequent `progressUpdate` and `progressEnd` events to make them refer to the same progress reporting.
IDs must be unique within a debug session.
:param string title: Short title of the progress reporting. Shown in the UI to describe the long running operation.
:param integer requestId: The request ID that this progress report is related to. If specified a debug adapter is expected to emit progress events for the long running request until the request has been either completed or cancelled.
If the request ID is omitted, the progress report is assumed to be related to some general activity of the debug adapter.
:param boolean cancellable: If true, the request that reports progress may be cancelled with a `cancel` request.
So this property basically controls whether the client should use UX that supports cancellation.
Clients that don't support cancellation are allowed to ignore the setting.
:param string message: More detailed progress message.
:param number percentage: Progress percentage to display (value range: 0 to 100). If omitted no percentage is shown.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, progressId, message, percentage, update_ids_from_dap)

**Description:** :param string progressId: The ID that was introduced in the initial `progressStart` event.
:param string message: More detailed progress message. If omitted, the previous message (if any) is used.
:param number percentage: Progress percentage to display (value range: 0 to 100). If omitted no percentage is shown.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, progressId, message, update_ids_from_dap)

**Description:** :param string progressId: The ID that was introduced in the initial `ProgressStartEvent`.
:param string message: More detailed progress message. If omitted, the previous message (if any) is used.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, areas, threadId, stackFrameId, update_ids_from_dap)

**Description:** :param array areas: Set of logical areas that got invalidated. This property has a hint characteristic: a client can only be expected to make a 'best effort' in honoring the areas but there are no guarantees. If this property is missing, empty, or if values are not understood, the client should assume a single value `all`.
:param integer threadId: If specified, the client only needs to refetch data related to this thread.
:param integer stackFrameId: If specified, the client only needs to refetch data related to this stack frame (and the `threadId` is ignored).

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, memoryReference, offset, count, update_ids_from_dap)

**Description:** :param string memoryReference: Memory reference of a memory range that has been updated.
:param integer offset: Starting offset in bytes where memory has been updated. Can be negative.
:param integer count: Number of bytes updated.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, update_ids_from_dap)

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, processId, shellProcessId, update_ids_from_dap)

**Description:** :param integer processId: The process ID. The value should be less than or equal to 2147483647 (2^31-1).
:param integer shellProcessId: The process ID of the terminal shell. The value should be less than or equal to 2147483647 (2^31-1).

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, update_ids_from_dap)

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, breakpoints, update_ids_from_dap)

**Description:** :param array breakpoints: Sorted set of possible breakpoint locations.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, breakpoints, update_ids_from_dap)

**Description:** :param array breakpoints: Information about the breakpoints.
The array elements are in the same order as the elements of the `breakpoints` (or the deprecated `lines`) array in the arguments.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, breakpoints, update_ids_from_dap)

**Description:** :param array breakpoints: Information about the breakpoints. The array elements correspond to the elements of the `breakpoints` array.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, breakpoints, update_ids_from_dap)

**Description:** :param array breakpoints: Information about the exception breakpoints or filters.
The breakpoints returned are in the same order as the elements of the `filters`, `filterOptions`, `exceptionOptions` arrays in the arguments. If both `filters` and `filterOptions` are given, the returned array must start with `filters` information first, followed by `filterOptions` information.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, dataId, description, accessTypes, canPersist, update_ids_from_dap)

**Description:** :param ['string', 'null'] dataId: An identifier for the data on which a data breakpoint can be registered with the `setDataBreakpoints` request or null if no data breakpoint is available. If a `variablesReference` or `frameId` is passed, the `dataId` is valid in the current suspended state, otherwise it's valid indefinitely. See 'Lifetime of Object References' in the Overview section for details. Breakpoints set using the `dataId` in the `setDataBreakpoints` request may outlive the lifetime of the associated `dataId`.
:param string description: UI string that describes on what data the breakpoint is set on or why a data breakpoint is not available.
:param array accessTypes: Attribute lists the available access types for a potential data breakpoint. A UI client could surface this information.
:param boolean canPersist: Attribute indicates that a potential data breakpoint could be persisted across sessions.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, breakpoints, update_ids_from_dap)

**Description:** :param array breakpoints: Information about the data breakpoints. The array elements correspond to the elements of the input argument `breakpoints` array.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, breakpoints, update_ids_from_dap)

**Description:** :param array breakpoints: Information about the breakpoints. The array elements correspond to the elements of the `breakpoints` array.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, allThreadsContinued, update_ids_from_dap)

**Description:** :param boolean allThreadsContinued: The value true (or a missing property) signals to the client that all threads have been resumed. The value false indicates that not all threads were resumed.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, stackFrames, totalFrames, update_ids_from_dap)

**Description:** :param array stackFrames: The frames of the stack frame. If the array has length zero, there are no stack frames available.
This means that there is no location information available.
:param integer totalFrames: The total number of frames available in the stack. If omitted or if `totalFrames` is larger than the available frames, a client is expected to request frames until a request returns less frames than requested (which indicates the end of the stack). Returning monotonically increasing `totalFrames` values for subsequent requests can be used to enforce paging in the client.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, scopes, update_ids_from_dap)

**Description:** :param array scopes: The scopes of the stack frame. If the array has length zero, there are no scopes available.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, variables, update_ids_from_dap)

**Description:** :param array variables: All (or a range) of variables for the given variable reference.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, value, type, variablesReference, namedVariables, indexedVariables, memoryReference, update_ids_from_dap)

**Description:** :param string value: The new value of the variable.
:param string type: The type of the new value. Typically shown in the UI when hovering over the value.
:param integer variablesReference: If `variablesReference` is > 0, the new value is structured and its children can be retrieved by passing `variablesReference` to the `variables` request as long as execution remains suspended. See 'Lifetime of Object References' in the Overview section for details.
:param integer namedVariables: The number of named child variables.
The client can use this information to present the variables in a paged UI and fetch them in chunks.
The value should be less than or equal to 2147483647 (2^31-1).
:param integer indexedVariables: The number of indexed child variables.
The client can use this information to present the variables in a paged UI and fetch them in chunks.
The value should be less than or equal to 2147483647 (2^31-1).
:param string memoryReference: A memory reference to a location appropriate for this result.
For pointer type eval results, this is generally a reference to the memory address contained in the pointer.
This attribute may be returned by a debug adapter if corresponding capability `supportsMemoryReferences` is true.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, content, mimeType, update_ids_from_dap)

**Description:** :param string content: Content of the source reference.
:param string mimeType: Content type (MIME type) of the source.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, threads, update_ids_from_dap)

**Description:** :param array threads: All threads.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, modules, totalModules, update_ids_from_dap)

**Description:** :param array modules: All modules or range of modules.
:param integer totalModules: The total number of modules available.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, sources, update_ids_from_dap)

**Description:** :param array sources: Set of loaded sources.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, result, variablesReference, type, presentationHint, namedVariables, indexedVariables, memoryReference, update_ids_from_dap)

**Description:** :param string result: The result of the evaluate request.
:param integer variablesReference: If `variablesReference` is > 0, the evaluate result is structured and its children can be retrieved by passing `variablesReference` to the `variables` request as long as execution remains suspended. See 'Lifetime of Object References' in the Overview section for details.
:param string type: The type of the evaluate result.
This attribute should only be returned by a debug adapter if the corresponding capability `supportsVariableType` is true.
:param VariablePresentationHint presentationHint: Properties of an evaluate result that can be used to determine how to render the result in the UI.
:param integer namedVariables: The number of named child variables.
The client can use this information to present the variables in a paged UI and fetch them in chunks.
The value should be less than or equal to 2147483647 (2^31-1).
:param integer indexedVariables: The number of indexed child variables.
The client can use this information to present the variables in a paged UI and fetch them in chunks.
The value should be less than or equal to 2147483647 (2^31-1).
:param string memoryReference: A memory reference to a location appropriate for this result.
For pointer type eval results, this is generally a reference to the memory address contained in the pointer.
This attribute may be returned by a debug adapter if corresponding capability `supportsMemoryReferences` is true.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, value, type, presentationHint, variablesReference, namedVariables, indexedVariables, memoryReference, update_ids_from_dap)

**Description:** :param string value: The new value of the expression.
:param string type: The type of the value.
This attribute should only be returned by a debug adapter if the corresponding capability `supportsVariableType` is true.
:param VariablePresentationHint presentationHint: Properties of a value that can be used to determine how to render the result in the UI.
:param integer variablesReference: If `variablesReference` is > 0, the evaluate result is structured and its children can be retrieved by passing `variablesReference` to the `variables` request as long as execution remains suspended. See 'Lifetime of Object References' in the Overview section for details.
:param integer namedVariables: The number of named child variables.
The client can use this information to present the variables in a paged UI and fetch them in chunks.
The value should be less than or equal to 2147483647 (2^31-1).
:param integer indexedVariables: The number of indexed child variables.
The client can use this information to present the variables in a paged UI and fetch them in chunks.
The value should be less than or equal to 2147483647 (2^31-1).
:param string memoryReference: A memory reference to a location appropriate for this result.
For pointer type eval results, this is generally a reference to the memory address contained in the pointer.
This attribute may be returned by a debug adapter if corresponding capability `supportsMemoryReferences` is true.

### Function: update_dict_ids_from_dap(cls, dct)

### Function: to_dict(self, update_ids_to_dap)

### Function: update_dict_ids_to_dap(cls, dct)

### Function: __init__(self, targets, update_ids_from_dap)

**Description:** :param array targets: The possible step-in targets of the specified source location.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, targets, update_ids_from_dap)

**Description:** :param array targets: The possible goto targets of the specified location.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, targets, update_ids_from_dap)

**Description:** :param array targets: The possible completions for .

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, exceptionId, breakMode, description, details, update_ids_from_dap)

**Description:** :param string exceptionId: ID of the exception that was thrown.
:param ExceptionBreakMode breakMode: Mode that caused the exception notification to be raised.
:param string description: Descriptive text for the exception.
:param ExceptionDetails details: Detailed information about the exception.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, address, unreadableBytes, data, update_ids_from_dap)

**Description:** :param string address: The address of the first byte of data returned.
Treated as a hex value if prefixed with `0x`, or as a decimal value otherwise.
:param integer unreadableBytes: The number of unreadable bytes encountered after the last successfully read byte.
This can be used to determine the number of bytes that should be skipped before a subsequent `readMemory` request succeeds.
:param string data: The bytes read from memory, encoded using base64. If the decoded length of `data` is less than the requested `count` in the original `readMemory` request, and `unreadableBytes` is zero or omitted, then the client should assume it's reached the end of readable memory.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, offset, bytesWritten, update_ids_from_dap)

**Description:** :param integer offset: Property that should be returned when `allowPartial` is true to indicate the offset of the first byte of data successfully written. Can be negative.
:param integer bytesWritten: Property that should be returned when `allowPartial` is true to indicate the number of bytes starting from address that were successfully written.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, instructions, update_ids_from_dap)

**Description:** :param array instructions: The list of disassembled instructions.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, update_ids_from_dap)

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, python, platform, process, pydevd, update_ids_from_dap)

**Description:** :param PydevdPythonInfo python: Information about the python version running in the current process.
:param PydevdPlatformInfo platform: Information about the plarforn on which the current process is running.
:param PydevdProcessInfo process: Information about the current process.
:param PydevdInfo pydevd: Information about pydevd.

### Function: to_dict(self, update_ids_to_dap)

### Function: __init__(self, clientAccessToken, update_ids_from_dap)

**Description:** :param string clientAccessToken: The access token to access the client (i.e.: usually the IDE).

### Function: to_dict(self, update_ids_to_dap)
