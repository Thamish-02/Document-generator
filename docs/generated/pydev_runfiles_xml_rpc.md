## AI Summary

A file named pydev_runfiles_xml_rpc.py.


## Class: _ServerHolder

**Description:** Helper so that we don't have to use a global here.

### Function: set_server(server)

## Class: ParallelNotification

## Class: KillServer

## Class: ServerFacade

## Class: ServerComm

### Function: initialize_server(port, daemon)

### Function: notifyTestsCollected(tests_count)

### Function: notifyStartTest(file, test)

**Description:** @param file: the tests file (c:/temp/test.py)
@param test: the test ran (i.e.: TestCase.test1)

### Function: _encode_if_needed(obj)

### Function: notifyTest(cond, captured_output, error_contents, file, test, time)

**Description:** @param cond: ok, fail, error
@param captured_output: output captured from stdout
@param captured_output: output captured from stderr
@param file: the tests file (c:/temp/test.py)
@param test: the test ran (i.e.: TestCase.test1)
@param time: float with the number of seconds elapsed

### Function: notifyTestRunFinished(total_time)

### Function: force_server_kill()

### Function: __init__(self, method, args)

### Function: to_tuple(self)

### Function: __init__(self, notifications_queue)

### Function: notifyTestsCollected(self)

### Function: notifyConnected(self)

### Function: notifyTestRunFinished(self)

### Function: notifyStartTest(self)

### Function: notifyTest(self)

### Function: __init__(self, notifications_queue, port, daemon)

### Function: run(self)
