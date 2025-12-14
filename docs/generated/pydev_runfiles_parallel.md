## AI Summary

A file named pydev_runfiles_parallel.py.


### Function: flatten_test_suite(test_suite, ret)

### Function: execute_tests_in_parallel(tests, jobs, split, verbosity, coverage_files, coverage_include)

**Description:** @param tests: list(PydevTestSuite)
    A list with the suites to be run

@param split: str
    Either 'module' or the number of tests that should be run in each batch

@param coverage_files: list(file)
    A list with the files that should be used for giving coverage information (if empty, coverage information
    should not be gathered).

@param coverage_include: str
    The pattern that should be included in the coverage.

@return: bool
    Returns True if the tests were actually executed in parallel. If the tests were not executed because only 1
    should be used (e.g.: 2 jobs were requested for running 1 test), False will be returned and no tests will be
    run.

    It may also return False if in debug mode (in which case, multi-processes are not accepted)

## Class: CommunicationThread

## Class: ClientThread

### Function: __init__(self, tests_queue)

### Function: GetTestsToRun(self, job_id)

**Description:** @param job_id:

@return: list(str)
    Each entry is a string in the format: filename|Test.testName

### Function: notifyCommands(self, job_id, commands)

### Function: notifyStartTest(self, job_id)

### Function: notifyTest(self, job_id)

### Function: shutdown(self)

### Function: run(self)

### Function: __init__(self, job_id, port, verbosity, coverage_output_file, coverage_include)

### Function: _reader_thread(self, pipe, target)

### Function: run(self)
