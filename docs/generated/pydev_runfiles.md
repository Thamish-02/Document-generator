## AI Summary

A file named pydev_runfiles.py.


## Class: Configuration

### Function: parse_cmdline(argv)

**Description:** Parses command line and returns test directories, verbosity, test filter and test suites

usage:
    runfiles.py  -v|--verbosity <level>  -t|--tests <Test.test1,Test2>  dirs|files

Multiprocessing options:
jobs=number (with the number of jobs to be used to run the tests)
split_jobs='module'|'tests'
    if == module, a given job will always receive all the tests from a module
    if == tests, the tests will be split independently of their originating module (default)

--exclude_files  = comma-separated list of patterns with files to exclude (fnmatch style)
--include_files = comma-separated list of patterns with files to include (fnmatch style)
--exclude_tests = comma-separated list of patterns with test names to exclude (fnmatch style)

Note: if --tests is given, --exclude_files, --include_files and --exclude_tests are ignored!

## Class: PydevTestRunner

**Description:** finds and runs a file or directory of files as a unit test

### Function: get_django_test_suite_runner()

### Function: main(configuration)

### Function: __init__(self, files_or_dirs, verbosity, include_tests, tests, port, files_to_tests, jobs, split_jobs, coverage_output_dir, coverage_include, coverage_output_file, exclude_files, exclude_tests, include_files, django)

### Function: __str__(self)

### Function: __init__(self, configuration)

### Function: __adjust_path(self)

**Description:** add the current file or directory to the python path

### Function: __is_valid_py_file(self, fname)

**Description:** tests that a particular file contains the proper file extension
and is not in the list of files to exclude

### Function: __unixify(self, s)

**Description:** stupid windows. converts the backslash to forwardslash for consistency

### Function: __importify(self, s, dir)

**Description:** turns directory separators into dots and removes the ".py*" extension
so the string can be used as import statement

### Function: __add_files(self, pyfiles, root, files)

**Description:** if files match, appends them to pyfiles. used by os.path.walk fcn

### Function: find_import_files(self)

**Description:** return a list of files to import

### Function: __get_module_from_str(self, modname, print_exception, pyfile)

**Description:** Import the module in the given import path.
* Returns the "final" module, so importing "coilib40.subject.visu"
returns the "visu" module, not the "coilib40" as returned by __import__

### Function: remove_duplicates_keeping_order(self, seq)

### Function: find_modules_from_files(self, pyfiles)

**Description:** returns a list of modules given a list of files

## Class: GetTestCaseNames

**Description:** Yes, we need a class for that (cannot use outer context on jython 2.1)

### Function: _decorate_test_suite(self, suite, pyfile, module_name)

### Function: find_tests_from_modules(self, file_and_modules_and_module_name)

**Description:** returns the unittests given a list of modules

### Function: filter_tests(self, test_objs, internal_call)

**Description:** based on a filter name, only return those tests that have
the test case names that match

### Function: iter_tests(self, test_objs)

### Function: list_test_names(self, test_objs)

### Function: __match_tests(self, tests, test_case, test_method_name)

### Function: __match(self, filter_list, name)

**Description:** returns whether a test name matches the test filter

### Function: run_tests(self, handle_coverage)

**Description:** runs all tests

### Function: __init__(self, accepted_classes, accepted_methods)

### Function: __call__(self, testCaseClass)

**Description:** Return a sorted sequence of method names found within testCaseClass

### Function: run_tests()

## Class: MyDjangoTestSuiteRunner

### Function: __init__(self, on_run_suite)

### Function: build_suite(self)

### Function: suite_result(self)

### Function: run_suite(self)

## Class: MyDjangoTestSuiteRunner

### Function: __init__(self, on_run_suite)

### Function: build_suite(self)

### Function: suite_result(self)

### Function: run_suite(self)

## Class: DjangoTestSuiteRunner

### Function: __init__(self)

### Function: run_tests(self)
