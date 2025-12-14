## AI Summary

A file named conftest.py.


### Function: pytest_configure(config)

### Function: pytest_addoption(parser)

### Function: pytest_sessionstart(session)

### Function: pytest_terminal_summary(terminalreporter, exitstatus, config)

### Function: pytest_itemcollected(item)

**Description:** Check FPU precision mode was not changed during test collection.

The clumsy way we do it here is mainly necessary because numpy
still uses yield tests, which can execute code at test collection
time.

### Function: check_fpu_mode(request)

**Description:** Check FPU precision mode was not changed during the test.

### Function: add_np(doctest_namespace)

### Function: env_setup(monkeypatch)

### Function: random_string_list()

### Function: coerce(request)

### Function: na_object(request)

### Function: dtype(na_object, coerce)

### Function: warnings_errors_and_rng(test)

**Description:** Filter out the wall of DeprecationWarnings.
        
