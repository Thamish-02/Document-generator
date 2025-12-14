## AI Summary

A file named test_refs.py.


### Function: test_trivial()

**Description:** A trivial passing test.

### Function: doctest_run()

**Description:** Test running a trivial script.

In [13]: run simplevars.py
x is: 1

### Function: doctest_runvars()

**Description:** Test that variables defined in scripts get loaded correctly via %run.

In [13]: run simplevars.py
x is: 1

In [14]: x
Out[14]: 1

### Function: doctest_ivars()

**Description:** Test that variables defined interactively are picked up.
In [5]: zz=1

In [6]: zz
Out[6]: 1
