## AI Summary

A file named test_iplib.py.


### Function: test_reset()

**Description:** reset must clear most namespaces.

### Function: doctest_tb_plain()

**Description:** In [18]: xmode plain
Exception reporting mode: Plain

In [19]: run simpleerr.py
Traceback (most recent call last):
  File ...:...
    bar(mode)
  File ...:... in bar
    div0()
  File ...:... in div0
    x/y
ZeroDivisionError: ...

### Function: doctest_tb_context()

**Description:** In [3]: xmode context
Exception reporting mode: Context

In [4]: run simpleerr.py
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
<BLANKLINE>
...
     30     except IndexError:
     31         mode = 'div'
---> 33     bar(mode)
<BLANKLINE>
... in bar(mode)
     15     "bar"
     16     if mode=='div':
---> 17         div0()
     18     elif mode=='exit':
     19         try:
<BLANKLINE>
... in div0()
      6     x = 1
      7     y = 0
----> 8     x/y
<BLANKLINE>
ZeroDivisionError: ...

### Function: doctest_tb_verbose()

**Description:** In [5]: xmode verbose
Exception reporting mode: Verbose

In [6]: run simpleerr.py
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
<BLANKLINE>
...
     30     except IndexError:
     31         mode = 'div'
---> 33     bar(mode)
        mode = 'div'
<BLANKLINE>
... in bar(mode='div')
     15     "bar"
     16     if mode=='div':
---> 17         div0()
     18     elif mode=='exit':
     19         try:
<BLANKLINE>
... in div0()
      6     x = 1
      7     y = 0
----> 8     x/y
        x = 1
        y = 0
<BLANKLINE>
ZeroDivisionError: ...

### Function: doctest_tb_sysexit()

**Description:** In [17]: %xmode plain
Exception reporting mode: Plain

In [18]: %run simpleerr.py exit
An exception has occurred, use %tb to see the full traceback.
SystemExit: (1, 'Mode = exit')

In [19]: %run simpleerr.py exit 2
An exception has occurred, use %tb to see the full traceback.
SystemExit: (2, 'Mode = exit')

In [20]: %tb
Traceback (most recent call last):
  File ...:... in execfile
    exec(compiler(f.read(), fname, "exec"), glob, loc)
  File ...:...
    bar(mode)
  File ...:... in bar
    sysexit(stat, mode)
  File ...:... in sysexit
    raise SystemExit(stat, f"Mode = {mode}")
SystemExit: (2, 'Mode = exit')

In [21]: %xmode context
Exception reporting mode: Context

In [22]: %tb
---------------------------------------------------------------------------
SystemExit                                Traceback (most recent call last)
File ..., in execfile(fname, glob, loc, compiler)
     ... with open(fname, "rb") as f:
     ...     compiler = compiler or compile
---> ...     exec(compiler(f.read(), fname, "exec"), glob, loc)
...
     30     except IndexError:
     31         mode = 'div'
---> 33     bar(mode)
<BLANKLINE>
...bar(mode)
     21         except:
     22             stat = 1
---> 23         sysexit(stat, mode)
     24     else:
     25         raise ValueError('Unknown mode')
<BLANKLINE>
...sysexit(stat, mode)
     10 def sysexit(stat, mode):
---> 11     raise SystemExit(stat, f"Mode = {mode}")
<BLANKLINE>
SystemExit: (2, 'Mode = exit')

### Function: test_run_cell()

### Function: test_db()

**Description:** Test the internal database used for variable persistence.

### Function: doctest_tb_sysexit_verbose_stack_data_05()

**Description:** In [18]: %run simpleerr.py exit
An exception has occurred, use %tb to see the full traceback.
SystemExit: (1, 'Mode = exit')

In [19]: %run simpleerr.py exit 2
An exception has occurred, use %tb to see the full traceback.
SystemExit: (2, 'Mode = exit')

In [23]: %xmode verbose
Exception reporting mode: Verbose

In [24]: %tb
---------------------------------------------------------------------------
SystemExit                                Traceback (most recent call last)
<BLANKLINE>
...
    30     except IndexError:
    31         mode = 'div'
---> 33     bar(mode)
        mode = 'exit'
<BLANKLINE>
... in bar(mode='exit')
    ...     except:
    ...         stat = 1
---> ...     sysexit(stat, mode)
        mode = 'exit'
        stat = 2
    ...     else:
    ...         raise ValueError('Unknown mode')
<BLANKLINE>
... in sysexit(stat=2, mode='exit')
    10 def sysexit(stat, mode):
---> 11     raise SystemExit(stat, f"Mode = {mode}")
        stat = 2
<BLANKLINE>
SystemExit: (2, 'Mode = exit')
