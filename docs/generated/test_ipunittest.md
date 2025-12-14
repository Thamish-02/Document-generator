## AI Summary

A file named test_ipunittest.py.


### Function: simple_dt()

**Description:** >>> print(1+1)
2

### Function: ipdt_flush()

**Description:** In [20]: print(1)
1

In [26]: for i in range(4):
   ....:     print(i)
   ....:     
   ....: 
0
1
2
3

In [27]: 3+4
Out[27]: 7

### Function: ipdt_indented_test()

**Description:** In [20]: print(1)
1

In [26]: for i in range(4):
   ....:     print(i)
   ....:     
   ....: 
0
1
2
3

In [27]: 3+4
Out[27]: 7

## Class: Foo

**Description:** For methods, the normal decorator doesn't work.

But rewriting the docstring with ip2py does, *but only if using nose
--with-doctest*.  Do we want to have that as a dependency?

### Function: ipdt_method(self)

**Description:** In [20]: print(1)
1

In [26]: for i in range(4):
   ....:     print(i)
   ....:     
   ....: 
0
1
2
3

In [27]: 3+4
Out[27]: 7

### Function: normaldt_method(self)

**Description:** >>> print(1+1)
2
