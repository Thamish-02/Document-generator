## AI Summary

A file named capi_maps.py.


### Function: load_f2cmap_file(f2cmap_file)

### Function: getctype(var)

**Description:** Determines C type

### Function: f2cexpr(expr)

**Description:** Rewrite Fortran expression as f2py supported C expression.

Due to the lack of a proper expression parser in f2py, this
function uses a heuristic approach that assumes that Fortran
arithmetic expressions are valid C arithmetic expressions when
mapping Fortran function calls to the corresponding C function/CPP
macros calls.

### Function: getstrlength(var)

### Function: getarrdims(a, var, verbose)

### Function: getpydocsign(a, var)

### Function: getarrdocsign(a, var)

### Function: getinit(a, var)

### Function: get_elsize(var)

### Function: sign2map(a, var)

**Description:** varname,ctype,atype
init,init.r,init.i,pytype
vardebuginfo,vardebugshowvalue,varshowvalue
varrformat

intent

### Function: routsign2map(rout)

**Description:** name,NAME,begintitle,endtitle
rname,ctype,rformat
routdebugshowvalue

### Function: modsign2map(m)

**Description:** modulename

### Function: cb_sign2map(a, var, index)

### Function: cb_routsign2map(rout, um)

**Description:** name,begintitle,endtitle,argname
ctype,rctype,maxnofargs,nofoptargs,returncptr

### Function: common_sign2map(a, var)
