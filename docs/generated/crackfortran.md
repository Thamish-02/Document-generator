## AI Summary

A file named crackfortran.py.


### Function: reset_global_f2py_vars()

### Function: outmess(line, flag)

### Function: rmbadname1(name)

### Function: rmbadname(names)

### Function: undo_rmbadname1(name)

### Function: undo_rmbadname(names)

### Function: openhook(filename, mode)

**Description:** Ensures that filename is opened with correct encoding parameter.

This function uses charset_normalizer package, when available, for
determining the encoding of the file to be opened. When charset_normalizer
is not available, the function detects only UTF encodings, otherwise, ASCII
encoding is used as fallback.

### Function: is_free_format(fname)

**Description:** Check if file is in free format Fortran.

### Function: readfortrancode(ffile, dowithline, istop)

**Description:** Read fortran codes from files and
 1) Get rid of comments, line continuations, and empty lines; lower cases.
 2) Call dowithline(line) on every line.
 3) Recursively call itself when statement "include '<filename>'" is met.

### Function: split_by_unquoted(line, characters)

**Description:** Splits the line into (line[:i], line[i:]),
where i is the index of first occurrence of one of the characters
not within quotes, or len(line) if no such index exists

### Function: _simplifyargs(argsline)

### Function: crackline(line, reset)

**Description:** reset=-1  --- initialize
reset=0   --- crack the line
reset=1   --- final check if mismatch of blocks occurred

Cracked data is saved in grouplist[0].

### Function: markouterparen(line)

### Function: markoutercomma(line, comma)

### Function: unmarkouterparen(line)

### Function: appenddecl(decl, decl2, force)

### Function: _is_intent_callback(vdecl)

### Function: _resolvetypedefpattern(line)

### Function: parse_name_for_bind(line)

### Function: _resolvenameargspattern(line)

### Function: analyzeline(m, case, line)

**Description:** Reads each line in the input file in sequence and updates global vars.

Effectively reads and collects information from the input file to the
global variable groupcache, a dictionary containing info about each part
of the fortran module.

At the end of analyzeline, information is filtered into the correct dict
keys, but parameter values and dimensions are not yet interpreted.

### Function: appendmultiline(group, context_name, ml)

### Function: cracktypespec0(typespec, ll)

### Function: removespaces(expr)

### Function: markinnerspaces(line)

**Description:** The function replace all spaces in the input variable line which are 
surrounded with quotation marks, with the triplet "@_@".

For instance, for the input "a 'b c'" the function returns "a 'b@_@c'"

Parameters
----------
line : str

Returns
-------
str

### Function: updatevars(typespec, selector, attrspec, entitydecl)

**Description:** Returns last_name, the variable name without special chars, parenthesis
    or dimension specifiers.

Alters groupcache to add the name, typespec, attrspec (and possibly value)
of current variable.

### Function: cracktypespec(typespec, selector)

### Function: setattrspec(decl, attr, force)

### Function: setkindselector(decl, sel, force)

### Function: setcharselector(decl, sel, force)

### Function: getblockname(block, unknown)

### Function: setmesstext(block)

### Function: get_usedict(block)

### Function: get_useparameters(block, param_map)

### Function: postcrack2(block, tab, param_map)

### Function: postcrack(block, args, tab)

**Description:** TODO:
      function return values
      determine expression types if in argument list

### Function: sortvarnames(vars)

### Function: analyzecommon(block)

### Function: analyzebody(block, args, tab)

### Function: buildimplicitrules(block)

### Function: myeval(e, g, l)

**Description:** Like `eval` but returns only integers and floats 

### Function: getlincoef(e, xset)

**Description:** Obtain ``a`` and ``b`` when ``e == "a*x+b"``, where ``x`` is a symbol in
xset.

>>> getlincoef('2*x + 1', {'x'})
(2, 1, 'x')
>>> getlincoef('3*x + x*2 + 2 + 1', {'x'})
(5, 3, 'x')
>>> getlincoef('0', {'x'})
(0, 0, None)
>>> getlincoef('0*x', {'x'})
(0, 0, 'x')
>>> getlincoef('x*x', {'x'})
(None, None, None)

This can be tricked by sufficiently complex expressions

>>> getlincoef('(x - 0.5)*(x - 1.5)*(x - 1)*x + 2*x + 3', {'x'})
(2.0, 3.0, 'x')

### Function: _get_depend_dict(name, vars, deps)

### Function: _calc_depend_dict(vars)

### Function: get_sorted_names(vars)

### Function: _kind_func(string)

### Function: _selected_int_kind_func(r)

### Function: _selected_real_kind_func(p, r, radix)

### Function: get_parameters(vars, global_params)

### Function: _eval_length(length, params)

### Function: _eval_scalar(value, params)

### Function: analyzevars(block)

**Description:** Sets correct dimension information for each variable/parameter

### Function: param_eval(v, g_params, params, dimspec)

**Description:** Creates a dictionary of indices and values for each parameter in a
parameter array to be evaluated later.

WARNING: It is not possible to initialize multidimensional array
parameters e.g. dimension(-3:1, 4, 3:5) at this point. This is because in
Fortran initialization through array constructor requires the RESHAPE
intrinsic function. Since the right-hand side of the parameter declaration
is not executed in f2py, but rather at the compiled c/fortran extension,
later, it is not possible to execute a reshape of a parameter array.
One issue remains: if the user wants to access the array parameter from
python, we should either
1) allow them to access the parameter array using python standard indexing
   (which is often incompatible with the original fortran indexing)
2) allow the parameter array to be accessed in python as a dictionary with
   fortran indices as keys
We are choosing 2 for now.

### Function: param_parse(d, params)

**Description:** Recursively parse array dimensions.

Parses the declaration of an array variable or parameter
`dimension` keyword, and is called recursively if the
dimension for this array is a previously defined parameter
(found in `params`).

Parameters
----------
d : str
    Fortran expression describing the dimension of an array.
params : dict
    Previously parsed parameters declared in the Fortran source file.

Returns
-------
out : str
    Parsed dimension expression.

Examples
--------

* If the line being analyzed is

  `integer, parameter, dimension(2) :: pa = (/ 3, 5 /)`

  then `d = 2` and we return immediately, with

>>> d = '2'
>>> param_parse(d, params)
2

* If the line being analyzed is

  `integer, parameter, dimension(pa) :: pb = (/1, 2, 3/)`

  then `d = 'pa'`; since `pa` is a previously parsed parameter,
  and `pa = 3`, we call `param_parse` recursively, to obtain

>>> d = 'pa'
>>> params = {'pa': 3}
>>> param_parse(d, params)
3

* If the line being analyzed is

  `integer, parameter, dimension(pa(1)) :: pb = (/1, 2, 3/)`

  then `d = 'pa(1)'`; since `pa` is a previously parsed parameter,
  and `pa(1) = 3`, we call `param_parse` recursively, to obtain

>>> d = 'pa(1)'
>>> params = dict(pa={1: 3, 2: 5})
>>> param_parse(d, params)
3

### Function: expr2name(a, block, args)

### Function: analyzeargs(block)

### Function: _ensure_exprdict(r)

### Function: determineexprtype(expr, vars, rules)

### Function: crack2fortrangen(block, tab, as_interface)

### Function: common2fortran(common, tab)

### Function: use2fortran(use, tab)

### Function: true_intent_list(var)

### Function: vars2fortran(block, vars, args, tab, as_interface)

### Function: crackfortran(files)

### Function: crack2fortran(block)

### Function: _is_visit_pair(obj)

### Function: traverse(obj, visit, parents, result)

**Description:** Traverse f2py data structure with the following visit function:

def visit(item, parents, result, *args, **kwargs):
    """

    parents is a list of key-"f2py data structure" pairs from which
    items are taken from.

    result is a f2py data structure that is filled with the
    return value of the visit function.

    item is 2-tuple (index, value) if parents[-1][1] is a list
    item is 2-tuple (key, value) if parents[-1][1] is a dict

    The return value of visit must be None, or of the same kind as
    item, that is, if parents[-1] is a list, the return value must
    be 2-tuple (new_index, new_value), or if parents[-1] is a
    dict, the return value must be 2-tuple (new_key, new_value).

    If new_index or new_value is None, the return value of visit
    is ignored, that is, it will not be added to the result.

    If the return value is None, the content of obj will be
    traversed, otherwise not.
    """

### Function: character_backward_compatibility_hook(item, parents, result)

**Description:** Previously, Fortran character was incorrectly treated as
character*1. This hook fixes the usage of the corresponding
variables in `check`, `dimension`, `=`, and `callstatement`
expressions.

The usage of `char*` in `callprotoargument` expression can be left
unchanged because C `character` is C typedef of `char`, although,
new implementations should use `character*` in the corresponding
expressions.

See https://github.com/numpy/numpy/pull/19388 for more information.

### Function: fix_usage(varname, value)

### Function: compute_deps(v, deps)

### Function: solve_v(s, a, b)
