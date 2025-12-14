## AI Summary

A file named auxfuncs.py.


### Function: outmess(t)

### Function: debugcapi(var)

### Function: _ischaracter(var)

### Function: _isstring(var)

### Function: ischaracter_or_characterarray(var)

### Function: ischaracter(var)

### Function: ischaracterarray(var)

### Function: isstring_or_stringarray(var)

### Function: isstring(var)

### Function: isstringarray(var)

### Function: isarrayofstrings(var)

### Function: isarray(var)

### Function: isscalar(var)

### Function: iscomplex(var)

### Function: islogical(var)

### Function: isinteger(var)

### Function: isreal(var)

### Function: get_kind(var)

### Function: isint1(var)

### Function: islong_long(var)

### Function: isunsigned_char(var)

### Function: isunsigned_short(var)

### Function: isunsigned(var)

### Function: isunsigned_long_long(var)

### Function: isdouble(var)

### Function: islong_double(var)

### Function: islong_complex(var)

### Function: iscomplexarray(var)

### Function: isint1array(var)

### Function: isunsigned_chararray(var)

### Function: isunsigned_shortarray(var)

### Function: isunsignedarray(var)

### Function: isunsigned_long_longarray(var)

### Function: issigned_chararray(var)

### Function: issigned_shortarray(var)

### Function: issigned_array(var)

### Function: issigned_long_longarray(var)

### Function: isallocatable(var)

### Function: ismutable(var)

### Function: ismoduleroutine(rout)

### Function: ismodule(rout)

### Function: isfunction(rout)

### Function: isfunction_wrap(rout)

### Function: issubroutine(rout)

### Function: issubroutine_wrap(rout)

### Function: isattr_value(var)

### Function: hasassumedshape(rout)

### Function: requiresf90wrapper(rout)

### Function: isroutine(rout)

### Function: islogicalfunction(rout)

### Function: islong_longfunction(rout)

### Function: islong_doublefunction(rout)

### Function: iscomplexfunction(rout)

### Function: iscomplexfunction_warn(rout)

### Function: isstringfunction(rout)

### Function: hasexternals(rout)

### Function: isthreadsafe(rout)

### Function: hasvariables(rout)

### Function: isoptional(var)

### Function: isexternal(var)

### Function: getdimension(var)

### Function: isrequired(var)

### Function: iscstyledirective(f2py_line)

### Function: isintent_in(var)

### Function: isintent_inout(var)

### Function: isintent_out(var)

### Function: isintent_hide(var)

### Function: isintent_nothide(var)

### Function: isintent_c(var)

### Function: isintent_cache(var)

### Function: isintent_copy(var)

### Function: isintent_overwrite(var)

### Function: isintent_callback(var)

### Function: isintent_inplace(var)

### Function: isintent_aux(var)

### Function: isintent_aligned4(var)

### Function: isintent_aligned8(var)

### Function: isintent_aligned16(var)

### Function: isprivate(var)

### Function: isvariable(var)

### Function: hasinitvalue(var)

### Function: hasinitvalueasstring(var)

### Function: hasnote(var)

### Function: hasresultnote(rout)

### Function: hascommon(rout)

### Function: containscommon(rout)

### Function: containsmodule(block)

### Function: hasbody(rout)

### Function: hascallstatement(rout)

### Function: istrue(var)

### Function: isfalse(var)

## Class: F2PYError

## Class: throw_error

### Function: l_and()

### Function: l_or()

### Function: l_not(f)

### Function: isdummyroutine(rout)

### Function: getfortranname(rout)

### Function: getmultilineblock(rout, blockname, comment, counter)

### Function: getcallstatement(rout)

### Function: getcallprotoargument(rout, cb_map)

### Function: getusercode(rout)

### Function: getusercode1(rout)

### Function: getpymethoddef(rout)

### Function: getargs(rout)

### Function: getargs2(rout)

### Function: getrestdoc(rout)

### Function: gentitle(name)

### Function: flatlist(lst)

### Function: stripcomma(s)

### Function: replace(str, d, defaultsep)

### Function: dictappend(rd, ar)

### Function: applyrules(rules, d, var)

### Function: get_f2py_modulename(source)

### Function: getuseblocks(pymod)

### Function: process_f2cmap_dict(f2cmap_all, new_map, c2py_map, verbose)

**Description:** Update the Fortran-to-C type mapping dictionary with new mappings and
return a list of successfully mapped C types.

This function integrates a new mapping dictionary into an existing
Fortran-to-C type mapping dictionary. It ensures that all keys are in
lowercase and validates new entries against a given C-to-Python mapping
dictionary. Redefinitions and invalid entries are reported with a warning.

Parameters
----------
f2cmap_all : dict
    The existing Fortran-to-C type mapping dictionary that will be updated.
    It should be a dictionary of dictionaries where the main keys represent
    Fortran types and the nested dictionaries map Fortran type specifiers
    to corresponding C types.

new_map : dict
    A dictionary containing new type mappings to be added to `f2cmap_all`.
    The structure should be similar to `f2cmap_all`, with keys representing
    Fortran types and values being dictionaries of type specifiers and their
    C type equivalents.

c2py_map : dict
    A dictionary used for validating the C types in `new_map`. It maps C
    types to corresponding Python types and is used to ensure that the C
    types specified in `new_map` are valid.

verbose : boolean
    A flag used to provide information about the types mapped

Returns
-------
tuple of (dict, list)
    The updated Fortran-to-C type mapping dictionary and a list of
    successfully mapped C types.

### Function: __init__(self, mess)

### Function: __call__(self, var)
