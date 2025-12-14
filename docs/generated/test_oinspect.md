## AI Summary

A file named test_oinspect.py.


### Function: setup_module()

## Class: SourceModuleMainTest

### Function: test_find_source_lines()

### Function: test_getsource()

### Function: test_inspect_getfile_raises_exception()

**Description:** Check oinspect.find_file/getsource/find_source_lines expectations

### Function: pyfile(fname)

### Function: match_pyfiles(f1, f2)

### Function: test_find_file()

### Function: test_find_file_decorated1()

### Function: test_find_file_decorated2()

### Function: test_find_file_magic()

## Class: Call

**Description:** This is the class docstring.

## Class: HasSignature

**Description:** This is the class docstring.

## Class: SimpleClass

## Class: Awkward

## Class: NoBoolCall

**Description:** callable with `__bool__` raising should still be inspect-able.

## Class: SerialLiar

**Description:** Attribute accesses always get another copy of the same class.

unittest.mock.call does something similar, but it's not ideal for testing
as the failure mode is to eat all your RAM. This gives up after 10k levels.

### Function: test_info()

**Description:** Check that Inspector.info fills out various fields as expected.

### Function: test_class_signature()

### Function: test_info_awkward()

### Function: test_bool_raise()

### Function: test_info_serialliar()

### Function: support_function_one(x, y)

**Description:** A simple function.

### Function: test_calldef_none()

### Function: f_kwarg(pos)

### Function: test_definition_kwonlyargs()

### Function: test_getdoc()

### Function: test_empty_property_has_no_source()

### Function: test_property_sources()

### Function: test_property_docstring_is_in_info_for_detail_level_0()

### Function: test_pdef()

### Function: cleanup_user_ns()

**Description:** On exit delete all the keys that were not in user_ns before entering.

It does not restore old values !

Parameters
----------

**kwargs
    used to update ip.user_ns

### Function: test_pinfo_bool_raise()

**Description:** Test that bool method is not called on parent.

### Function: test_pinfo_getindex()

### Function: test_qmark_getindex()

### Function: test_qmark_getindex_negatif()

### Function: test_pinfo_nonascii()

### Function: test_pinfo_type()

**Description:** type can fail in various edge case, for example `type.__subclass__()`

### Function: test_pinfo_docstring_no_source()

**Description:** Docstring should be included with detail_level=1 if there is no source

### Function: test_pinfo_no_docstring_if_source()

**Description:** Docstring should not be included with detail_level=1 if source is found

### Function: test_pinfo_docstring_if_detail_and_no_source()

**Description:** Docstring should be displayed if source info not available 

### Function: test_pinfo_docstring_dynamic(capsys)

### Function: test_pinfo_magic()

### Function: test_init_colors()

### Function: test_builtin_init()

### Function: test_render_signature_short()

### Function: test_render_signature_long()

### Function: noop1(f)

### Function: f(x)

**Description:** My docstring

### Function: noop2(f)

### Function: f(x)

**Description:** My docstring 2

### Function: __init__(self, x, y)

**Description:** This is the constructor docstring.

### Function: __call__(self)

**Description:** This is the call docstring.

### Function: method(self, x, z)

**Description:** Some method's docstring

### Function: __init__(self)

**Description:** This is the init docstring

### Function: method(self, x, z)

**Description:** Some method's docstring

### Function: __getattr__(self, name)

### Function: __call__(self)

**Description:** does nothing

### Function: __bool__(self)

**Description:** just raise NotImplemented

### Function: __init__(self, max_fibbing_twig, lies_told)

### Function: __getattr__(self, item)

## Class: A

**Description:** standard docstring

## Class: B

**Description:** standard docstring

## Class: C

**Description:** standard docstring

### Function: simple_add(a, b)

**Description:** Adds two numbers

## Class: A

## Class: A

### Function: foo()

## Class: RaiseBool

### Function: dummy()

**Description:** MARKER

### Function: dummy()

**Description:** MARKER 2

### Function: dummy()

**Description:** MARKER 3

### Function: foo()

**Description:** foo has a docstring

### Function: short_fun(a)

### Function: long_function(a_really_long_parameter, and_another_long_one, let_us_make_sure_this_is_looong)

### Function: wrapper()

### Function: getdoc(self)

### Function: getdoc(self)

### Function: foo(self)

### Function: foobar(self)

**Description:** This is `foobar` property.

### Function: __bool__(self)
