## AI Summary

A file named test_preprocess_data.py.


### Function: plot_func(ax, x, y, ls, label, w)

### Function: test_compiletime_checks()

**Description:** Test decorator invocations -> no replacements.

### Function: test_function_call_without_data(func)

**Description:** Test without data -> no replacements.

### Function: test_function_call_with_dict_input(func)

**Description:** Tests with dict input, unpacking via preprocess_pipeline

### Function: test_function_call_with_dict_data(func)

**Description:** Test with dict data -> label comes from the value of 'x' parameter.

### Function: test_function_call_with_dict_data_not_in_data(func)

**Description:** Test the case that one var is not in data -> half replaces, half kept

### Function: test_function_call_with_pandas_data(func, pd)

**Description:** Test with pandas dataframe -> label comes from ``data["col"].name``.

### Function: test_function_call_replace_all()

**Description:** Test without a "replace_names" argument, all vars should be replaced.

### Function: test_no_label_replacements()

**Description:** Test with "label_namer=None" -> no label replacement at all.

### Function: test_more_args_than_pos_parameter()

### Function: test_docstring_addition()

### Function: test_data_parameter_replacement()

**Description:** Test that the docstring contains the correct *data* parameter stub
for all methods that we run _preprocess_data() on.

## Class: TestPlotTypes

### Function: func(ax, x, y)

### Function: func_args(ax, x, y)

### Function: func_kwargs(ax, x, y)

### Function: func_no_ax_args()

### Function: func_replace_all(ax, x, y, ls, label, w)

### Function: func_no_label(ax, x, y, ls, label, w)

### Function: func(ax, x, y, z)

### Function: funcy(ax)

**Description:** Parameters
----------
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

### Function: funcy(ax, x, y, z, bar)

**Description:** Parameters
----------
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

### Function: funcy(ax, x, y, z, bar)

**Description:** Parameters
----------
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

### Function: funcy(ax, x, y, z, t)

**Description:** Parameters
----------
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

### Function: test_dict_unpack(self, plotter, fig_test, fig_ref)

### Function: test_data_kwarg(self, plotter, fig_test, fig_ref)
