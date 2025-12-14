## AI Summary

A file named pydevd_vars.py.


## Class: VariableError

### Function: iter_frames(frame)

### Function: dump_frames(thread_id)

### Function: getVariable(dbg, thread_id, frame_id, scope, locator)

**Description:** returns the value of a variable

:scope: can be BY_ID, EXPRESSION, GLOBAL, LOCAL, FRAME

BY_ID means we'll traverse the list of all objects alive to get the object.

:locator: after reaching the proper scope, we have to get the attributes until we find
        the proper location (i.e.: obj      attr1   attr2)

:note: when BY_ID is used, the frame_id is considered the id of the object to find and
       not the frame (as we don't care about the frame in this case).

### Function: resolve_compound_variable_fields(dbg, thread_id, frame_id, scope, attrs)

**Description:** Resolve compound variable in debugger scopes by its name and attributes

:param thread_id: id of the variable's thread
:param frame_id: id of the variable's frame
:param scope: can be BY_ID, EXPRESSION, GLOBAL, LOCAL, FRAME
:param attrs: after reaching the proper scope, we have to get the attributes until we find
        the proper location (i.e.: obj      attr1   attr2)
:return: a dictionary of variables's fields

### Function: resolve_var_object(var, attrs)

**Description:** Resolve variable's attribute

:param var: an object of variable
:param attrs: a sequence of variable's attributes separated by       (i.e.: obj     attr1   attr2)
:return: a value of resolved variable's attribute

### Function: resolve_compound_var_object_fields(var, attrs)

**Description:** Resolve compound variable by its object and attributes

:param var: an object of variable
:param attrs: a sequence of variable's attributes separated by       (i.e.: obj     attr1   attr2)
:return: a dictionary of variables's fields

### Function: custom_operation(dbg, thread_id, frame_id, scope, attrs, style, code_or_file, operation_fn_name)

**Description:** We'll execute the code_or_file and then search in the namespace the operation_fn_name to execute with the given var.

code_or_file: either some code (i.e.: from pprint import pprint) or a file to be executed.
operation_fn_name: the name of the operation to execute after the exec (i.e.: pprint)

### Function: _expression_to_evaluate(expression)

### Function: eval_in_context(expression, global_vars, local_vars, py_db)

### Function: _run_with_interrupt_thread(original_func, py_db, curr_thread, frame, expression, is_exec)

### Function: _run_with_unblock_threads(original_func, py_db, curr_thread, frame, expression, is_exec)

### Function: _evaluate_with_timeouts(original_func)

**Description:** Provides a decorator that wraps the original evaluate to deal with slow evaluates.

If some evaluation is too slow, we may show a message, resume threads or interrupt them
as needed (based on the related configurations).

### Function: compile_as_eval(expression)

**Description:** :param expression:
    The expression to be _compiled.

:return: code object

:raises Exception if the expression cannot be evaluated.

### Function: _compile_as_exec(expression)

**Description:** :param expression:
    The expression to be _compiled.

:return: code object

:raises Exception if the expression cannot be evaluated.

## Class: _EvalAwaitInNewEventLoop

### Function: evaluate_expression(py_db, frame, expression, is_exec)

**Description:** :param str expression:
    The expression to be evaluated.

    Note that if the expression is indented it's automatically dedented (based on the indentation
    found on the first non-empty line).

    i.e.: something as:

    `
        def method():
            a = 1
    `

    becomes:

    `
    def method():
        a = 1
    `

    Also, it's possible to evaluate calls with a top-level await (currently this is done by
    creating a new event loop in a new thread and making the evaluate at that thread -- note
    that this is still done synchronously so the evaluation has to finish before this
    function returns).

:param is_exec: determines if we should do an exec or an eval.
    There are some changes in this function depending on whether it's an exec or an eval.

    When it's an exec (i.e.: is_exec==True):
        This function returns None.
        Any exception that happens during the evaluation is reraised.
        If the expression could actually be evaluated, the variable is printed to the console if not None.

    When it's an eval (i.e.: is_exec==False):
        This function returns the result from the evaluation.
        If some exception happens in this case, the exception is caught and a ExceptionOnEvaluate is returned.
        Also, in this case we try to resolve name-mangling (i.e.: to be able to add a self.__my_var watch).

:param py_db:
    The debugger. Only needed if some top-level await is detected (for creating a
    PyDBDaemonThread).

### Function: change_attr_expression(scope)

**Description:** Changes some attribute in a given frame.

### Function: table_like_struct_to_xml(array, name, roffset, coffset, rows, cols, format)

### Function: array_to_xml(array, roffset, coffset, rows, cols, format)

### Function: array_to_meta_xml(array, name, format)

### Function: dataframe_to_xml(df, name, roffset, coffset, rows, cols, format)

**Description:** :type df: pandas.core.frame.DataFrame
:type name: str
:type coffset: int
:type roffset: int
:type rows: int
:type cols: int
:type format: str

### Function: new_func(py_db, frame, expression, is_exec)

### Function: __init__(self, py_db, compiled, updated_globals, updated_locals)

### Function: _on_run(self)

### Function: on_timeout_unblock_threads()

### Function: on_warn_evaluation_timeout()
