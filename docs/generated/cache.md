## AI Summary

A file named cache.py.


### Function: _memoize_default(default, inference_state_is_first_arg, second_arg_is_inference_state)

**Description:** This is a typical memoization decorator, BUT there is one difference:
To prevent recursion it sets defaults.

Preventing recursion is in this case the much bigger use than speed. I
don't think, that there is a big speed difference, but there are many cases
where recursion could happen (think about a = b; b = a).

### Function: inference_state_function_cache(default)

### Function: inference_state_method_cache(default)

### Function: inference_state_as_method_param_cache()

## Class: CachedMetaClass

**Description:** This is basically almost the same than the decorator above, it just caches
class initializations. Either you do it this way or with decorators, but
with decorators you lose class access (isinstance, etc).

### Function: inference_state_method_generator_cache()

**Description:** This is a special memoizer. It memoizes generators and also checks for
recursion errors and returns no further iterator elemends in that case.

### Function: func(function)

### Function: decorator(func)

### Function: decorator(func)

### Function: decorator(call)

### Function: __call__(self)

### Function: func(function)

### Function: wrapper(obj)

### Function: wrapper(obj)
