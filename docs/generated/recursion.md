## AI Summary

A file named recursion.py.


## Class: RecursionDetector

### Function: execution_allowed(inference_state, node)

**Description:** A decorator to detect recursions in statements. In a recursion a statement
at the same place, in the same module may not be executed two times.

### Function: execution_recursion_decorator(default)

## Class: ExecutionRecursionDetector

**Description:** Catches recursions of executions.

### Function: __init__(self)

### Function: decorator(func)

### Function: __init__(self, inference_state)

### Function: pop_execution(self)

### Function: push_execution(self, execution)

### Function: wrapper(self)
