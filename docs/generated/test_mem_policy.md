## AI Summary

A file named test_mem_policy.py.


### Function: get_module(tmp_path)

**Description:** Add a memory policy that returns a false pointer 64 bytes into the
actual allocation, and fill the prefix with some text. Then check at each
memory manipulation that the prefix exists, to make sure all alloc/realloc/
free/calloc go via the functions here.

### Function: test_set_policy(get_module)

### Function: test_default_policy_singleton(get_module)

### Function: test_policy_propagation(get_module)

### Function: test_context_locality(get_module)

### Function: concurrent_thread1(get_module, event)

### Function: concurrent_thread2(get_module, event)

### Function: test_thread_locality(get_module)

### Function: test_new_policy(get_module)

### Function: test_switch_owner(get_module, policy)

### Function: test_owner_is_base(get_module)

## Class: MyArr
