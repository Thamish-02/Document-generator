## AI Summary

A file named introspect.py.


### Function: opt_func_info(func_name, signature)

**Description:** Returns a dictionary containing the currently supported CPU dispatched
features for all optimized functions.

Parameters
----------
func_name : str (optional)
    Regular expression to filter by function name.

signature : str (optional)
    Regular expression to filter by data type.

Returns
-------
dict
    A dictionary where keys are optimized function names and values are
    nested dictionaries indicating supported targets based on data types.

Examples
--------
Retrieve dispatch information for functions named 'add' or 'sub' and
data types 'float64' or 'float32':

>>> import numpy as np
>>> dict = np.lib.introspect.opt_func_info(
...     func_name="add|abs", signature="float64|complex64"
... )
>>> import json
>>> print(json.dumps(dict, indent=2))
    {
      "absolute": {
        "dd": {
          "current": "SSE41",
          "available": "SSE41 baseline(SSE SSE2 SSE3)"
        },
        "Ff": {
          "current": "FMA3__AVX2",
          "available": "AVX512F FMA3__AVX2 baseline(SSE SSE2 SSE3)"
        },
        "Dd": {
          "current": "FMA3__AVX2",
          "available": "AVX512F FMA3__AVX2 baseline(SSE SSE2 SSE3)"
        }
      },
      "add": {
        "ddd": {
          "current": "FMA3__AVX2",
          "available": "FMA3__AVX2 baseline(SSE SSE2 SSE3)"
        },
        "FFF": {
          "current": "FMA3__AVX2",
          "available": "FMA3__AVX2 baseline(SSE SSE2 SSE3)"
        }
      }
    }
