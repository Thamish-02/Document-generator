## AI Summary

A file named convert.py.


### Function: _warn_if_invalid(nb, version)

**Description:** Log validation errors, if there are any.

### Function: upgrade(nb, from_version, from_minor)

**Description:** Convert a notebook to latest v4.

Parameters
----------
nb : NotebookNode
    The Python representation of the notebook to convert.
from_version : int
    The original version of the notebook to convert.
from_minor : int
    The original minor version of the notebook to convert (only relevant for v >= 3).

### Function: upgrade_cell(cell)

**Description:** upgrade a cell from v3 to v4

heading cell:
    - -> markdown heading
code cell:
    - remove language metadata
    - cell.input -> cell.source
    - cell.prompt_number -> cell.execution_count
    - update outputs

### Function: downgrade_cell(cell)

**Description:** downgrade a cell from v4 to v3

code cell:
    - set cell.language
    - cell.input <- cell.source
    - cell.prompt_number <- cell.execution_count
    - update outputs
markdown cell:
    - single-line heading -> heading cell

### Function: to_mime_key(d)

**Description:** convert dict with v3 aliases to plain mime-type keys

### Function: from_mime_key(d)

**Description:** convert dict with mime-type keys to v3 aliases

### Function: upgrade_output(output)

**Description:** upgrade a single code cell output from v3 to v4

- pyout -> execute_result
- pyerr -> error
- output.type -> output.data.mime/type
- mime-type keys
- stream.stream -> stream.name

### Function: downgrade_output(output)

**Description:** downgrade a single code cell output to v3 from v4

- pyout <- execute_result
- pyerr <- error
- output.data.mime/type -> output.type
- un-mime-type keys
- stream.stream <- stream.name

### Function: upgrade_outputs(outputs)

**Description:** upgrade outputs of a code cell from v3 to v4

### Function: downgrade_outputs(outputs)

**Description:** downgrade outputs of a code cell to v3 from v4

### Function: downgrade(nb)

**Description:** Convert a v4 notebook to v3.

Parameters
----------
nb : NotebookNode
    The Python representation of the notebook to convert.
