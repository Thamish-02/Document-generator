## AI Summary

A file named nbbase.py.


### Function: validate(node, ref)

**Description:** validate a v4 node

### Function: new_output(output_type, data)

**Description:** Create a new output, to go in the ``cell.outputs`` list of a code cell.

### Function: output_from_msg(msg)

**Description:** Create a NotebookNode for an output from a kernel's IOPub message.

Returns
-------
NotebookNode: the output as a notebook node.

Raises
------
ValueError: if the message is not an output message.

### Function: new_code_cell(source)

**Description:** Create a new code cell

### Function: new_markdown_cell(source)

**Description:** Create a new markdown cell

### Function: new_raw_cell(source)

**Description:** Create a new raw cell

### Function: new_notebook()

**Description:** Create a new notebook
