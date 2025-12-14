## AI Summary

A file named client.py.


### Function: timestamp(msg)

**Description:** Get the timestamp for a message.

## Class: NotebookClient

**Description:** Encompasses a Client for executing cells in a notebook

### Function: execute(nb, cwd, km)

**Description:** Execute a notebook's code, updating outputs within the notebook object.

This is a convenient wrapper around NotebookClient. It returns the
modified notebook object.

Parameters
----------
nb : NotebookNode
  The notebook object to be executed
cwd : str, optional
  If supplied, the kernel will run in this directory
km : AsyncKernelManager, optional
  If supplied, the specified kernel manager will be used for code execution.
kwargs :
  Any other options for NotebookClient, e.g. timeout, kernel_name

### Function: _kernel_manager_class_default(self)

**Description:** Use a dynamic default to avoid importing jupyter_client at startup

### Function: __init__(self, nb, km)

**Description:** Initializes the execution manager.

Parameters
----------
nb : NotebookNode
    Notebook being executed.
km : KernelManager (optional)
    Optional kernel manager. If none is provided, a kernel manager will
    be created.

### Function: reset_execution_trackers(self)

**Description:** Resets any per-execution trackers.

### Function: create_kernel_manager(self)

**Description:** Creates a new kernel manager.

Returns
-------
km : KernelManager
    Kernel manager whose client class is asynchronous.

### Function: setup_kernel(self)

**Description:** Context manager for setting up the kernel to execute a notebook.

The assigns the Kernel Manager (``self.km``) if missing and Kernel Client(``self.kc``).

When control returns from the yield it stops the client's zmq channels, and shuts
down the kernel.

### Function: set_widgets_metadata(self)

**Description:** Set with widget metadata.

### Function: _update_display_id(self, display_id, msg)

**Description:** Update outputs with a given display_id

### Function: _get_timeout(self, cell)

### Function: _passed_deadline(self, deadline)

### Function: process_message(self, msg, cell, cell_index)

**Description:** Processes a kernel message, updates cell state, and returns the
resulting output object that was appended to cell.outputs.

The input argument *cell* is modified in-place.

Parameters
----------
msg : dict
    The kernel message being processed.
cell : nbformat.NotebookNode
    The cell which is currently being processed.
cell_index : int
    The position of the cell within the notebook object.

Returns
-------
output : NotebookNode
    The execution output payload (or None for no output).

Raises
------
CellExecutionComplete
  Once a message arrives which indicates computation completeness.

### Function: output(self, outs, msg, display_id, cell_index)

**Description:** Handle output.

### Function: clear_output(self, outs, msg, cell_index)

**Description:** Clear output.

### Function: clear_display_id_mapping(self, cell_index)

**Description:** Clear a display id mapping for a cell.

### Function: handle_comm_msg(self, outs, msg, cell_index)

**Description:** Handle a comm message.

### Function: _serialize_widget_state(self, state)

**Description:** Serialize a widget state, following format in @jupyter-widgets/schema.

### Function: _get_buffer_data(self, msg)

### Function: register_output_hook(self, msg_id, hook)

**Description:** Registers an override object that handles output/clear_output instead.

Multiple hooks can be registered, where the last one will be used (stack based)

### Function: remove_output_hook(self, msg_id, hook)

**Description:** Unregisters an override object that handles output/clear_output instead

### Function: on_comm_open_jupyter_widget(self, msg)

**Description:** Handle a jupyter widget comm open.

### Function: on_signal()

**Description:** Handle signals.
