## AI Summary

A file named clearmetadata.py.


## Class: ClearMetadataPreprocessor

**Description:** Removes all the metadata from all code cells in a notebook.

### Function: current_key(self, mask_key)

**Description:** Get the current key for a mask key.

### Function: current_mask(self, mask)

**Description:** Get the current mask for a mask.

### Function: nested_masks(self, mask)

**Description:** Get the nested masks for a mask.

### Function: nested_filter(self, items, mask)

**Description:** Get the nested filter for items given a mask.

### Function: preprocess_cell(self, cell, resources, cell_index)

**Description:** All the code cells are returned with an empty metadata field.

### Function: preprocess(self, nb, resources)

**Description:** Preprocessing to apply on each notebook.

Must return modified nb, resources.

Parameters
----------
nb : NotebookNode
    Notebook being converted
resources : dictionary
    Additional resources used in the conversion process.  Allows
    preprocessors to pass variables into the Jinja engine.
