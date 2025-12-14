## AI Summary

A file named tagremove.py.


## Class: TagRemovePreprocessor

**Description:** Removes inputs, outputs, or cells from a notebook that
have tags that designate they are to be removed prior to exporting
the notebook.

remove_cell_tags
    removes cells tagged with these values

remove_all_outputs_tags
    removes entire output areas on cells
    tagged with these values

remove_single_output_tags
    removes individual output objects on
    outputs tagged with these values

remove_input_tags
    removes inputs tagged with these values

### Function: check_cell_conditions(self, cell, resources, index)

**Description:** Checks that a cell has a tag that is to be removed

Returns: Boolean.
True means cell should *not* be removed.

### Function: preprocess(self, nb, resources)

**Description:** Preprocessing to apply to each notebook. See base.py for details.

### Function: preprocess_cell(self, cell, resources, cell_index)

**Description:** Apply a transformation on each cell. See base.py for details.

### Function: check_output_conditions(self, output, resources, cell_index, output_index)

**Description:** Checks that an output has a tag that indicates removal.

Returns: Boolean.
True means output should *not* be removed.
