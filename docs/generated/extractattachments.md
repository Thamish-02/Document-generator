## AI Summary

A file named extractattachments.py.


## Class: ExtractAttachmentsPreprocessor

**Description:** Extracts attachments from all (markdown and raw) cells in a notebook.
The extracted attachments are stored in a directory ('attachments' by default).
https://nbformat.readthedocs.io/en/latest/format_description.html#cell-attachments

### Function: __init__(self)

**Description:** Public constructor

### Function: preprocess(self, nb, resources)

**Description:** Determine some settings and apply preprocessor to notebook

### Function: preprocess_cell(self, cell, resources, index)

**Description:** Extract attachments to individual files and
change references to them.
E.g.
'![image.png](attachment:021fdd80.png)'
becomes
'![image.png]({path_name}/021fdd80.png)'
Assumes self.path_name and self.resources_item_key is set properly (usually in preprocess).
