## AI Summary

A file named upgrade_extension.py.


### Function: update_extension(target, vcs_ref, interactive)

**Description:** Update an extension to the current JupyterLab

target: str
    Path to the extension directory containing the extension
vcs_ref: str [default: None]
    Template vcs_ref to checkout
interactive: bool [default: true]
    Whether to ask before overwriting content
