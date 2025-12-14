## AI Summary

A file named checkpoints.py.


## Class: Checkpoints

**Description:** Base class for managing checkpoints for a ContentsManager.

Subclasses are required to implement:

create_checkpoint(self, contents_mgr, path)
restore_checkpoint(self, contents_mgr, checkpoint_id, path)
rename_checkpoint(self, checkpoint_id, old_path, new_path)
delete_checkpoint(self, checkpoint_id, path)
list_checkpoints(self, path)

## Class: GenericCheckpointsMixin

**Description:** Helper for creating Checkpoints subclasses that can be used with any
ContentsManager.

Provides a ContentsManager-agnostic implementation of `create_checkpoint`
and `restore_checkpoint` in terms of the following operations:

- create_file_checkpoint(self, content, format, path)
- create_notebook_checkpoint(self, nb, path)
- get_file_checkpoint(self, checkpoint_id, path)
- get_notebook_checkpoint(self, checkpoint_id, path)

To create a generic CheckpointManager, add this mixin to a class that
implement the above four methods plus the remaining Checkpoints API
methods:

- delete_checkpoint(self, checkpoint_id, path)
- list_checkpoints(self, path)
- rename_checkpoint(self, checkpoint_id, old_path, new_path)

## Class: AsyncCheckpoints

**Description:** Base class for managing checkpoints for a ContentsManager asynchronously.

## Class: AsyncGenericCheckpointsMixin

**Description:** Helper for creating Asynchronous Checkpoints subclasses that can be used with any
ContentsManager.

### Function: create_checkpoint(self, contents_mgr, path)

**Description:** Create a checkpoint.

### Function: restore_checkpoint(self, contents_mgr, checkpoint_id, path)

**Description:** Restore a checkpoint

### Function: rename_checkpoint(self, checkpoint_id, old_path, new_path)

**Description:** Rename a single checkpoint from old_path to new_path.

### Function: delete_checkpoint(self, checkpoint_id, path)

**Description:** delete a checkpoint for a file

### Function: list_checkpoints(self, path)

**Description:** Return a list of checkpoints for a given file

### Function: rename_all_checkpoints(self, old_path, new_path)

**Description:** Rename all checkpoints for old_path to new_path.

### Function: delete_all_checkpoints(self, path)

**Description:** Delete all checkpoints for the given path.

### Function: create_checkpoint(self, contents_mgr, path)

### Function: restore_checkpoint(self, contents_mgr, checkpoint_id, path)

**Description:** Restore a checkpoint.

### Function: create_file_checkpoint(self, content, format, path)

**Description:** Create a checkpoint of the current state of a file

Returns a checkpoint model for the new checkpoint.

### Function: create_notebook_checkpoint(self, nb, path)

**Description:** Create a checkpoint of the current state of a file

Returns a checkpoint model for the new checkpoint.

### Function: get_file_checkpoint(self, checkpoint_id, path)

**Description:** Get the content of a checkpoint for a non-notebook file.

Returns a dict of the form::

    {
        'type': 'file',
        'content': <str>,
        'format': {'text','base64'},
    }

### Function: get_notebook_checkpoint(self, checkpoint_id, path)

**Description:** Get the content of a checkpoint for a notebook.

Returns a dict of the form::

    {
        'type': 'notebook',
        'content': <output of nbformat.read>,
    }
