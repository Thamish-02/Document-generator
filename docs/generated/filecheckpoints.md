## AI Summary

A file named filecheckpoints.py.


## Class: FileCheckpoints

**Description:** A Checkpoints that caches checkpoints for files in adjacent
directories.

Only works with FileContentsManager.  Use GenericFileCheckpoints if
you want file-based checkpoints with another ContentsManager.

## Class: AsyncFileCheckpoints

## Class: GenericFileCheckpoints

**Description:** Local filesystem Checkpoints that works with any conforming
ContentsManager.

## Class: AsyncGenericFileCheckpoints

**Description:** Asynchronous Local filesystem Checkpoints that works with any conforming
ContentsManager.

### Function: _root_dir_default(self)

### Function: create_checkpoint(self, contents_mgr, path)

**Description:** Create a checkpoint.

### Function: restore_checkpoint(self, contents_mgr, checkpoint_id, path)

**Description:** Restore a checkpoint.

### Function: rename_checkpoint(self, checkpoint_id, old_path, new_path)

**Description:** Rename a checkpoint from old_path to new_path.

### Function: delete_checkpoint(self, checkpoint_id, path)

**Description:** delete a file's checkpoint

### Function: list_checkpoints(self, path)

**Description:** list the checkpoints for a given file

This contents manager currently only supports one checkpoint per file.

### Function: checkpoint_path(self, checkpoint_id, path)

**Description:** find the path to a checkpoint

### Function: checkpoint_model(self, checkpoint_id, os_path)

**Description:** construct the info dict for a given checkpoint

### Function: no_such_checkpoint(self, path, checkpoint_id)

### Function: create_file_checkpoint(self, content, format, path)

**Description:** Create a checkpoint from the current content of a file.

### Function: create_notebook_checkpoint(self, nb, path)

**Description:** Create a checkpoint from the current content of a notebook.

### Function: get_notebook_checkpoint(self, checkpoint_id, path)

**Description:** Get a checkpoint for a notebook.

### Function: get_file_checkpoint(self, checkpoint_id, path)

**Description:** Get a checkpoint for a file.
