## AI Summary

A file named manager.py.


## Class: ContentsManager

**Description:** Base class for serving files and directories.

This serves any text or binary file,
as well as directories,
with special handling for JSON notebook documents.

Most APIs take a path argument,
which is always an API-style unicode path,
and always refers to a directory.

- unicode, not url-escaped
- '/'-separated
- leading and trailing '/' will be stripped
- if unspecified, path defaults to '',
  indicating the root path.

## Class: AsyncContentsManager

**Description:** Base class for serving files and directories asynchronously.

### Function: _default_event_logger(self)

### Function: emit(self, data)

**Description:** Emit event using the core event schema from Jupyter Server's Contents Manager.

### Function: _validate_preferred_dir(self, proposal)

### Function: _notary_default(self)

### Function: _validate_pre_save_hook(self, proposal)

### Function: _validate_post_save_hook(self, proposal)

### Function: run_pre_save_hook(self, model, path)

**Description:** Run the pre-save hook if defined, and log errors

### Function: run_post_save_hook(self, model, os_path)

**Description:** Run the post-save hook if defined, and log errors

### Function: register_pre_save_hook(self, hook)

**Description:** Register a pre save hook.

### Function: register_post_save_hook(self, hook)

**Description:** Register a post save hook.

### Function: run_pre_save_hooks(self, model, path)

**Description:** Run the pre-save hooks if any, and log errors

### Function: run_post_save_hooks(self, model, os_path)

**Description:** Run the post-save hooks if any, and log errors

### Function: _default_checkpoints(self)

### Function: _default_checkpoints_kwargs(self)

### Function: get_extra_handlers(self)

**Description:** Return additional handlers

Default: self.files_handler_class on /files/.*

### Function: dir_exists(self, path)

**Description:** Does a directory exist at the given path?

Like os.path.isdir

Override this method in subclasses.

Parameters
----------
path : str
    The path to check

Returns
-------
exists : bool
    Whether the path does indeed exist.

### Function: is_hidden(self, path)

**Description:** Is path a hidden directory or file?

Parameters
----------
path : str
    The path to check. This is an API path (`/` separated,
    relative to root dir).

Returns
-------
hidden : bool
    Whether the path is hidden.

### Function: file_exists(self, path)

**Description:** Does a file exist at the given path?

Like os.path.isfile

Override this method in subclasses.

Parameters
----------
path : str
    The API path of a file to check for.

Returns
-------
exists : bool
    Whether the file exists.

### Function: exists(self, path)

**Description:** Does a file or directory exist at the given path?

Like os.path.exists

Parameters
----------
path : str
    The API path of a file or directory to check for.

Returns
-------
exists : bool
    Whether the target exists.

### Function: get(self, path, content, type, format, require_hash)

**Description:** Get a file or directory model.

Parameters
----------
require_hash : bool
    Whether the file hash must be returned or not.

*Changed in version 2.11*: The *require_hash* parameter was added.

### Function: save(self, model, path)

**Description:** Save a file or directory model to path.

Should return the saved model with no content.  Save implementations
should call self.run_pre_save_hook(model=model, path=path) prior to
writing any data.

### Function: delete_file(self, path)

**Description:** Delete the file or directory at path.

### Function: rename_file(self, old_path, new_path)

**Description:** Rename a file or directory.

### Function: delete(self, path)

**Description:** Delete a file/directory and any associated checkpoints.

### Function: rename(self, old_path, new_path)

**Description:** Rename a file and any checkpoints associated with that file.

### Function: update(self, model, path)

**Description:** Update the file's path

For use in PATCH requests, to enable renaming a file without
re-uploading its contents. Only used for renaming at the moment.

### Function: info_string(self)

**Description:** The information string for the manager.

### Function: get_kernel_path(self, path, model)

**Description:** Return the API path for the kernel

KernelManagers can turn this value into a filesystem path,
or ignore it altogether.

The default value here will start kernels in the directory of the
notebook server. FileContentsManager overrides this to use the
directory containing the notebook.

### Function: increment_filename(self, filename, path, insert)

**Description:** Increment a filename until it is unique.

Parameters
----------
filename : unicode
    The name of a file, including extension
path : unicode
    The API path of the target's directory
insert : unicode
    The characters to insert after the base filename

Returns
-------
name : unicode
    A filename that is unique, based on the input filename.

### Function: validate_notebook_model(self, model, validation_error)

**Description:** Add failed-validation message to model

### Function: new_untitled(self, path, type, ext)

**Description:** Create a new untitled file or directory in path

path must be a directory

File extension can be specified.

Use `new` to create files with a fully specified path (including filename).

### Function: new(self, model, path)

**Description:** Create a new file or directory and return its model with no content.

To create a new untitled entity in a directory, use `new_untitled`.

### Function: copy(self, from_path, to_path)

**Description:** Copy an existing file and return its new model.

If to_path not specified, it will be the parent directory of from_path.
If to_path is a directory, filename will increment `from_path-Copy#.ext`.
Considering multi-part extensions, the Copy# part will be placed before the first dot for all the extensions except `ipynb`.
For easier manual searching in case of notebooks, the Copy# part will be placed before the last dot.

from_path must be a full path to a file.

### Function: log_info(self)

**Description:** Log the information string for the manager.

### Function: trust_notebook(self, path)

**Description:** Explicitly trust a notebook

Parameters
----------
path : str
    The path of a notebook

### Function: check_and_sign(self, nb, path)

**Description:** Check for trusted cells, and sign the notebook.

Called as a part of saving notebooks.

Parameters
----------
nb : dict
    The notebook dict
path : str
    The notebook's path (for logging)

### Function: mark_trusted_cells(self, nb, path)

**Description:** Mark cells as trusted if the notebook signature matches.

Called as a part of loading notebooks.

Parameters
----------
nb : dict
    The notebook object (in current nbformat)
path : str
    The notebook's path (for logging)

### Function: should_list(self, name)

**Description:** Should this file/directory name be displayed in a listing?

### Function: create_checkpoint(self, path)

**Description:** Create a checkpoint.

### Function: restore_checkpoint(self, checkpoint_id, path)

**Description:** Restore a checkpoint.

### Function: list_checkpoints(self, path)

### Function: delete_checkpoint(self, checkpoint_id, path)

### Function: _default_checkpoints(self)

### Function: _default_checkpoints_kwargs(self)
