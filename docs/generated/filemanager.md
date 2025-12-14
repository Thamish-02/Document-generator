## AI Summary

A file named filemanager.py.


## Class: FileContentsManager

**Description:** A file contents manager.

## Class: AsyncFileContentsManager

**Description:** An async file contents manager.

### Function: _default_root_dir(self)

### Function: _validate_root_dir(self, proposal)

### Function: _default_preferred_dir(self)

### Function: _validate_preferred_dir(self, proposal)

### Function: _checkpoints_class_default(self)

### Function: _files_handler_class_default(self)

### Function: _files_handler_params_default(self)

### Function: is_hidden(self, path)

**Description:** Does the API style path correspond to a hidden directory or file?

Parameters
----------
path : str
    The path to check. This is an API path (`/` separated,
    relative to root_dir).

Returns
-------
hidden : bool
    Whether the path exists and is hidden.

### Function: is_writable(self, path)

**Description:** Does the API style path correspond to a writable directory or file?

Parameters
----------
path : str
    The path to check. This is an API path (`/` separated,
    relative to root_dir).

Returns
-------
hidden : bool
    Whether the path exists and is writable.

### Function: file_exists(self, path)

**Description:** Returns True if the file exists, else returns False.

API-style wrapper for os.path.isfile

Parameters
----------
path : str
    The relative path to the file (with '/' as separator)

Returns
-------
exists : bool
    Whether the file exists.

### Function: dir_exists(self, path)

**Description:** Does the API-style path refer to an extant directory?

API-style wrapper for os.path.isdir

Parameters
----------
path : str
    The path to check. This is an API path (`/` separated,
    relative to root_dir).

Returns
-------
exists : bool
    Whether the path is indeed a directory.

### Function: exists(self, path)

**Description:** Returns True if the path exists, else returns False.

API-style wrapper for os.path.exists

Parameters
----------
path : str
    The API path to the file (with '/' as separator)

Returns
-------
exists : bool
    Whether the target exists.

### Function: _base_model(self, path)

**Description:** Build the common base of a contents model

### Function: _dir_model(self, path, content)

**Description:** Build a model for a directory

if content is requested, will include a listing of the directory

### Function: _file_model(self, path, content, format, require_hash)

**Description:** Build a model for a file

if content is requested, include the file contents.

format:
  If 'text', the contents will be decoded as UTF-8.
  If 'base64', the raw bytes contents will be encoded as base64.
  If not specified, try to decode as UTF-8, and fall back to base64

if require_hash is true, the model will include 'hash'

### Function: _notebook_model(self, path, content, require_hash)

**Description:** Build a notebook model

if content is requested, the notebook content will be populated
as a JSON structure (not double-serialized)

if require_hash is true, the model will include 'hash'

### Function: get(self, path, content, type, format, require_hash)

**Description:** Takes a path for an entity and returns its model

Parameters
----------
path : str
    the API path that describes the relative path for the target
content : bool
    Whether to include the contents in the reply
type : str, optional
    The requested type - 'file', 'notebook', or 'directory'.
    Will raise HTTPError 400 if the content doesn't match.
format : str, optional
    The requested format for file contents. 'text' or 'base64'.
    Ignored if this returns a notebook or directory model.
require_hash: bool, optional
    Whether to include the hash of the file contents.

Returns
-------
model : dict
    the contents model. If content=True, returns the contents
    of the file or directory as well.

### Function: _save_directory(self, os_path, model, path)

**Description:** create a directory

### Function: save(self, model, path)

**Description:** Save the file model and return the model with no content.

### Function: delete_file(self, path)

**Description:** Delete file at path.

### Function: rename_file(self, old_path, new_path)

**Description:** Rename a file.

### Function: info_string(self)

**Description:** Get the information string for the manager.

### Function: get_kernel_path(self, path, model)

**Description:** Return the initial API path of  a kernel associated with a given notebook

### Function: copy(self, from_path, to_path)

**Description:** Copy an existing file or directory and return its new model.
If to_path not specified, it will be the parent directory of from_path.
If copying a file and to_path is a directory, filename/directoryname will increment `from_path-Copy#.ext`.
Considering multi-part extensions, the Copy# part will be placed before the first dot for all the extensions except `ipynb`.
For easier manual searching in case of notebooks, the Copy# part will be placed before the last dot.
from_path must be a full path to a file or directory.

### Function: _copy_dir(self, from_path, to_path_original, to_name, to_path)

**Description:** handles copying directories
returns the model for the copied directory

### Function: check_folder_size(self, path)

**Description:** limit the size of folders being copied to be no more than the
trait max_copy_folder_size_mb to prevent a timeout error

### Function: _get_dir_size(self, path)

**Description:** calls the command line program du to get the directory size

### Function: _human_readable_size(self, size)

**Description:** returns folder size in a human readable format

### Function: _checkpoints_class_default(self)

### Function: is_non_empty_dir(os_path)
