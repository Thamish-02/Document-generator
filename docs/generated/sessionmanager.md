## AI Summary

A file named sessionmanager.py.


## Class: KernelSessionRecordConflict

**Description:** Exception class to use when two KernelSessionRecords cannot
merge because of conflicting data.

## Class: KernelSessionRecord

**Description:** A record object for tracking a Jupyter Server Kernel Session.

Two records that share a session_id must also share a kernel_id, while
kernels can have multiple session (and thereby) session_ids
associated with them.

## Class: KernelSessionRecordList

**Description:** An object for storing and managing a list of KernelSessionRecords.

When adding a record to the list, the KernelSessionRecordList
first checks if the record already exists in the list. If it does,
the record will be updated with the new information; otherwise,
it will be appended.

## Class: SessionManager

**Description:** A session manager.

### Function: __eq__(self, other)

**Description:** Whether a record equals another.

### Function: update(self, other)

**Description:** Updates in-place a kernel from other (only accepts positive updates

### Function: __init__(self)

**Description:** Initialize a record list.

### Function: __str__(self)

**Description:** The string representation of a record list.

### Function: __contains__(self, record)

**Description:** Search for records by kernel_id and session_id

### Function: __len__(self)

**Description:** The length of the record list.

### Function: get(self, record)

**Description:** Return a full KernelSessionRecord from a session_id, kernel_id, or
incomplete KernelSessionRecord.

### Function: update(self, record)

**Description:** Update a record in-place or append it if not in the list.

### Function: remove(self, record)

**Description:** Remove a record if its found in the list. If it's not found,
do nothing.

### Function: _validate_database_filepath(self, proposal)

**Description:** Validate a database file path.

### Function: __init__(self)

**Description:** Initialize a record list.

### Function: cursor(self)

**Description:** Start a cursor and create a database called 'session'

### Function: connection(self)

**Description:** Start a database connection

### Function: close(self)

**Description:** Close the sqlite connection

### Function: __del__(self)

**Description:** Close connection once SessionManager closes

### Function: new_session_id(self)

**Description:** Create a uuid for a new session

### Function: get_kernel_env(self, path, name)

**Description:** Return the environment variables that need to be set in the kernel

Parameters
----------
path : str
    the url path for the given session.
name: ModelName(str), optional
    Here the name is likely to be the name of the associated file
    with the current kernel at startup time.
