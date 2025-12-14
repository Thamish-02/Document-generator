## AI Summary

A file named sign.py.


## Class: SignatureStore

**Description:** Base class for a signature store.

## Class: MemorySignatureStore

**Description:** Non-persistent storage of signatures in memory.

## Class: SQLiteSignatureStore

**Description:** Store signatures in an SQLite database.

### Function: yield_everything(obj)

**Description:** Yield every item in a container as bytes

Allows any JSONable object to be passed to an HMAC digester
without having to serialize the whole thing.

### Function: yield_code_cells(nb)

**Description:** Iterator that yields all cells in a notebook

nbformat version independent

### Function: signature_removed(nb)

**Description:** Context manager for operating on a notebook with its signature removed

Used for excluding the previous signature when computing a notebook's signature.

## Class: NotebookNotary

**Description:** A class for computing and verifying notebook signatures.

## Class: TrustNotebookApp

**Description:** An application for handling notebook trust.

### Function: adapt_datetime_iso(val)

**Description:** Adapt datetime.datetime to timezone-naive ISO 8601 date.

### Function: convert_datetime(val)

**Description:** Convert ISO 8601 datetime to datetime.datetime object.

### Function: store_signature(self, digest, algorithm)

**Description:** Implement in subclass to store a signature.

Should not raise if the signature is already stored.

### Function: check_signature(self, digest, algorithm)

**Description:** Implement in subclass to check if a signature is known.

Return True for a known signature, False for unknown.

### Function: remove_signature(self, digest, algorithm)

**Description:** Implement in subclass to delete a signature.

Should not raise if the signature is not stored.

### Function: close(self)

**Description:** Close any open connections this store may use.

If the store maintains any open connections (e.g. to a database),
they should be closed.

### Function: __init__(self)

**Description:** Initialize a memory signature store.

### Function: store_signature(self, digest, algorithm)

**Description:** Store a signature.

### Function: _maybe_cull(self)

**Description:** If more than cache_size signatures are stored, delete the oldest 25%

### Function: check_signature(self, digest, algorithm)

**Description:** Check a signature.

### Function: remove_signature(self, digest, algorithm)

**Description:** Remove a signature.

### Function: __init__(self, db_file)

**Description:** Initialize a sql signature store.

### Function: close(self)

**Description:** Close the db.

### Function: _connect_db(self, db_file)

### Function: init_db(self, db)

**Description:** Initialize the db.

### Function: store_signature(self, digest, algorithm)

**Description:** Store a signature in the db.

### Function: check_signature(self, digest, algorithm)

**Description:** Check a signature against the db.

### Function: remove_signature(self, digest, algorithm)

**Description:** Remove a signature from the db.

### Function: cull_db(self)

**Description:** Cull oldest 25% of the trusted signatures when the size limit is reached

### Function: _data_dir_default(self)

### Function: _store_factory_default(self)

### Function: _db_file_default(self)

### Function: _algorithm_changed(self, change)

### Function: _digestmod_default(self)

### Function: _secret_file_default(self)

### Function: _secret_default(self)

### Function: __init__(self)

**Description:** Initialize the notary.

### Function: _write_secret_file(self, secret)

**Description:** write my secret to my secret_file

### Function: compute_signature(self, nb)

**Description:** Compute a notebook's signature

by hashing the entire contents of the notebook via HMAC digest.

### Function: check_signature(self, nb)

**Description:** Check a notebook's stored signature

If a signature is stored in the notebook's metadata,
a new signature is computed and compared with the stored value.

Returns True if the signature is found and matches, False otherwise.

The following conditions must all be met for a notebook to be trusted:
- a signature is stored in the form 'scheme:hexdigest'
- the stored scheme matches the requested scheme
- the requested scheme is available from hashlib
- the computed hash from notebook_signature matches the stored hash

### Function: sign(self, nb)

**Description:** Sign a notebook, indicating that its output is trusted on this machine

Stores hash algorithm and hmac digest in a local database of trusted notebooks.

### Function: unsign(self, nb)

**Description:** Ensure that a notebook is untrusted

by removing its signature from the trusted database, if present.

### Function: mark_cells(self, nb, trusted)

**Description:** Mark cells as trusted if the notebook's signature can be verified

Sets ``cell.metadata.trusted = True | False`` on all code cells,
depending on the *trusted* parameter. This will typically be the return
value from ``self.check_signature(nb)``.

This function is the inverse of check_cells

### Function: _check_cell(self, cell, nbformat_version)

**Description:** Do we trust an individual cell?

Return True if:

- cell is explicitly trusted
- cell has no potentially unsafe rich output

If a cell has no output, or only simple print statements,
it will always be trusted.

### Function: check_cells(self, nb)

**Description:** Return whether all code cells are trusted.

A cell is trusted if the 'trusted' field in its metadata is truthy, or
if it has no potentially unsafe outputs.
If there are no code cells, return True.

This function is the inverse of mark_cells.

### Function: _config_file_name_default(self)

### Function: _notary_default(self)

### Function: sign_notebook_file(self, notebook_path)

**Description:** Sign a notebook from the filesystem

### Function: sign_notebook(self, nb, notebook_path)

**Description:** Sign a notebook that's been loaded

### Function: generate_new_key(self)

**Description:** Generate a new notebook signature key

### Function: start(self)

**Description:** Start the trust notebook app.

### Function: factory()
