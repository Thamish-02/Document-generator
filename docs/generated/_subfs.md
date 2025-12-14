## AI Summary

A file named _subfs.py.


## Class: SubFS

**Description:** Maps a sub-directory of another filesystem.

## Class: ClosingSubFS

**Description:** Like SubFS, but auto-closes the parent filesystem when closed.

### Function: __init__(self, parent, sub_path)

### Function: delegate_fs(self)

### Function: _full(self, rel)

### Function: open(self, path, mode)

### Function: exists(self, path)

### Function: isdir(self, path)

### Function: isfile(self, path)

### Function: listdir(self, path)

### Function: makedir(self, path, recreate)

### Function: makedirs(self, path, recreate)

### Function: getinfo(self, path, namespaces)

### Function: remove(self, path)

### Function: removedir(self, path)

### Function: removetree(self, path)

### Function: movedir(self, src, dst, create)

### Function: getsyspath(self, path)

### Function: readbytes(self, path)

### Function: writebytes(self, path, data)

### Function: __repr__(self)

### Function: __str__(self)

### Function: close(self)
