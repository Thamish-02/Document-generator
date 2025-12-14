## AI Summary

A file named _osfs.py.


## Class: OSFS

**Description:** Filesystem for a directory on the local disk.

A thin layer on top of `pathlib.Path`.

### Function: __init__(self, root, create)

### Function: _abs(self, rel_path)

### Function: open(self, path, mode)

### Function: exists(self, path)

### Function: isdir(self, path)

### Function: isfile(self, path)

### Function: listdir(self, path)

### Function: _mkdir(self, path, parents, exist_ok)

### Function: makedir(self, path, recreate)

### Function: makedirs(self, path, recreate)

### Function: getinfo(self, path, namespaces)

### Function: remove(self, path)

### Function: removedir(self, path)

### Function: removetree(self, path)

### Function: movedir(self, src_dir, dst_dir, create)

### Function: getsyspath(self, path)

### Function: __repr__(self)

### Function: __str__(self)
