## AI Summary

A file named macRes.py.


## Class: ResourceError

## Class: ResourceReader

**Description:** Reader for Mac OS resource forks.

Parses a resource fork and returns resources according to their type.
If run on OS X, this will open the resource fork in the filesystem.
Otherwise, it will open the file itself and attempt to read it as
though it were a resource fork.

The returned object can be indexed by type and iterated over,
returning in each case a list of py:class:`Resource` objects
representing all the resources of a certain type.

## Class: Resource

**Description:** Represents a resource stored within a resource fork.

Attributes:
        type: resource type.
        data: resource data.
        id: ID.
        name: resource name.
        attr: attributes.

### Function: __init__(self, fileOrPath)

**Description:** Open a file

Args:
        fileOrPath: Either an object supporting a ``read`` method, an
                ``os.PathLike`` object, or a string.

### Function: openResourceFork(path)

### Function: openDataFork(path)

### Function: _readFile(self)

### Function: _read(self, numBytes, offset)

### Function: _readHeaderAndMap(self)

### Function: _readTypeList(self)

### Function: _readReferenceList(self, resType, refListOffset, numRes)

### Function: __getitem__(self, resType)

### Function: __delitem__(self, resType)

### Function: __setitem__(self, resType, resources)

### Function: __len__(self)

### Function: __iter__(self)

### Function: keys(self)

### Function: types(self)

**Description:** A list of the types of resources in the resource fork.

### Function: countResources(self, resType)

**Description:** Return the number of resources of a given type.

### Function: getIndices(self, resType)

**Description:** Returns a list of indices of resources of a given type.

### Function: getNames(self, resType)

**Description:** Return list of names of all resources of a given type.

### Function: getIndResource(self, resType, index)

**Description:** Return resource of given type located at an index ranging from 1
to the number of resources for that type, or None if not found.

### Function: getNamedResource(self, resType, name)

**Description:** Return the named resource of given type, else return None.

### Function: close(self)

### Function: __init__(self, resType, resData, resID, resName, resAttr)

### Function: decompile(self, refData, reader)
