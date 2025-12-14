## AI Summary

A file named alias.py.


### Function: default_aliases()

**Description:** Return list of shell aliases to auto-define.
    

## Class: AliasError

## Class: InvalidAliasError

## Class: Alias

**Description:** Callable object storing the details of one alias.

Instances are registered as magic functions to allow use of aliases.

## Class: AliasManager

### Function: __init__(self, shell, name, cmd)

### Function: validate(self)

**Description:** Validate the alias, and return the number of arguments.

### Function: __repr__(self)

### Function: __call__(self, rest)

### Function: __init__(self, shell)

### Function: init_aliases(self)

### Function: aliases(self)

### Function: soft_define_alias(self, name, cmd)

**Description:** Define an alias, but don't raise on an AliasError.

### Function: define_alias(self, name, cmd)

**Description:** Define a new alias after validating it.

This will raise an :exc:`AliasError` if there are validation
problems.

### Function: get_alias(self, name)

**Description:** Return an alias, or None if no alias by that name exists.

### Function: is_alias(self, name)

**Description:** Return whether or not a given name has been defined as an alias

### Function: undefine_alias(self, name)

### Function: clear_aliases(self)

### Function: retrieve_alias(self, name)

**Description:** Retrieve the command to which an alias expands.
