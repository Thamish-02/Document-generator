## AI Summary

A file named config_manager.py.


### Function: recursive_update(target, new)

**Description:** Recursively update one dictionary using another.

None values will delete their keys.

### Function: remove_defaults(data, defaults)

**Description:** Recursively remove items from dict that are already in defaults

## Class: BaseJSONConfigManager

**Description:** General JSON config manager

Deals with persisting/storing config in a json file with optionally
default values in a {section_name}.d directory.

### Function: ensure_config_dir_exists(self)

**Description:** Will try to create the config_dir directory.

### Function: file_name(self, section_name)

**Description:** Returns the json filename for the section_name: {config_dir}/{section_name}.json

### Function: directory(self, section_name)

**Description:** Returns the directory name for the section name: {config_dir}/{section_name}.d

### Function: get(self, section_name, include_root)

**Description:** Retrieve the config data for the specified section.

Returns the data as a dictionary, or an empty dictionary if the file
doesn't exist.

When include_root is False, it will not read the root .json file,
effectively returning the default values.

### Function: set(self, section_name, data)

**Description:** Store the given config data.

### Function: update(self, section_name, new_data)

**Description:** Modify the config section by recursively updating it with new_data.

Returns the modified config data as a dictionary.
