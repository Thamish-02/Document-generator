## AI Summary

A file named completerlib.py.


### Function: module_list(path)

**Description:** Return the list containing the names of the modules available in the given
folder.

### Function: get_root_modules()

**Description:** Returns a list containing the names of all the modules available in the
folders of the pythonpath.

ip.db['rootmodules_cache'] maps sys.path entries to list of modules.

### Function: is_importable(module, attr, only_modules)

### Function: is_possible_submodule(module, attr)

### Function: try_import(mod, only_modules)

**Description:** Try to import given module and return list of potential completions.

### Function: quick_completer(cmd, completions)

**Description:** Easily create a trivial completer for a command.

Takes either a list of completions, or all completions in string (that will
be split on whitespace).

Example::

    [d:\ipython]|1> import ipy_completers
    [d:\ipython]|2> ipy_completers.quick_completer('foo', ['bar','baz'])
    [d:\ipython]|3> foo b<TAB>
    bar baz
    [d:\ipython]|3> foo ba

### Function: module_completion(line)

**Description:** Returns a list containing the completion possibilities for an import line.

The line looks like this :
'import xml.d'
'from xml.dom import'

### Function: module_completer(self, event)

**Description:** Give completions after user has typed 'import ...' or 'from ...'

### Function: magic_run_completer(self, event)

**Description:** Complete files that end in .py or .ipy or .ipynb for the %run command.
    

### Function: cd_completer(self, event)

**Description:** Completer function for cd, which only returns directories.

### Function: reset_completer(self, event)

**Description:** A completer for %reset magic

### Function: do_complete(self, event)
