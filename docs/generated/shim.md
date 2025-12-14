## AI Summary

A file named shim.py.


### Function: NBAPP_AND_SVAPP_SHIM_MSG(trait_name)

### Function: NBAPP_TO_SVAPP_SHIM_MSG(trait_name)

### Function: EXTAPP_AND_NBAPP_AND_SVAPP_SHIM_MSG(trait_name, extapp_name)

### Function: EXTAPP_AND_SVAPP_SHIM_MSG(trait_name, extapp_name)

### Function: EXTAPP_AND_NBAPP_SHIM_MSG(trait_name, extapp_name)

### Function: NOT_EXTAPP_NBAPP_AND_SVAPP_SHIM_MSG(trait_name, extapp_name)

### Function: EXTAPP_TO_SVAPP_SHIM_MSG(trait_name, extapp_name)

### Function: EXTAPP_TO_NBAPP_SHIM_MSG(trait_name, extapp_name)

## Class: NotebookConfigShimMixin

**Description:** A Mixin class for shimming configuration from
NotebookApp to ServerApp. This class handles warnings, errors,
etc.

This class should be used during a transition period for apps
that are switching from depending on NotebookApp to ServerApp.

After one release cycle, this class can be safely removed
from the inheriting class.

TL;DR

The entry point to shimming is at the `update_config` method.
Once traits are loaded, before updating config across all
configurable objects, this class injects a method to reroute
traits to their *most logical* classes.

This class raises warnings when:
    1. a trait has moved.
    2. a trait is redundant across classes.

Redundant traits across multiple classes now must be
configured separately, *or* removed from their old
location to avoid this warning.

For a longer description on how individual traits are handled,
read the docstring under `shim_config_from_notebook_to_jupyter_server`.

### Function: update_config(self, config)

### Function: shim_config_from_notebook_to_jupyter_server(self, config)

**Description:** Reorganizes a config object to reroute traits to their expected destinations
after the transition from NotebookApp to ServerApp.

A detailed explanation of how traits are handled:

1. If the argument is prefixed with `ServerApp`,
    pass this trait to `ServerApp`.
2. If the argument is prefixed with `NotebookApp`,
    * If the argument is a trait of `NotebookApp` *and* `ServerApp`:
        1. Raise a warning—**for the extension developers**—that
            there's redundant traits.
        2. Pass trait to `NotebookApp`.
    * If the argument is a trait of just `ServerApp` only
        (i.e. the trait moved from `NotebookApp` to `ServerApp`):
        1. Raise a "this trait has moved" **for the user**.
        3. Pass trait to `ServerApp`.
    * If the argument is a trait of `NotebookApp` only, pass trait
        to `NotebookApp`.
    * If the argument is not found in any object, raise a
        `"Trait not found."` error.
3. If the argument is prefixed with `ExtensionApp`:
    * If the argument is a trait of `ExtensionApp`,
        `NotebookApp`, and `ServerApp`,
        1. Raise a warning about redundancy.
        2. Pass to the ExtensionApp
    * If the argument is a trait of `ExtensionApp` and `NotebookApp`,
        1. Raise a warning about redundancy.
        2. Pass to ExtensionApp.
    * If the argument is a trait of `ExtensionApp` and `ServerApp`,
        1. Raise a warning about redundancy.
        2. Pass to ExtensionApp.
    * If the argument is a trait of `ExtensionApp`.
        1. Pass to ExtensionApp.
    * If the argument is a trait of `NotebookApp` but not `ExtensionApp`,
        1. Raise a warning that trait has likely moved to NotebookApp.
        2. Pass to NotebookApp
    * If the arguent is a trait of `ServerApp` but not `ExtensionApp`,
        1. Raise a warning that the trait has likely moved to ServerApp.
        2. Pass to ServerApp.
    * else
        * Raise a TraitError: "trait not found."
