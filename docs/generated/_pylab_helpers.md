## AI Summary

A file named _pylab_helpers.py.


## Class: Gcf

**Description:** Singleton to maintain the relation between figures and their managers, and
keep track of and "active" figure and manager.

The canvas of a figure created through pyplot is associated with a figure
manager, which handles the interaction between the figure and the backend.
pyplot keeps track of figure managers using an identifier, the "figure
number" or "manager number" (which can actually be any hashable value);
this number is available as the :attr:`number` attribute of the manager.

This class is never instantiated; it consists of an `OrderedDict` mapping
figure/manager numbers to managers, and a set of class methods that
manipulate this `OrderedDict`.

Attributes
----------
figs : OrderedDict
    `OrderedDict` mapping numbers to managers; the active manager is at the
    end.

### Function: get_fig_manager(cls, num)

**Description:** If manager number *num* exists, make it the active one and return it;
otherwise return *None*.

### Function: destroy(cls, num)

**Description:** Destroy manager *num* -- either a manager instance or a manager number.

In the interactive backends, this is bound to the window "destroy" and
"delete" events.

It is recommended to pass a manager instance, to avoid confusion when
two managers share the same number.

### Function: destroy_fig(cls, fig)

**Description:** Destroy figure *fig*.

### Function: destroy_all(cls)

**Description:** Destroy all figures.

### Function: has_fignum(cls, num)

**Description:** Return whether figure number *num* exists.

### Function: get_all_fig_managers(cls)

**Description:** Return a list of figure managers.

### Function: get_num_fig_managers(cls)

**Description:** Return the number of figures being managed.

### Function: get_active(cls)

**Description:** Return the active manager, or *None* if there is no manager.

### Function: _set_new_active_manager(cls, manager)

**Description:** Adopt *manager* into pyplot and make it the active manager.

### Function: set_active(cls, manager)

**Description:** Make *manager* the active manager.

### Function: draw_all(cls, force)

**Description:** Redraw all stale managed figures, or, if *force* is True, all managed
figures.
