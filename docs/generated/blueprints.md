## AI Summary

A file named blueprints.py.


## Class: BlueprintSetupState

**Description:** Temporary holder object for registering a blueprint with the
application.  An instance of this class is created by the
:meth:`~flask.Blueprint.make_setup_state` method and later passed
to all register callback functions.

## Class: Blueprint

**Description:** Represents a blueprint, a collection of routes and other
app-related functions that can be registered on a real application
later.

A blueprint is an object that allows defining application functions
without requiring an application object ahead of time. It uses the
same decorators as :class:`~flask.Flask`, but defers the need for an
application by recording them for later registration.

Decorating a function with a blueprint creates a deferred function
that is called with :class:`~flask.blueprints.BlueprintSetupState`
when the blueprint is registered on an application.

See :doc:`/blueprints` for more information.

:param name: The name of the blueprint. Will be prepended to each
    endpoint name.
:param import_name: The name of the blueprint package, usually
    ``__name__``. This helps locate the ``root_path`` for the
    blueprint.
:param static_folder: A folder with static files that should be
    served by the blueprint's static route. The path is relative to
    the blueprint's root path. Blueprint static files are disabled
    by default.
:param static_url_path: The url to serve static files from.
    Defaults to ``static_folder``. If the blueprint does not have
    a ``url_prefix``, the app's static route will take precedence,
    and the blueprint's static files won't be accessible.
:param template_folder: A folder with templates that should be added
    to the app's template search path. The path is relative to the
    blueprint's root path. Blueprint templates are disabled by
    default. Blueprint templates have a lower precedence than those
    in the app's templates folder.
:param url_prefix: A path to prepend to all of the blueprint's URLs,
    to make them distinct from the rest of the app's routes.
:param subdomain: A subdomain that blueprint routes will match on by
    default.
:param url_defaults: A dict of default values that blueprint routes
    will receive by default.
:param root_path: By default, the blueprint will automatically set
    this based on ``import_name``. In certain situations this
    automatic detection can fail, so the path can be specified
    manually instead.

.. versionchanged:: 1.1.0
    Blueprints have a ``cli`` group to register nested CLI commands.
    The ``cli_group`` parameter controls the name of the group under
    the ``flask`` command.

.. versionadded:: 0.7

### Function: __init__(self, blueprint, app, options, first_registration)

### Function: add_url_rule(self, rule, endpoint, view_func)

**Description:** A helper method to register a rule (and optionally a view function)
to the application.  The endpoint is automatically prefixed with the
blueprint's name.

### Function: __init__(self, name, import_name, static_folder, static_url_path, template_folder, url_prefix, subdomain, url_defaults, root_path, cli_group)

### Function: _check_setup_finished(self, f_name)

### Function: record(self, func)

**Description:** Registers a function that is called when the blueprint is
registered on the application.  This function is called with the
state as argument as returned by the :meth:`make_setup_state`
method.

### Function: record_once(self, func)

**Description:** Works like :meth:`record` but wraps the function in another
function that will ensure the function is only called once.  If the
blueprint is registered a second time on the application, the
function passed is not called.

### Function: make_setup_state(self, app, options, first_registration)

**Description:** Creates an instance of :meth:`~flask.blueprints.BlueprintSetupState`
object that is later passed to the register callback functions.
Subclasses can override this to return a subclass of the setup state.

### Function: register_blueprint(self, blueprint)

**Description:** Register a :class:`~flask.Blueprint` on this blueprint. Keyword
arguments passed to this method will override the defaults set
on the blueprint.

.. versionchanged:: 2.0.1
    The ``name`` option can be used to change the (pre-dotted)
    name the blueprint is registered with. This allows the same
    blueprint to be registered multiple times with unique names
    for ``url_for``.

.. versionadded:: 2.0

### Function: register(self, app, options)

**Description:** Called by :meth:`Flask.register_blueprint` to register all
views and callbacks registered on the blueprint with the
application. Creates a :class:`.BlueprintSetupState` and calls
each :meth:`record` callback with it.

:param app: The application this blueprint is being registered
    with.
:param options: Keyword arguments forwarded from
    :meth:`~Flask.register_blueprint`.

.. versionchanged:: 2.3
    Nested blueprints now correctly apply subdomains.

.. versionchanged:: 2.1
    Registering the same blueprint with the same name multiple
    times is an error.

.. versionchanged:: 2.0.1
    Nested blueprints are registered with their dotted name.
    This allows different blueprints with the same name to be
    nested at different locations.

.. versionchanged:: 2.0.1
    The ``name`` option can be used to change the (pre-dotted)
    name the blueprint is registered with. This allows the same
    blueprint to be registered multiple times with unique names
    for ``url_for``.

### Function: _merge_blueprint_funcs(self, app, name)

### Function: add_url_rule(self, rule, endpoint, view_func, provide_automatic_options)

**Description:** Register a URL rule with the blueprint. See :meth:`.Flask.add_url_rule` for
full documentation.

The URL rule is prefixed with the blueprint's URL prefix. The endpoint name,
used with :func:`url_for`, is prefixed with the blueprint's name.

### Function: app_template_filter(self, name)

**Description:** Register a template filter, available in any template rendered by the
application. Equivalent to :meth:`.Flask.template_filter`.

:param name: the optional name of the filter, otherwise the
             function name will be used.

### Function: add_app_template_filter(self, f, name)

**Description:** Register a template filter, available in any template rendered by the
application. Works like the :meth:`app_template_filter` decorator. Equivalent to
:meth:`.Flask.add_template_filter`.

:param name: the optional name of the filter, otherwise the
             function name will be used.

### Function: app_template_test(self, name)

**Description:** Register a template test, available in any template rendered by the
application. Equivalent to :meth:`.Flask.template_test`.

.. versionadded:: 0.10

:param name: the optional name of the test, otherwise the
             function name will be used.

### Function: add_app_template_test(self, f, name)

**Description:** Register a template test, available in any template rendered by the
application. Works like the :meth:`app_template_test` decorator. Equivalent to
:meth:`.Flask.add_template_test`.

.. versionadded:: 0.10

:param name: the optional name of the test, otherwise the
             function name will be used.

### Function: app_template_global(self, name)

**Description:** Register a template global, available in any template rendered by the
application. Equivalent to :meth:`.Flask.template_global`.

.. versionadded:: 0.10

:param name: the optional name of the global, otherwise the
             function name will be used.

### Function: add_app_template_global(self, f, name)

**Description:** Register a template global, available in any template rendered by the
application. Works like the :meth:`app_template_global` decorator. Equivalent to
:meth:`.Flask.add_template_global`.

.. versionadded:: 0.10

:param name: the optional name of the global, otherwise the
             function name will be used.

### Function: before_app_request(self, f)

**Description:** Like :meth:`before_request`, but before every request, not only those handled
by the blueprint. Equivalent to :meth:`.Flask.before_request`.

### Function: after_app_request(self, f)

**Description:** Like :meth:`after_request`, but after every request, not only those handled
by the blueprint. Equivalent to :meth:`.Flask.after_request`.

### Function: teardown_app_request(self, f)

**Description:** Like :meth:`teardown_request`, but after every request, not only those
handled by the blueprint. Equivalent to :meth:`.Flask.teardown_request`.

### Function: app_context_processor(self, f)

**Description:** Like :meth:`context_processor`, but for templates rendered by every view, not
only by the blueprint. Equivalent to :meth:`.Flask.context_processor`.

### Function: app_errorhandler(self, code)

**Description:** Like :meth:`errorhandler`, but for every request, not only those handled by
the blueprint. Equivalent to :meth:`.Flask.errorhandler`.

### Function: app_url_value_preprocessor(self, f)

**Description:** Like :meth:`url_value_preprocessor`, but for every request, not only those
handled by the blueprint. Equivalent to :meth:`.Flask.url_value_preprocessor`.

### Function: app_url_defaults(self, f)

**Description:** Like :meth:`url_defaults`, but for every request, not only those handled by
the blueprint. Equivalent to :meth:`.Flask.url_defaults`.

### Function: wrapper(state)

### Function: extend(bp_dict, parent_dict)

### Function: decorator(f)

### Function: register_template(state)

### Function: decorator(f)

### Function: register_template(state)

### Function: decorator(f)

### Function: register_template(state)

### Function: decorator(f)

### Function: from_blueprint(state)
