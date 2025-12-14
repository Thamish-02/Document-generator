## AI Summary

A file named commands.py.


## Class: ProgressProcess

### Function: pjoin()

**Description:** Join paths to create a real path.

### Function: get_user_settings_dir()

**Description:** Get the configured JupyterLab user settings directory.

### Function: get_workspaces_dir()

**Description:** Get the configured JupyterLab workspaces directory.

### Function: get_app_dir()

**Description:** Get the configured JupyterLab app directory.

### Function: dedupe_yarn(path, logger)

**Description:** `yarn-deduplicate` with the `fewer` strategy to minimize total
packages installed in a given staging directory

This means a extension (or dependency) _could_ cause a downgrade of an
version expected at publication time, but core should aggressively set
pins above, for example, known-bad versions

### Function: ensure_node_modules(cwd, logger)

**Description:** Ensure that node_modules is up to date.

Returns true if the node_modules was updated.

### Function: ensure_dev(logger)

**Description:** Ensure that the dev assets are available.

### Function: ensure_core(logger)

**Description:** Ensure that the core assets are available.

### Function: ensure_app(app_dir)

**Description:** Ensure that an application directory is available.

If it does not exist, return a list of messages to prompt the user.

### Function: watch_packages(logger)

**Description:** Run watch mode for the source packages.

Parameters
----------
logger: :class:`~logger.Logger`, optional
    The logger instance.

Returns
-------
A list of `WatchHelper` objects.

### Function: watch_dev(logger)

**Description:** Run watch mode in a given directory.

Parameters
----------
logger: :class:`~logger.Logger`, optional
    The logger instance.

Returns
-------
A list of `WatchHelper` objects.

## Class: AppOptions

**Description:** Options object for build system

### Function: _ensure_options(options)

**Description:** Helper to use deprecated kwargs for AppOption

### Function: watch(app_options)

**Description:** Watch the application.

Parameters
----------
app_options: :class:`AppOptions`, optional
    The application options.

Returns
-------
A list of processes to run asynchronously.

### Function: install_extension(extension, app_options, pin)

**Description:** Install an extension package into JupyterLab.

The extension is first validated.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: uninstall_extension(name, app_options, all_)

**Description:** Uninstall an extension by name or path.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: update_extension(name, all_, app_dir, app_options)

**Description:** Update an extension by name, or all extensions.
Either `name` must be given as a string, or `all_` must be `True`.
If `all_` is `True`, the value of `name` is ignored.
Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: clean(app_options)

**Description:** Clean the JupyterLab application directory.

### Function: build(name, version, static_url, kill_event, clean_staging, app_options, production, minimize)

**Description:** Build the JupyterLab application.

### Function: get_app_info(app_options)

**Description:** Get a dictionary of information about the app.

### Function: enable_extension(extension, app_options, level)

**Description:** Enable a JupyterLab extension/plugin.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: disable_extension(extension, app_options, level)

**Description:** Disable a JupyterLab extension/plugin.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: check_extension(extension, installed, app_options)

**Description:** Check if a JupyterLab extension is enabled or disabled.

### Function: lock_extension(extension, app_options, level)

**Description:** Lock a JupyterLab extension/plugin.

### Function: unlock_extension(extension, app_options, level)

**Description:** Unlock a JupyterLab extension/plugin.

### Function: build_check(app_options)

**Description:** Determine whether JupyterLab should be built.

Returns a list of messages.

### Function: list_extensions(app_options)

**Description:** List the extensions.

### Function: link_package(path, app_options)

**Description:** Link a package against the JupyterLab build.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: unlink_package(package, app_options)

**Description:** Unlink a package from JupyterLab by path or name.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: get_app_version(app_options)

**Description:** Get the application version.

### Function: get_latest_compatible_package_versions(names, app_options)

**Description:** Get the latest compatible version of a list of packages.

### Function: read_package(target)

**Description:** Read the package data in a given target tarball.

## Class: _AppHandler

### Function: _node_check(logger)

**Description:** Check for the existence of nodejs with the correct version.

### Function: _yarn_config(logger)

**Description:** Get the yarn configuration.

Returns
-------
{"yarn config": dict, "npm config": dict}
if unsuccessful, the subdictionaries are empty

### Function: _ensure_logger(logger)

**Description:** Ensure that we have a logger

### Function: _normalize_path(extension)

**Description:** Normalize a given extension if it is a path.

### Function: _rmtree(path, logger)

**Description:** Remove a tree, logging errors

### Function: _unlink(path, logger)

**Description:** Remove a file, logging errors

### Function: _rmtree_star(path, logger)

**Description:** Remove all files/trees within a dir, logging errors

### Function: _validate_extension(data)

**Description:** Detect if a package is an extension using its metadata.

Returns any problems it finds.

### Function: _tarsum(input_file)

**Description:** Compute the recursive sha sum of a tar file.

### Function: _get_static_data(app_dir)

**Description:** Get the data for the app static dir.

### Function: _validate_compatibility(extension, deps, core_data)

**Description:** Validate the compatibility of an extension.

### Function: _test_overlap(spec1, spec2, drop_prerelease1, drop_prerelease2)

**Description:** Test whether two version specs overlap.

Returns `None` if we cannot determine compatibility,
otherwise whether there is an overlap

### Function: _compare_ranges(spec1, spec2, drop_prerelease1, drop_prerelease2)

**Description:** Test whether two version specs overlap.

Returns `None` if we cannot determine compatibility,
otherwise return 0 if there is an overlap, 1 if
spec1 is lower/older than spec2, and -1 if spec1
is higher/newer than spec2.

### Function: _is_disabled(name, disabled)

**Description:** Test whether the package is disabled.

## Class: LockStatus

### Function: _is_locked(name, locked)

**Description:** Test whether the package is locked.

If only a subset of extension plugins is locked return them.

### Function: _format_compatibility_errors(name, version, errors)

**Description:** Format a message for compatibility errors.

### Function: _log_multiple_compat_errors(logger, errors_map, verbose)

**Description:** Log compatibility errors for multiple extensions at once

### Function: _log_single_compat_errors(logger, name, version, errors)

**Description:** Log compatibility errors for a single extension

### Function: _compat_error_age(errors)

**Description:** Compare all incompatibilities for an extension.

Returns a number > 0 if all extensions are older than that supported by lab.
Returns a number < 0 if all extensions are newer than that supported by lab.
Returns 0 otherwise (i.e. a mix).

### Function: _get_core_extensions(core_data)

**Description:** Get the core extensions.

### Function: _semver_prerelease_key(prerelease)

**Description:** Sort key for prereleases.

Precedence for two pre-release versions with the same
major, minor, and patch version MUST be determined by
comparing each dot separated identifier from left to
right until a difference is found as follows:
identifiers consisting of only digits are compare
numerically and identifiers with letters or hyphens
are compared lexically in ASCII sort order. Numeric
identifiers always have lower precedence than non-
numeric identifiers. A larger set of pre-release
fields has a higher precedence than a smaller set,
if all of the preceding identifiers are equal.

### Function: _semver_key(version, prerelease_first)

**Description:** A sort key-function for sorting semver version string.

The default sorting order is ascending (0.x -> 1.x -> 2.x).

If `prerelease_first`, pre-releases will come before
ALL other semver keys (not just those with same version).
I.e (1.0-pre, 2.0-pre -> 0.x -> 1.x -> 2.x).

Otherwise it will sort in the standard way that it simply
comes before any release with shared version string
(0.x -> 1.0-pre -> 1.x -> 2.0-pre -> 2.x).

### Function: _fetch_package_metadata(registry, name, logger)

**Description:** Fetch the metadata for a package from the npm registry

### Function: __init__(self, cmd, logger, cwd, kill_event, env)

**Description:** Start a subprocess that can be run asynchronously.

Parameters
----------
cmd: list
    The command to run.
logger: :class:`~logger.Logger`, optional
    The logger instance.
cwd: string, optional
    The cwd of the process.
kill_event: :class:`~threading.Event`, optional
    An event used to kill the process operation.
env: dict, optional
    The environment for the process.

### Function: wait(self)

### Function: __init__(self, logger, core_config)

### Function: _default_logger(self)

### Function: _default_app_dir(self)

### Function: _default_core_config(self)

### Function: _default_registry(self)

### Function: __init__(self, options)

**Description:** Create a new _AppHandler object

### Function: install_extension(self, extension, existing, pin)

**Description:** Install an extension package into JupyterLab.

The extension is first validated.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: build(self, name, version, static_url, clean_staging, production, minimize)

**Description:** Build the application.

### Function: watch(self)

**Description:** Start the application watcher and then run the watch in
the background.

### Function: list_extensions(self)

**Description:** Print an output of the extensions.

### Function: build_check(self, fast)

**Description:** Determine whether JupyterLab should be built.

Returns a list of messages.

### Function: uninstall_extension(self, name)

**Description:** Uninstall an extension by name.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: uninstall_all_extensions(self)

**Description:** Uninstalls all extensions

Returns `True` if a rebuild is recommended, `False` otherwise

### Function: update_all_extensions(self)

**Description:** Update all non-local extensions.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: update_extension(self, name)

**Description:** Update an extension by name.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: _update_extension(self, name)

**Description:** Update an extension by name.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: link_package(self, path)

**Description:** Link a package at the given path.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: unlink_package(self, path)

**Description:** Unlink a package by name or at the given path.

A ValueError is raised if the path is not an unlinkable package.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: _is_extension_locked(self, extension, level, include_higher_levels)

### Function: toggle_extension(self, extension, value, level)

**Description:** Enable or disable a lab extension.

Returns `True` if a rebuild is recommended, `False` otherwise.

### Function: _maybe_mirror_disabled_in_locked(self, level)

**Description:** Lock all extensions that were previously disabled.

This exists to facilitate migration from 4.0 (which did not include lock
function) to 4.1 which exposes the plugin management to users in UI.

Returns `True` if migration happened, `False` otherwise.

### Function: toggle_extension_lock(self, extension, value, level)

**Description:** Lock or unlock a lab extension (/plugin).

### Function: check_extension(self, extension, check_installed_only)

**Description:** Check if a lab extension is enabled or disabled

### Function: _check_core_extension(self, extension, info, check_installed_only)

**Description:** Check if a core extension is enabled or disabled

### Function: _check_common_extension(self, extension, info, check_installed_only)

**Description:** Check if a common (non-core) extension is enabled or disabled

### Function: _get_app_info(self)

**Description:** Get information about the app.

### Function: _ensure_disabled_info(self)

### Function: _populate_staging(self, name, version, static_url, clean)

**Description:** Set up the assets in the staging directory.

### Function: _get_package_template(self, silent)

**Description:** Get the template the for staging package.json file.

### Function: _check_local(self, name, source, dname)

**Description:** Check if a local package has changed.

`dname` is the directory name of existing package tar archives.

### Function: _update_local(self, name, source, dname, data, dtype)

**Description:** Update a local dependency.  Return `True` if changed.

### Function: _get_extensions(self, core_data)

**Description:** Get the extensions for the application.

### Function: _get_extensions_in_dir(self, dname, core_data)

**Description:** Get the extensions in a given directory.

### Function: _get_extension_compat(self)

**Description:** Get the extension compatibility info.

### Function: _get_local_extensions(self)

**Description:** Get the locally installed extensions.

### Function: _get_linked_packages(self)

**Description:** Get the linked packages.

### Function: _get_uninstalled_core_extensions(self)

**Description:** Get the uninstalled core extensions.

### Function: _ensure_app_dirs(self)

**Description:** Ensure that the application directories exist

### Function: _list_extensions(self, info, ext_type)

**Description:** List the extensions of a given type.

### Function: _list_federated_extensions(self)

### Function: _compose_extra_status(self, name, info, data, errors)

### Function: _read_build_config(self)

**Description:** Get the build config data for the app dir.

### Function: _write_build_config(self, config)

**Description:** Write the build config to the app dir.

### Function: _get_local_data(self, source)

**Description:** Get the local data for extensions or linked packages.

### Function: _install_extension(self, extension, tempdir, pin)

**Description:** Install an extension with validation and return the name and path.

### Function: _extract_package(self, source, tempdir, pin)

**Description:** Call `npm pack` for an extension.

The pack command will download the package tar if `source` is
a package name, or run `npm pack` locally if `source` is a
directory.

### Function: _latest_compatible_package_version(self, name)

**Description:** Get the latest compatible version of a package

### Function: latest_compatible_package_versions(self, names)

**Description:** Get the latest compatible versions of several packages

Like _latest_compatible_package_version, but optimized for
retrieving the latest version for several packages in one go.

### Function: _format_no_compatible_package_version(self, name)

**Description:** Get the latest compatible version of a package

### Function: _run(self, cmd)

**Description:** Run the command using our logger and abort callback.

Returns the exit code.

### Function: onerror()

### Function: format_path(path)

### Function: sort_key(key_value)

### Function: noop(x, y, z)

### Function: sort_key(key_value)

### Function: sort_key(key_value)
