## AI Summary

A file named serverapp.py.


### Function: random_ports(port, n)

**Description:** Generate a list of n random ports near the given port.

The first 5 ports will be sequential, and the remaining n-5 will be
randomly selected in the range [port-2*n, port+2*n].

### Function: load_handlers(name)

**Description:** Load the (URL pattern, handler) tuples for each component.

## Class: ServerWebApplication

**Description:** A server web application.

### Function: _has_tornado_web_authenticated(method)

**Description:** Check if given method was decorated with @web.authenticated.

Note: it is ok if we reject on @authorized @web.authenticated
because the correct order is @web.authenticated @authorized.

## Class: JupyterPasswordApp

**Description:** Set a password for the Jupyter server.

Setting a password secures the Jupyter server
and removes the need for token-based authentication.

### Function: shutdown_server(server_info, timeout, log)

**Description:** Shutdown a Jupyter server in a separate process.

*server_info* should be a dictionary as produced by list_running_servers().

Will first try to request shutdown using /api/shutdown .
On Unix, if the server is still running after *timeout* seconds, it will
send SIGTERM. After another timeout, it escalates to SIGKILL.

Returns True if the server was stopped by any means, False if stopping it
failed (on Windows).

## Class: JupyterServerStopApp

**Description:** An application to stop a Jupyter server.

## Class: JupyterServerListApp

**Description:** An application to list running Jupyter servers.

## Class: ServerApp

**Description:** The Jupyter Server application class.

### Function: list_running_servers(runtime_dir, log)

**Description:** Iterate over the server info files of running Jupyter servers.

Given a runtime directory, find jpserver-* files in the security directory,
and yield dicts of their information, each one pertaining to
a currently running Jupyter server instance.

### Function: __init__(self, jupyter_app, default_services, kernel_manager, contents_manager, session_manager, kernel_spec_manager, config_manager, event_logger, extra_services, log, base_url, default_url, settings_overrides, jinja_env_options)

**Description:** Initialize a server web application.

### Function: add_handlers(self, host_pattern, host_handlers)

### Function: init_settings(self, jupyter_app, kernel_manager, contents_manager, session_manager, kernel_spec_manager, config_manager, event_logger, extra_services, log, base_url, default_url, settings_overrides, jinja_env_options)

**Description:** Initialize settings for the web application.

### Function: init_handlers(self, default_services, settings)

**Description:** Load the (URL pattern, handler) tuples for each component.

### Function: last_activity(self)

**Description:** Get a UTC timestamp for when the server last did something.

Includes: API activity, kernel activity, kernel shutdown, and terminal
activity.

### Function: _check_handler_auth(self, matcher, handler)

### Function: _config_file_default(self)

**Description:** the default config file.

### Function: start(self)

**Description:** Start the password app.

### Function: parse_command_line(self, argv)

**Description:** Parse command line options.

### Function: shutdown_server(self, server)

**Description:** Shut down a server.

### Function: _shutdown_or_exit(self, target_endpoint, server)

**Description:** Handle a shutdown.

### Function: _maybe_remove_unix_socket(socket_path)

**Description:** Try to remove a socket path.

### Function: start(self)

**Description:** Start the server stop app.

### Function: start(self)

**Description:** Start the server list application.

### Function: _default_log_level(self)

### Function: _default_log_format(self)

**Description:** override default log format to include date & time

### Function: _default_ip(self)

**Description:** Return localhost if available, 127.0.0.1 otherwise.

On some (horribly broken) systems, localhost cannot be bound.

### Function: _validate_ip(self, proposal)

### Function: _port_default(self)

### Function: _port_retries_default(self)

### Function: _validate_sock_mode(self, proposal)

### Function: _default_cookie_secret_file(self)

### Function: _default_cookie_secret(self)

### Function: _write_cookie_secret_file(self, secret)

**Description:** write my secret to my secret_file

### Function: _deprecated_token(self, change)

### Function: _deprecated_token_access(self)

### Function: _default_min_open_files_limit(self)

### Function: _warn_deprecated_config(self, change, clsname, new_name)

**Description:** Warn on deprecated config.

### Function: _deprecated_password(self, change)

### Function: _deprecated_password_config(self, change)

### Function: _allow_unauthenticated_access_default(self)

### Function: _default_allow_remote(self)

**Description:** Disallow remote access if we're listening only on loopback addresses

### Function: _deprecated_cookie_config(self, change)

### Function: _update_base_url(self, proposal)

### Function: static_file_path(self)

**Description:** return extra paths + the default location

### Function: _default_static_custom_path(self)

### Function: template_file_path(self)

**Description:** return extra paths + the default locations

### Function: _default_kernel_manager_class(self)

### Function: _default_session_manager_class(self)

### Function: _default_kernel_websocket_connection_class(self)

### Function: _default_kernel_spec_manager_class(self)

### Function: _default_info_file(self)

### Function: _default_browser_open_file(self)

### Function: _default_browser_open_file_to_run(self)

### Function: _update_pylab(self, change)

**Description:** when --pylab is specified, display a warning and exit

### Function: _update_notebook_dir(self, change)

### Function: _default_root_dir(self)

### Function: _normalize_dir(self, value)

**Description:** Normalize a directory.

### Function: _root_dir_validate(self, proposal)

### Function: _root_dir_changed(self, change)

### Function: _default_prefered_dir(self)

### Function: _preferred_dir_validate(self, proposal)

### Function: _update_server_extensions(self, change)

### Function: _deprecated_kernel_ws_protocol(self, change)

### Function: _deprecated_limit_rate(self, change)

### Function: _deprecated_iopub_msg_rate_limit(self, change)

### Function: _deprecated_iopub_data_rate_limit(self, change)

### Function: _deprecated_rate_limit_window(self, change)

### Function: _default_terminals_enabled(self)

### Function: starter_app(self)

**Description:** Get the Extension that started this server.

### Function: parse_command_line(self, argv)

**Description:** Parse the command line options.

### Function: init_configurables(self)

**Description:** Initialize configurables.

### Function: init_logging(self)

**Description:** Initialize logging.

### Function: init_event_logger(self)

**Description:** Initialize the Event Bus.

### Function: init_webapp(self)

**Description:** initialize tornado webapp

### Function: init_resources(self)

**Description:** initialize system resources

### Function: _get_urlparts(self, path, include_token)

**Description:** Constructs a urllib named tuple, ParseResult,
with default values set by server config.
The returned tuple can be manipulated using the `_replace` method.

### Function: public_url(self)

### Function: local_url(self)

### Function: display_url(self)

**Description:** Human readable string with URLs for interacting
with the running Jupyter Server

### Function: connection_url(self)

### Function: init_signal(self)

**Description:** Initialize signal handlers.

### Function: _handle_sigint(self, sig, frame)

**Description:** SIGINT handler spawns confirmation dialog

Note:
    JupyterHub replaces this method with _signal_stop
    in order to bypass the interactive prompt.
    https://github.com/jupyterhub/jupyterhub/pull/4864

### Function: _restore_sigint_handler(self)

**Description:** callback for restoring original SIGINT handler

### Function: _confirm_exit(self)

**Description:** confirm shutdown on ^C

A second ^C, or answering 'y' within 5s will cause shutdown,
otherwise original SIGINT handler will be restored.

This doesn't work on Windows.

### Function: _signal_stop(self, sig, frame)

**Description:** Handle a stop signal.

Note:
    JupyterHub configures this method to be called for SIGINT.
    https://github.com/jupyterhub/jupyterhub/pull/4864

### Function: _signal_info(self, sig, frame)

**Description:** Handle an info signal.

### Function: init_components(self)

**Description:** Check the components submodule, and warn if it's unclean

### Function: find_server_extensions(self)

**Description:** Searches Jupyter paths for jpserver_extensions.

### Function: init_server_extensions(self)

**Description:** If an extension's metadata includes an 'app' key,
the value must be a subclass of ExtensionApp. An instance
of the class will be created at this step. The config for
this instance will inherit the ServerApp's config object
and load its own config.

### Function: load_server_extensions(self)

**Description:** Load any extensions specified by config.

Import the module, then call the load_jupyter_server_extension function,
if one exists.

The extension API is experimental, and may change in future releases.

### Function: init_mime_overrides(self)

### Function: shutdown_no_activity(self)

**Description:** Shutdown server on timeout when there are no kernels or terminals.

### Function: init_shutdown_no_activity(self)

**Description:** Initialize a shutdown on no activity.

### Function: http_server(self)

**Description:** An instance of Tornado's HTTPServer class for the Server Web Application.

### Function: init_httpserver(self)

**Description:** Creates an instance of a Tornado HTTPServer for the Server Web Application
and sets the http_server attribute.

### Function: _bind_http_server(self)

**Description:** Bind our http server.

### Function: _bind_http_server_unix(self)

**Description:** Bind an http server on unix.

### Function: _bind_http_server_tcp(self)

**Description:** Bind a tcp server.

### Function: _find_http_port(self)

**Description:** Find an available http port.

### Function: _init_asyncio_patch()

**Description:** set default asyncio policy to be compatible with tornado

Tornado 6.0 is not compatible with default asyncio
ProactorEventLoop, which lacks basic *_reader methods.
Tornado 6.1 adds a workaround to add these methods in a thread,
but SelectorEventLoop should still be preferred
to avoid the extra thread for ~all of our events,
at least until asyncio adds *_reader methods
to proactor.

### Function: init_metrics(self)

**Description:** Initialize any prometheus metrics that need to be set up on server startup

### Function: initialize(self, argv, find_extensions, new_httpserver, starter_extension)

**Description:** Initialize the Server application class, configurables, web application, and http server.

Parameters
----------
argv : list or None
    CLI arguments to parse.
find_extensions : bool
    If True, find and load extensions listed in Jupyter config paths. If False,
    only load extensions that are passed to ServerApp directly through
    the `argv`, `config`, or `jpserver_extensions` arguments.
new_httpserver : bool
    If True, a tornado HTTPServer instance will be created and configured for the Server Web
    Application. This will set the http_server attribute of this class.
starter_extension : str
    If given, it references the name of an extension point that started the Server.
    We will try to load configuration from extension point

### Function: running_server_info(self, kernel_count)

**Description:** Return the current working directory and the server url information

### Function: server_info(self)

**Description:** Return a JSONable dict of information about this server.

### Function: write_server_info_file(self)

**Description:** Write the result of server_info() to the JSON file info_file.

### Function: remove_server_info_file(self)

**Description:** Remove the jpserver-<pid>.json file created for this server.

Ignores the error raised when the file has already been removed.

### Function: _resolve_file_to_run_and_root_dir(self)

**Description:** Returns a relative path from file_to_run
to root_dir. If root_dir and file_to_run
are incompatible, i.e. on different subtrees,
crash the app and log a critical message. Note
that if root_dir is not configured and file_to_run
is configured, root_dir will be set to the parent
directory of file_to_run.

### Function: _write_browser_open_file(self, url, fh)

**Description:** Write the browser open file.

### Function: write_browser_open_files(self)

**Description:** Write an `browser_open_file` and `browser_open_file_to_run` files

This can be used to open a file directly in a browser.

### Function: write_browser_open_file(self)

**Description:** Write an jpserver-<pid>-open.html file

This can be used to open the notebook in a browser

### Function: remove_browser_open_files(self)

**Description:** Remove the `browser_open_file` and `browser_open_file_to_run` files
created for this server.

Ignores the error raised when the file has already been removed.

### Function: remove_browser_open_file(self)

**Description:** Remove the jpserver-<pid>-open.html file created for this server.

Ignores the error raised when the file has already been removed.

### Function: _prepare_browser_open(self)

**Description:** Prepare to open the browser.

### Function: launch_browser(self)

**Description:** Launch the browser.

### Function: start_app(self)

**Description:** Start the Jupyter Server application.

### Function: start_ioloop(self)

**Description:** Start the IO Loop.

### Function: init_ioloop(self)

**Description:** init self.io_loop so that an extension can use it by io_loop.call_later() to create background tasks

### Function: start(self)

**Description:** Start the Jupyter server app, after initialization

This method takes no arguments so all configuration and initialization
must be done prior to calling this method.

### Function: stop(self, from_signal)

**Description:** Cleanup resources and stop the server.

### Function: target()
