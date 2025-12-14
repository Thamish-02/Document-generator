## AI Summary

A file named licenses_handler.py.


## Class: LicensesManager

**Description:** A manager for listing the licenses for all frontend end code distributed
by an application and any federated extensions

## Class: LicensesHandler

**Description:** A handler for serving licenses used by the application

### Function: federated_extensions(self)

**Description:** Lazily load the currrently-available federated extensions.

This is expensive, but probably the only way to be sure to get
up-to-date license information for extensions installed interactively.

### Function: report(self, report_format, bundles_pattern, full_text)

**Description:** create a human- or machine-readable report

### Function: report_json(self, bundles)

**Description:** create a JSON report
TODO: SPDX

### Function: report_csv(self, bundles)

**Description:** create a CSV report

### Function: report_markdown(self, bundles, full_text)

**Description:** create a markdown report

### Function: license_bundle(self, path, bundle)

**Description:** Return the content of a packages's license bundles

### Function: app_static_info(self)

**Description:** get the static directory for this app

This will usually be in `static_dir`, but may also appear in the
parent of `static_dir`.

### Function: bundles(self, bundles_pattern)

**Description:** Read all of the licenses
TODO: schema

### Function: initialize(self, manager)

**Description:** Initialize the handler.
