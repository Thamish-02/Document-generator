## AI Summary

A file named serve.py.


## Class: ProxyHandler

**Description:** handler the proxies requests from a local prefix to a CDN

## Class: ServePostProcessor

**Description:** Post processor designed to serve files

Proxies reveal.js requests to a CDN if no local reveal.js is present

### Function: main(path)

**Description:** allow running this module to serve the slides

### Function: get(self, prefix, url)

**Description:** proxy a request to a CDN

### Function: postprocess(self, input)

**Description:** Serve the build directory with a webserver.
