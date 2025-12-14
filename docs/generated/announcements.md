## AI Summary

A file named announcements.py.


### Function: format_datetime(dt_str)

## Class: Notification

**Description:** Notification

Attributes:
    createdAt: Creation date
    message: Notification message
    modifiedAt: Modification date
    type: Notification type — ["default", "error", "info", "success", "warning"]
    link: Notification link button as a tuple (label, URL)
    options: Notification options

## Class: CheckForUpdateABC

**Description:** Abstract class to check for update.

Args:
    version: Current JupyterLab version

Attributes:
    version - str: Current JupyterLab version
    logger - logging.Logger: Server logger

## Class: CheckForUpdate

**Description:** Default class to check for update.

Args:
    version: Current JupyterLab version

Attributes:
    version - str: Current JupyterLab version
    logger - logging.Logger: Server logger

## Class: NeverCheckForUpdate

**Description:** Check update version that does nothing.

This is provided for administrators that want to
turn off requesting external resources.

Args:
    version: Current JupyterLab version

Attributes:
    version - str: Current JupyterLab version
    logger - logging.Logger: Server logger

## Class: CheckForUpdateHandler

**Description:** Check for Updates API handler.

Args:
    update_check: The class checking for a new version

## Class: NewsHandler

**Description:** News API handler.

Args:
    news_url: The Atom feed to fetch for news

### Function: __init__(self, version)

### Function: initialize(self, update_checker)

### Function: initialize(self, news_url)

### Function: build_entry(node)

### Function: get_xml_text(attr, default)
