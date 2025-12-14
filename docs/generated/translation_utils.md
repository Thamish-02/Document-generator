## AI Summary

A file named translation_utils.py.


### Function: _get_default_schema_selectors()

### Function: _prepare_schema_patterns(schema)

### Function: _get_installed_language_pack_locales()

**Description:** Get available installed language pack locales.

Returns
-------
tuple
    A tuple, where the first item is the result and the second item any
    error messages.

### Function: _get_installed_package_locales()

**Description:** Get available installed packages containing locale information.

Returns
-------
tuple
    A tuple, where the first item is the result and the second item any
    error messages. The value for the key points to the root location
    the package.

### Function: is_valid_locale(locale_)

**Description:** Check if a `locale_` value is valid.

Parameters
----------
locale_: str
    Language locale code.

Notes
-----
A valid locale is in the form language (See ISO-639 standard) and an
optional territory (See ISO-3166 standard).

Examples of valid locales:
- English: DEFAULT_LOCALE
- Australian English: "en_AU"
- Portuguese: "pt"
- Brazilian Portuguese: "pt_BR"

Examples of invalid locales:
- Australian Spanish: "es_AU"
- Brazilian German: "de_BR"

### Function: get_display_name(locale_, display_locale)

**Description:** Return the language name to use with a `display_locale` for a given language locale.

Parameters
----------
locale_: str
    The language name to use.
display_locale: str, optional
    The language to display the `locale_`.

Returns
-------
str
    Localized `locale_` and capitalized language name using `display_locale` as language.

### Function: merge_locale_data(language_pack_locale_data, package_locale_data)

**Description:** Merge language pack data with locale data bundled in packages.

Parameters
----------
language_pack_locale_data: dict
    The dictionary with language pack locale data.
package_locale_data: dict
    The dictionary with package locale data.

Returns
-------
dict
    Merged locale data.

### Function: get_installed_packages_locale(locale_)

**Description:** Get all jupyterlab extensions installed that contain locale data.

Returns
-------
tuple
    A tuple in the form `(locale_data_dict, message)`,
    where the `locale_data_dict` is an ordered list
    of available language packs:
        >>> {"package-name": locale_data, ...}

Examples
--------
- `entry_points={"jupyterlab.locale": "package-name = package_module"}`
- `entry_points={"jupyterlab.locale": "jupyterlab-git = jupyterlab_git"}`

### Function: get_language_packs(display_locale)

**Description:** Return the available language packs installed in the system.

The returned information contains the languages displayed in the current
locale.

Parameters
----------
display_locale: str, optional
    Default is DEFAULT_LOCALE.

Returns
-------
tuple
    A tuple in the form `(locale_data_dict, message)`.

### Function: get_language_pack(locale_)

**Description:** Get a language pack for a given `locale_` and update with any installed
package locales.

Returns
-------
tuple
    A tuple in the form `(locale_data_dict, message)`.

Notes
-----
We call `_get_installed_language_pack_locales` via a subprocess to
guarantee the results represent the most up-to-date entry point
information, which seems to be defined on interpreter startup.

## Class: TranslationBundle

**Description:** Translation bundle providing gettext translation functionality.

## Class: translator

**Description:** Translations manager.

### Function: __init__(self, domain, locale_)

**Description:** Initialize the bundle.

### Function: update_locale(self, locale_)

**Description:** Update the locale.

Parameters
----------
locale_: str
    The language name to use.

### Function: gettext(self, msgid)

**Description:** Translate a singular string.

Parameters
----------
msgid: str
    The singular string to translate.

Returns
-------
str
    The translated string.

### Function: ngettext(self, msgid, msgid_plural, n)

**Description:** Translate a singular string with pluralization.

Parameters
----------
msgid: str
    The singular string to translate.
msgid_plural: str
    The plural string to translate.
n: int
    The number for pluralization.

Returns
-------
str
    The translated string.

### Function: pgettext(self, msgctxt, msgid)

**Description:** Translate a singular string with context.

Parameters
----------
msgctxt: str
    The message context.
msgid: str
    The singular string to translate.

Returns
-------
str
    The translated string.

### Function: npgettext(self, msgctxt, msgid, msgid_plural, n)

**Description:** Translate a singular string with context and pluralization.

Parameters
----------
msgctxt: str
    The message context.
msgid: str
    The singular string to translate.
msgid_plural: str
    The plural string to translate.
n: int
    The number for pluralization.

Returns
-------
str
    The translated string.

### Function: __(self, msgid)

**Description:** Shorthand for gettext.

Parameters
----------
msgid: str
    The singular string to translate.

Returns
-------
str
    The translated string.

### Function: _n(self, msgid, msgid_plural, n)

**Description:** Shorthand for ngettext.

Parameters
----------
msgid: str
    The singular string to translate.
msgid_plural: str
    The plural string to translate.
n: int
    The number for pluralization.

Returns
-------
str
    The translated string.

### Function: _p(self, msgctxt, msgid)

**Description:** Shorthand for pgettext.

Parameters
----------
msgctxt: str
    The message context.
msgid: str
    The singular string to translate.

Returns
-------
str
    The translated string.

### Function: _np(self, msgctxt, msgid, msgid_plural, n)

**Description:** Shorthand for npgettext.

Parameters
----------
msgctxt: str
    The message context.
msgid: str
    The singular string to translate.
msgid_plural: str
    The plural string to translate.
n: int
    The number for pluralization.

Returns
-------
str
    The translated string.

### Function: normalize_domain(domain)

**Description:** Normalize a domain name.

Parameters
----------
domain: str
    Domain to normalize

Returns
-------
str
    Normalized domain

### Function: set_locale(cls, locale_)

**Description:** Set locale for the translation bundles based on the settings.

Parameters
----------
locale_: str
    The language name to use.

### Function: load(cls, domain)

**Description:** Load translation domain.

The domain is usually the normalized ``package_name``.

Parameters
----------
domain: str
    The translations domain. The normalized python package name.

Returns
-------
Translator
    A translator instance bound to the domain.

### Function: _translate_schema_strings(translations, schema, prefix, to_translate)

**Description:** Translate a schema in-place.

### Function: translate_schema(schema)

**Description:** Translate a schema.

Parameters
----------
schema: dict
    The schema to be translated

Returns
-------
Dict
    The translated schema
