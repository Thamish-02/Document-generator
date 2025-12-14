## AI Summary

A file named linkifier.py.


### Function: build_url_re(tlds, protocols)

**Description:** Builds the url regex used by linkifier

If you want a different set of tlds or allowed protocols, pass those in
and stomp on the existing ``url_re``::

    from bleach import linkifier

    my_url_re = linkifier.build_url_re(my_tlds_list, my_protocols)

    linker = LinkifyFilter(url_re=my_url_re)

### Function: build_email_re(tlds)

**Description:** Builds the email regex used by linkifier

If you want a different set of tlds, pass those in and stomp on the existing ``email_re``::

    from bleach import linkifier

    my_email_re = linkifier.build_email_re(my_tlds_list)

    linker = LinkifyFilter(email_re=my_url_re)

## Class: Linker

**Description:** Convert URL-like strings in an HTML fragment to links

This function converts strings that look like URLs, domain names and email
addresses in text that may be an HTML fragment to links, while preserving:

1. links already in the string
2. urls found in attributes
3. email addresses

linkify does a best-effort approach and tries to recover from bad
situations due to crazy text.

## Class: LinkifyFilter

**Description:** html5lib filter that linkifies text

This will do the following:

* convert email addresses into links
* convert urls into links
* edit existing links by running them through callbacks--the default is to
  add a ``rel="nofollow"``

This filter can be used anywhere html5lib filters can be used.

### Function: __init__(self, callbacks, skip_tags, parse_email, url_re, email_re, recognized_tags)

**Description:** Creates a Linker instance

:arg list callbacks: list of callbacks to run when adjusting tag attributes;
    defaults to ``bleach.linkifier.DEFAULT_CALLBACKS``

:arg set skip_tags: set of tags that you don't want to linkify the
    contents of; for example, you could set this to ``{'pre'}`` to skip
    linkifying contents of ``pre`` tags; ``None`` means you don't
    want linkify to skip any tags

:arg bool parse_email: whether or not to linkify email addresses

:arg url_re: url matching regex

:arg email_re: email matching regex

:arg set recognized_tags: the set of tags that linkify knows about;
    everything else gets escaped

:returns: linkified text as unicode

### Function: linkify(self, text)

**Description:** Linkify specified text

:arg str text: the text to add links to

:returns: linkified text as unicode

:raises TypeError: if ``text`` is not a text type

### Function: __init__(self, source, callbacks, skip_tags, parse_email, url_re, email_re)

**Description:** Creates a LinkifyFilter instance

:arg source: stream as an html5lib TreeWalker

:arg list callbacks: list of callbacks to run when adjusting tag attributes;
    defaults to ``bleach.linkifier.DEFAULT_CALLBACKS``

:arg set skip_tags: set of tags that you don't want to linkify the
    contents of; for example, you could set this to ``{'pre'}`` to skip
    linkifying contents of ``pre`` tags

:arg bool parse_email: whether or not to linkify email addresses

:arg url_re: url matching regex

:arg email_re: email matching regex

### Function: apply_callbacks(self, attrs, is_new)

**Description:** Given an attrs dict and an is_new bool, runs through callbacks

Callbacks can return an adjusted attrs dict or ``None``. In the case of
``None``, we stop going through callbacks and return that and the link
gets dropped.

:arg dict attrs: map of ``(namespace, name)`` -> ``value``

:arg bool is_new: whether or not this link was added by linkify

:returns: adjusted attrs dict or ``None``

### Function: extract_character_data(self, token_list)

**Description:** Extracts and squashes character sequences in a token stream

### Function: handle_email_addresses(self, src_iter)

**Description:** Handle email addresses in character tokens

### Function: strip_non_url_bits(self, fragment)

**Description:** Strips non-url bits from the url

This accounts for over-eager matching by the regex.

### Function: handle_links(self, src_iter)

**Description:** Handle links in character tokens

### Function: handle_a_tag(self, token_buffer)

**Description:** Handle the "a" tag

This could adjust the link or drop it altogether depending on what the
callbacks return.

This yields the new set of tokens.

### Function: extract_entities(self, token)

**Description:** Handles Characters tokens with entities

Our overridden tokenizer doesn't do anything with entities. However,
that means that the serializer will convert all ``&`` in Characters
tokens to ``&amp;``.

Since we don't want that, we extract entities here and convert them to
Entity tokens so the serializer will let them be.

:arg token: the Characters token to work on

:returns: generator of tokens

### Function: __iter__(self)
