## AI Summary

A file named markdown_mistune.py.


## Class: InvalidNotebook

**Description:** An invalid notebook model.

### Function: _dotall(pattern)

**Description:** Makes the '.' special character match any character inside the pattern, including a newline.

This is implemented with the inline flag `(?s:...)` and is equivalent to using `re.DOTALL`.
It is useful for LaTeX environments, where line breaks may be present.

## Class: IPythonRenderer

**Description:** An ipython html renderer.

## Class: MarkdownWithMath

**Description:** Markdown text with math enabled.

### Function: markdown2html_mistune(source)

**Description:** Convert a markdown string to HTML using mistune

## Class: MathBlockParser

**Description:** This acts as a pass-through to the MathInlineParser. It is needed in
order to avoid other block level rules splitting math sections apart.

It works by matching each multiline math environment as a single paragraph,
so that other rules don't think each section is its own paragraph. Inline
is ignored here.

## Class: MathInlineParser

**Description:** This interprets the content of LaTeX style math objects.

In particular this grabs ``$$...$$``, ``\\[...\\]``, ``\\(...\\)``, ``$...$``,
and ``\begin{foo}...\end{foo}`` styles for declaring mathematics. It strips
delimiters from all these varieties, and extracts the type of environment
in the last case (``foo`` in this example).

## Class: MathBlockParser

**Description:** This acts as a pass-through to the MathInlineParser. It is needed in
order to avoid other block level rules splitting math sections apart.

## Class: MathInlineParser

**Description:** This interprets the content of LaTeX style math objects.

In particular this grabs ``$$...$$``, ``\\[...\\]``, ``\\(...\\)``, ``$...$``,
and ``\begin{foo}...\end{foo}`` styles for declaring mathematics. It strips
delimiters from all these varieties, and extracts the type of environment
in the last case (``foo`` in this example).

### Function: __init__(self, escape, allow_harmful_protocols, embed_images, exclude_anchor_links, anchor_link_text, path, attachments)

**Description:** Initialize the renderer.

### Function: block_code(self, code, info)

**Description:** Handle block code.

### Function: block_mermaidjs(self, code)

**Description:** Handle mermaid syntax.

### Function: block_html(self, html)

**Description:** Handle block html.

### Function: inline_html(self, html)

**Description:** Handle inline html.

### Function: heading(self, text, level)

**Description:** Handle a heading.

### Function: escape_html(self, text)

**Description:** Escape html content.

### Function: block_math(self, body)

**Description:** Handle block math.

### Function: multiline_math(self, text)

**Description:** Handle mulitline math for older mistune versions.

### Function: latex_environment(self, name, body)

**Description:** Handle a latex environment.

### Function: inline_math(self, body)

**Description:** Handle inline math.

### Function: image(self, text, url, title)

**Description:** Rendering a image with title and text.

:param text: alt text of the image.
:param url: source link of the image.
:param title: title text of the image.

:note: The parameters `text` and `url` are swapped in older versions
    of mistune.

### Function: _embed_image_or_attachment(self, src)

**Description:** Embed an image or attachment, depending on the configuration.
If neither is possible, returns the original URL.

### Function: _src_to_base64(self, src)

**Description:** Turn the source file into a base64 url.

:param src: source link of the file.
:return: the base64 url or None if the file was not found.

### Function: _html_embed_images(self, html)

### Function: __init__(self, renderer, block, inline, plugins)

**Description:** Initialize the parser.

### Function: render(self, source)

**Description:** Render the HTML output for a Markdown source.

### Function: import_plugin(name)

**Description:** Simple implementation of Mistune V3's import_plugin for V2.

### Function: parse_multiline_math(self, m, state)

**Description:** Send mutiline math as a single paragraph to MathInlineParser.

### Function: parse_block_math_tex(self, m, state)

**Description:** Parse older TeX-style display math.

### Function: parse_block_math_latex(self, m, state)

**Description:** Parse newer LaTeX-style display math.

### Function: parse_inline_math_tex(self, m, state)

**Description:** Parse older TeX-style inline math.

### Function: parse_inline_math_latex(self, m, state)

**Description:** Parse newer LaTeX-style inline math.

### Function: parse_latex_environment(self, m, state)

**Description:** Parse a latex environment.

### Function: parse_multiline_math(self, m, state)

**Description:** Pass token through mutiline math.

### Function: parse_block_math_tex(self, m, state)

**Description:** Parse block text math.

### Function: parse_block_math_latex(self, m, state)

**Description:** Parse block latex math .

### Function: parse_inline_math_tex(self, m, state)

**Description:** Parse inline tex math.

### Function: parse_inline_math_latex(self, m, state)

**Description:** Parse inline latex math.

### Function: parse_latex_environment(self, m, state)

**Description:** Parse a latex environment.

## Class: Plugin

**Description:** Mistune plugin interface.

### Function: __call__(self, markdown)

**Description:** Apply the plugin on the markdown document.
