## AI Summary

A file named _mathtext.py.


### Function: get_unicode_index(symbol)

**Description:** Return the integer index (from the Unicode table) of *symbol*.

Parameters
----------
symbol : str
    A single (Unicode) character, a TeX command (e.g. r'\pi') or a Type1
    symbol name (e.g. 'phi').

## Class: VectorParse

**Description:** The namedtuple type returned by ``MathTextParser("path").parse(...)``.

Attributes
----------
width, height, depth : float
    The global metrics.
glyphs : list
    The glyphs including their positions.
rect : list
    The list of rectangles.

## Class: RasterParse

**Description:** The namedtuple type returned by ``MathTextParser("agg").parse(...)``.

Attributes
----------
ox, oy : float
    The offsets are always zero.
width, height, depth : float
    The global metrics.
image : FT2Image
    A raster image.

## Class: Output

**Description:** Result of `ship`\ping a box: lists of positioned glyphs and rectangles.

This class is not exposed to end users, but converted to a `VectorParse` or
a `RasterParse` by `.MathTextParser.parse`.

## Class: FontMetrics

**Description:** Metrics of a font.

Attributes
----------
advance : float
    The advance distance (in points) of the glyph.
height : float
    The height of the glyph in points.
width : float
    The width of the glyph in points.
xmin, xmax, ymin, ymax : float
    The ink rectangle of the glyph.
iceberg : float
    The distance from the baseline to the top of the glyph. (This corresponds to
    TeX's definition of "height".)
slanted : bool
    Whether the glyph should be considered as "slanted" (currently used for kerning
    sub/superscripts).

## Class: FontInfo

## Class: Fonts

**Description:** An abstract base class for a system of fonts to use for mathtext.

The class must be able to take symbol keys and font file names and
return the character metrics.  It also delegates to a backend class
to do the actual drawing.

## Class: TruetypeFonts

**Description:** A generic base class for all font setups that use Truetype fonts
(through FT2Font).

## Class: BakomaFonts

**Description:** Use the Bakoma TrueType fonts for rendering.

Symbols are strewn about a number of font files, each of which has
its own proprietary 8-bit encoding.

## Class: UnicodeFonts

**Description:** An abstract base class for handling Unicode fonts.

While some reasonably complete Unicode fonts (such as DejaVu) may
work in some situations, the only Unicode font I'm aware of with a
complete set of math symbols is STIX.

This class will "fallback" on the Bakoma fonts when a required
symbol cannot be found in the font.

## Class: DejaVuFonts

## Class: DejaVuSerifFonts

**Description:** A font handling class for the DejaVu Serif fonts

If a glyph is not found it will fallback to Stix Serif

## Class: DejaVuSansFonts

**Description:** A font handling class for the DejaVu Sans fonts

If a glyph is not found it will fallback to Stix Sans

## Class: StixFonts

**Description:** A font handling class for the STIX fonts.

In addition to what UnicodeFonts provides, this class:

- supports "virtual fonts" which are complete alpha numeric
  character sets with different font styles at special Unicode
  code points, such as "Blackboard".

- handles sized alternative characters for the STIXSizeX fonts.

## Class: StixSansFonts

**Description:** A font handling class for the STIX fonts (that uses sans-serif
characters by default).

## Class: FontConstantsBase

**Description:** A set of constants that controls how certain things, such as sub-
and superscripts are laid out.  These are all metrics that can't
be reliably retrieved from the font metrics in the font itself.

## Class: ComputerModernFontConstants

## Class: STIXFontConstants

## Class: STIXSansFontConstants

## Class: DejaVuSerifFontConstants

## Class: DejaVuSansFontConstants

### Function: _get_font_constant_set(state)

## Class: Node

**Description:** A node in the TeX box model.

## Class: Box

**Description:** A node with a physical location.

## Class: Vbox

**Description:** A box with only height (zero width).

## Class: Hbox

**Description:** A box with only width (zero height and depth).

## Class: Char

**Description:** A single character.

Unlike TeX, the font information and metrics are stored with each `Char`
to make it easier to lookup the font metrics when needed.  Note that TeX
boxes have a width, height, and depth, unlike Type1 and TrueType which use
a full bounding box and an advance in the x-direction.  The metrics must
be converted to the TeX model, and the advance (if different from width)
must be converted into a `Kern` node when the `Char` is added to its parent
`Hlist`.

## Class: Accent

**Description:** The font metrics need to be dealt with differently for accents,
since they are already offset correctly from the baseline in
TrueType fonts.

## Class: List

**Description:** A list of nodes (either horizontal or vertical).

## Class: Hlist

**Description:** A horizontal list of boxes.

## Class: Vlist

**Description:** A vertical list of boxes.

## Class: Rule

**Description:** A solid black rectangle.

It has *width*, *depth*, and *height* fields just as in an `Hlist`.
However, if any of these dimensions is inf, the actual value will be
determined by running the rule up to the boundary of the innermost
enclosing box.  This is called a "running dimension".  The width is never
running in an `Hlist`; the height and depth are never running in a `Vlist`.

## Class: Hrule

**Description:** Convenience class to create a horizontal rule.

## Class: Vrule

**Description:** Convenience class to create a vertical rule.

## Class: _GlueSpec

## Class: Glue

**Description:** Most of the information in this object is stored in the underlying
``_GlueSpec`` class, which is shared between multiple glue objects.
(This is a memory optimization which probably doesn't matter anymore, but
it's easier to stick to what TeX does.)

## Class: HCentered

**Description:** A convenience class to create an `Hlist` whose contents are
centered within its enclosing box.

## Class: VCentered

**Description:** A convenience class to create a `Vlist` whose contents are
centered within its enclosing box.

## Class: Kern

**Description:** A `Kern` node has a width field to specify a (normally
negative) amount of spacing. This spacing correction appears in
horizontal lists between letters like A and V when the font
designer said that it looks better to move them closer together or
further apart. A kern node can also appear in a vertical list,
when its *width* denotes additional spacing in the vertical
direction.

## Class: AutoHeightChar

**Description:** A character as close to the given height and depth as possible.

When using a font with multiple height versions of some characters (such as
the BaKoMa fonts), the correct glyph will be selected, otherwise this will
always just return a scaled version of the glyph.

## Class: AutoWidthChar

**Description:** A character as close to the given width as possible.

When using a font with multiple width versions of some characters (such as
the BaKoMa fonts), the correct glyph will be selected, otherwise this will
always just return a scaled version of the glyph.

### Function: ship(box, xy)

**Description:** Ship out *box* at offset *xy*, converting it to an `Output`.

Since boxes can be inside of boxes inside of boxes, the main work of `ship`
is done by two mutually recursive routines, `hlist_out` and `vlist_out`,
which traverse the `Hlist` nodes and `Vlist` nodes inside of horizontal
and vertical boxes.  The global variables used in TeX to store state as it
processes have become local variables here.

### Function: Error(msg)

**Description:** Helper class to raise parser errors.

## Class: ParserState

**Description:** Parser state.

States are pushed and popped from a stack as necessary, and the "current"
state is always at the top of the stack.

Upon entering and leaving a group { } or math/non-math, the stack is pushed
and popped accordingly.

### Function: cmd(expr, args)

**Description:** Helper to define TeX commands.

``cmd("\cmd", args)`` is equivalent to
``"\cmd" - (args | Error("Expected \cmd{arg}{...}"))`` where the names in
the error message are taken from element names in *args*.  If *expr*
already includes arguments (e.g. "\cmd{arg}{...}"), then they are stripped
when constructing the parse element, but kept (and *expr* is used as is) in
the error message.

## Class: Parser

**Description:** A pyparsing-based parser for strings containing math expressions.

Raw text may also appear outside of pairs of ``$``.

The grammar is based directly on that in TeX, though it cuts a few corners.

### Function: __init__(self, box)

### Function: to_vector(self)

### Function: to_raster(self)

### Function: __init__(self, default_font_prop, load_glyph_flags)

**Description:** Parameters
----------
default_font_prop : `~.font_manager.FontProperties`
    The default non-math font, or the base font for Unicode (generic)
    font rendering.
load_glyph_flags : `.ft2font.LoadFlags`
    Flags passed to the glyph loader (e.g. ``FT_Load_Glyph`` and
    ``FT_Load_Char`` for FreeType-based fonts).

### Function: get_kern(self, font1, fontclass1, sym1, fontsize1, font2, fontclass2, sym2, fontsize2, dpi)

**Description:** Get the kerning distance for font between *sym1* and *sym2*.

See `~.Fonts.get_metrics` for a detailed description of the parameters.

### Function: _get_font(self, font)

### Function: _get_info(self, font, font_class, sym, fontsize, dpi)

### Function: get_metrics(self, font, font_class, sym, fontsize, dpi)

**Description:** Parameters
----------
font : str
    One of the TeX font names: "tt", "it", "rm", "cal", "sf", "bf",
    "default", "regular", "bb", "frak", "scr".  "default" and "regular"
    are synonyms and use the non-math font.
font_class : str
    One of the TeX font names (as for *font*), but **not** "bb",
    "frak", or "scr".  This is used to combine two font classes.  The
    only supported combination currently is ``get_metrics("frak", "bf",
    ...)``.
sym : str
    A symbol in raw TeX form, e.g., "1", "x", or "\sigma".
fontsize : float
    Font size in points.
dpi : float
    Rendering dots-per-inch.

Returns
-------
FontMetrics

### Function: render_glyph(self, output, ox, oy, font, font_class, sym, fontsize, dpi)

**Description:** At position (*ox*, *oy*), draw the glyph specified by the remaining
parameters (see `get_metrics` for their detailed description).

### Function: render_rect_filled(self, output, x1, y1, x2, y2)

**Description:** Draw a filled rectangle from (*x1*, *y1*) to (*x2*, *y2*).

### Function: get_xheight(self, font, fontsize, dpi)

**Description:** Get the xheight for the given *font* and *fontsize*.

### Function: get_underline_thickness(self, font, fontsize, dpi)

**Description:** Get the line thickness that matches the given font.  Used as a
base unit for drawing lines such as in a fraction or radical.

### Function: get_sized_alternatives_for_symbol(self, fontname, sym)

**Description:** Override if your font provides multiple sizes of the same
symbol.  Should return a list of symbols matching *sym* in
various sizes.  The expression renderer will select the most
appropriate size for a given situation from this list.

### Function: __init__(self, default_font_prop, load_glyph_flags)

### Function: _get_font(self, font)

### Function: _get_offset(self, font, glyph, fontsize, dpi)

### Function: _get_glyph(self, fontname, font_class, sym)

### Function: _get_info(self, fontname, font_class, sym, fontsize, dpi)

### Function: get_xheight(self, fontname, fontsize, dpi)

### Function: get_underline_thickness(self, font, fontsize, dpi)

### Function: get_kern(self, font1, fontclass1, sym1, fontsize1, font2, fontclass2, sym2, fontsize2, dpi)

### Function: __init__(self, default_font_prop, load_glyph_flags)

### Function: _get_glyph(self, fontname, font_class, sym)

### Function: get_sized_alternatives_for_symbol(self, fontname, sym)

### Function: __init__(self, default_font_prop, load_glyph_flags)

### Function: _map_virtual_font(self, fontname, font_class, uniindex)

### Function: _get_glyph(self, fontname, font_class, sym)

### Function: get_sized_alternatives_for_symbol(self, fontname, sym)

### Function: __init__(self, default_font_prop, load_glyph_flags)

### Function: _get_glyph(self, fontname, font_class, sym)

### Function: __init__(self, default_font_prop, load_glyph_flags)

### Function: _map_virtual_font(self, fontname, font_class, uniindex)

### Function: get_sized_alternatives_for_symbol(self, fontname, sym)

### Function: __init__(self)

### Function: __repr__(self)

### Function: get_kerning(self, next)

### Function: shrink(self)

**Description:** Shrinks one level smaller.  There are only three levels of
sizes, after which things will no longer get smaller.

### Function: render(self, output, x, y)

**Description:** Render this node.

### Function: __init__(self, width, height, depth)

### Function: shrink(self)

### Function: render(self, output, x1, y1, x2, y2)

### Function: __init__(self, height, depth)

### Function: __init__(self, width)

### Function: __init__(self, c, state)

### Function: __repr__(self)

### Function: _update_metrics(self)

### Function: is_slanted(self)

### Function: get_kerning(self, next)

**Description:** Return the amount of kerning between this and the given character.

This method is called when characters are strung together into `Hlist`
to create `Kern` nodes.

### Function: render(self, output, x, y)

### Function: shrink(self)

### Function: _update_metrics(self)

### Function: shrink(self)

### Function: render(self, output, x, y)

### Function: __init__(self, elements)

### Function: __repr__(self)

### Function: _set_glue(self, x, sign, totals, error_type)

### Function: shrink(self)

### Function: __init__(self, elements, w, m, do_kern)

### Function: kern(self)

**Description:** Insert `Kern` nodes between `Char` nodes to set kerning.

The `Char` nodes themselves determine the amount of kerning they need
(in `~Char.get_kerning`), and this function just creates the correct
linked list.

### Function: hpack(self, w, m)

**Description:** Compute the dimensions of the resulting boxes, and adjust the glue if
one of those dimensions is pre-specified.  The computed sizes normally
enclose all of the material inside the new box; but some items may
stick out if negative glue is used, if the box is overfull, or if a
``\vbox`` includes other boxes that have been shifted left.

Parameters
----------
w : float, default: 0
    A width.
m : {'exactly', 'additional'}, default: 'additional'
    Whether to produce a box whose width is 'exactly' *w*; or a box
    with the natural width of the contents, plus *w* ('additional').

Notes
-----
The defaults produce a box with the natural width of the contents.

### Function: __init__(self, elements, h, m)

### Function: vpack(self, h, m, l)

**Description:** Compute the dimensions of the resulting boxes, and to adjust the glue
if one of those dimensions is pre-specified.

Parameters
----------
h : float, default: 0
    A height.
m : {'exactly', 'additional'}, default: 'additional'
    Whether to produce a box whose height is 'exactly' *h*; or a box
    with the natural height of the contents, plus *h* ('additional').
l : float, default: np.inf
    The maximum height.

Notes
-----
The defaults produce a box with the natural height of the contents.

### Function: __init__(self, width, height, depth, state)

### Function: render(self, output, x, y, w, h)

### Function: __init__(self, state, thickness)

### Function: __init__(self, state)

### Function: __init__(self, glue_type)

### Function: shrink(self)

### Function: __init__(self, elements)

### Function: __init__(self, elements)

### Function: __init__(self, width)

### Function: __repr__(self)

### Function: shrink(self)

### Function: __init__(self, c, height, depth, state, always, factor)

### Function: __init__(self, c, width, state, always, char_class)

### Function: clamp(value)

### Function: hlist_out(box)

### Function: vlist_out(box)

### Function: raise_error(s, loc, toks)

### Function: __init__(self, fontset, font, font_class, fontsize, dpi)

### Function: copy(self)

### Function: font(self)

### Function: font(self, name)

### Function: get_current_underline_thickness(self)

**Description:** Return the underline thickness for this state.

### Function: names(elt)

## Class: _MathStyle

### Function: __init__(self)

### Function: parse(self, s, fonts_object, fontsize, dpi)

**Description:** Parse expression *s* using the given *fonts_object* for
output, at the given *fontsize* and *dpi*.

Returns the parse tree of `Node` instances.

### Function: get_state(self)

**Description:** Get the current `State` of the parser.

### Function: pop_state(self)

**Description:** Pop a `State` off of the stack.

### Function: push_state(self)

**Description:** Push a new `State` onto the stack, copying the current state.

### Function: main(self, toks)

### Function: math_string(self, toks)

### Function: math(self, toks)

### Function: non_math(self, toks)

### Function: text(self, toks)

### Function: _make_space(self, percentage)

### Function: space(self, toks)

### Function: customspace(self, toks)

### Function: symbol(self, s, loc, toks)

### Function: unknown_symbol(self, s, loc, toks)

### Function: accent(self, toks)

### Function: function(self, s, loc, toks)

### Function: operatorname(self, s, loc, toks)

### Function: start_group(self, toks)

### Function: group(self, toks)

### Function: required_group(self, toks)

### Function: end_group(self)

### Function: unclosed_group(self, s, loc, toks)

### Function: font(self, toks)

### Function: is_overunder(self, nucleus)

### Function: is_dropsub(self, nucleus)

### Function: is_slanted(self, nucleus)

### Function: subsuper(self, s, loc, toks)

### Function: _genfrac(self, ldelim, rdelim, rule, style, num, den)

### Function: style_literal(self, toks)

### Function: genfrac(self, toks)

### Function: frac(self, toks)

### Function: dfrac(self, toks)

### Function: binom(self, toks)

### Function: _genset(self, s, loc, toks)

### Function: sqrt(self, toks)

### Function: overline(self, toks)

### Function: _auto_sized_delimiter(self, front, middle, back)

### Function: auto_delim(self, toks)

### Function: boldsymbol(self, toks)

### Function: substack(self, toks)

### Function: set_names_and_parse_actions()

### Function: csnames(group, names)
