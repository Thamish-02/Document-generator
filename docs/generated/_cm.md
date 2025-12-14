## AI Summary

A file named _cm.py.


### Function: _flag_red(x)

### Function: _flag_green(x)

### Function: _flag_blue(x)

### Function: _prism_red(x)

### Function: _prism_green(x)

### Function: _prism_blue(x)

### Function: _ch_helper(gamma, s, r, h, p0, p1, x)

**Description:** Helper function for generating picklable cubehelix colormaps.

### Function: cubehelix(gamma, s, r, h)

**Description:** Return custom data dictionary of (r, g, b) conversion functions, which can
be used with `.ColormapRegistry.register`, for the cubehelix color scheme.

Unlike most other color schemes cubehelix was designed by D.A. Green to
be monotonically increasing in terms of perceived brightness.
Also, when printed on a black and white postscript printer, the scheme
results in a greyscale with monotonically increasing brightness.
This color scheme is named cubehelix because the (r, g, b) values produced
can be visualised as a squashed helix around the diagonal in the
(r, g, b) color cube.

For a unit color cube (i.e. 3D coordinates for (r, g, b) each in the
range 0 to 1) the color scheme starts at (r, g, b) = (0, 0, 0), i.e. black,
and finishes at (r, g, b) = (1, 1, 1), i.e. white. For some fraction *x*,
between 0 and 1, the color is the corresponding grey value at that
fraction along the black to white diagonal (x, x, x) plus a color
element. This color element is calculated in a plane of constant
perceived intensity and controlled by the following parameters.

Parameters
----------
gamma : float, default: 1
    Gamma factor emphasizing either low intensity values (gamma < 1), or
    high intensity values (gamma > 1).
s : float, default: 0.5 (purple)
    The starting color.
r : float, default: -1.5
    The number of r, g, b rotations in color that are made from the start
    to the end of the color scheme.  The default of -1.5 corresponds to ->
    B -> G -> R -> B.
h : float, default: 1
    The hue, i.e. how saturated the colors are. If this parameter is zero
    then the color scheme is purely a greyscale.

### Function: _g0(x)

### Function: _g1(x)

### Function: _g2(x)

### Function: _g3(x)

### Function: _g4(x)

### Function: _g5(x)

### Function: _g6(x)

### Function: _g7(x)

### Function: _g8(x)

### Function: _g9(x)

### Function: _g10(x)

### Function: _g11(x)

### Function: _g12(x)

### Function: _g13(x)

### Function: _g14(x)

### Function: _g15(x)

### Function: _g16(x)

### Function: _g17(x)

### Function: _g18(x)

### Function: _g19(x)

### Function: _g20(x)

### Function: _g21(x)

### Function: _g22(x)

### Function: _g23(x)

### Function: _g24(x)

### Function: _g25(x)

### Function: _g26(x)

### Function: _g27(x)

### Function: _g28(x)

### Function: _g29(x)

### Function: _g30(x)

### Function: _g31(x)

### Function: _g32(x)

### Function: _g33(x)

### Function: _g34(x)

### Function: _g35(x)

### Function: _g36(x)

### Function: _gist_heat_red(x)

### Function: _gist_heat_green(x)

### Function: _gist_heat_blue(x)

### Function: _gist_yarg(x)
