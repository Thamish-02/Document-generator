## AI Summary

A file named test_constrainedlayout.py.


### Function: example_plot(ax, fontsize, nodec)

### Function: example_pcolor(ax, fontsize)

### Function: test_constrained_layout1()

**Description:** Test constrained_layout for a single subplot

### Function: test_constrained_layout2()

**Description:** Test constrained_layout for 2x2 subplots

### Function: test_constrained_layout3()

**Description:** Test constrained_layout for colorbars with subplots

### Function: test_constrained_layout4()

**Description:** Test constrained_layout for a single colorbar with subplots

### Function: test_constrained_layout5()

**Description:** Test constrained_layout for a single colorbar with subplots,
colorbar bottom

### Function: test_constrained_layout6()

**Description:** Test constrained_layout for nested gridspecs

### Function: test_identical_subgridspec()

### Function: test_constrained_layout7()

**Description:** Test for proper warning if fig not set in GridSpec

### Function: test_constrained_layout8()

**Description:** Test for gridspecs that are not completely full

### Function: test_constrained_layout9()

**Description:** Test for handling suptitle and for sharex and sharey

### Function: test_constrained_layout10()

**Description:** Test for handling legend outside axis

### Function: test_constrained_layout11()

**Description:** Test for multiple nested gridspecs

### Function: test_constrained_layout11rat()

**Description:** Test for multiple nested gridspecs with width_ratios

### Function: test_constrained_layout12()

**Description:** Test that very unbalanced labeling still works.

### Function: test_constrained_layout13()

**Description:** Test that padding works.

### Function: test_constrained_layout14()

**Description:** Test that padding works.

### Function: test_constrained_layout15()

**Description:** Test that rcparams work.

### Function: test_constrained_layout16()

**Description:** Test ax.set_position.

### Function: test_constrained_layout17()

**Description:** Test uneven gridspecs

### Function: test_constrained_layout18()

**Description:** Test twinx

### Function: test_constrained_layout19()

**Description:** Test twiny

### Function: test_constrained_layout20()

**Description:** Smoke test cl does not mess up added Axes

### Function: test_constrained_layout21()

**Description:** #11035: repeated calls to suptitle should not alter the layout

### Function: test_constrained_layout22()

**Description:** #11035: suptitle should not be include in CL if manually positioned

### Function: test_constrained_layout23()

**Description:** Comment in #11035: suptitle used to cause an exception when
reusing a figure w/ CL with ``clear=True``.

### Function: test_colorbar_location()

**Description:** Test that colorbar handling is as expected for various complicated
cases...

### Function: test_hidden_axes()

### Function: test_colorbar_align()

### Function: test_colorbars_no_overlapV()

### Function: test_colorbars_no_overlapH()

### Function: test_manually_set_position()

### Function: test_bboxtight()

### Function: test_bbox()

### Function: test_align_labels()

**Description:** Tests for a bug in which constrained layout and align_ylabels on
three unevenly sized subplots, one of whose y tick labels include
negative numbers, drives the non-negative subplots' y labels off
the edge of the plot

### Function: test_suplabels()

### Function: test_gridspec_addressing()

### Function: test_discouraged_api()

### Function: test_kwargs()

### Function: test_rect()

### Function: test_compressed1()

### Function: test_compressed_suptitle()

### Function: test_set_constrained_layout(arg, state)

### Function: test_constrained_toggle()

### Function: test_layout_leak()

### Function: test_submerged_subfig()

**Description:** Test that the submerged margin logic does not get called multiple times
on same axes if it is already in a subfigure
