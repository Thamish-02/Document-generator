## AI Summary

A file named test_triangulation.py.


## Class: TestTriangulationParams

### Function: test_extract_triangulation_positional_mask()

### Function: test_triangulation_init()

### Function: test_triangulation_set_mask()

### Function: test_delaunay()

### Function: test_delaunay_duplicate_points()

### Function: test_delaunay_points_in_line()

### Function: test_delaunay_insufficient_points(x, y)

### Function: test_delaunay_robust()

### Function: test_tripcolor()

### Function: test_tripcolor_color()

### Function: test_tripcolor_clim()

### Function: test_tripcolor_warnings()

### Function: test_no_modify()

### Function: test_trifinder()

### Function: test_triinterp()

### Function: test_triinterpcubic_C1_continuity()

### Function: test_triinterpcubic_cg_solver()

### Function: test_triinterpcubic_geom_weights()

### Function: test_triinterp_colinear()

### Function: test_triinterp_transformations()

### Function: test_tri_smooth_contouring()

### Function: test_tri_smooth_gradient()

### Function: test_tritools()

### Function: test_trirefine()

### Function: test_trirefine_masked(interpolator)

### Function: meshgrid_triangles(n)

**Description:** Return (2*(N-1)**2, 3) array of triangles to mesh (N, N)-point np.meshgrid.

### Function: test_triplot_return()

### Function: test_trirefiner_fortran_contiguous_triangles()

### Function: test_qhull_triangle_orientation()

### Function: test_trianalyzer_mismatched_indices()

### Function: test_tricontourf_decreasing_levels()

### Function: test_internal_cpp_api()

### Function: test_qhull_large_offset()

### Function: test_tricontour_non_finite_z()

### Function: test_tricontourset_reuse()

### Function: test_triplot_with_ls(fig_test, fig_ref)

### Function: test_triplot_label()

### Function: test_tricontour_path()

### Function: test_tricontourf_path()

### Function: test_extract_triangulation_params(self, args, kwargs, expected)

### Function: tri_contains_point(xtri, ytri, xy)

### Function: tris_contain_point(triang, xy)

### Function: quad(x, y)

### Function: gradient_quad(x, y)

### Function: check_continuity(interpolator, loc, values)

**Description:** Checks the continuity of interpolator (and its derivatives) near
location loc. Can check the value at loc itself if *values* is
provided.

*interpolator* TriInterpolator
*loc* location to test (x0, y0)
*values* (optional) array [z0, dzx0, dzy0] to check the value at *loc*

### Function: poisson_sparse_matrix(n, m)

**Description:** Return the sparse, (n*m, n*m) matrix in coo format resulting from the
discretisation of the 2-dimensional Poisson equation according to a
finite difference numerical scheme on a uniform (n, m) grid.

### Function: z(x, y)

### Function: z(x, y)

### Function: dipole_potential(x, y)

**Description:** An electric dipole potential V.

### Function: power(x, a)
