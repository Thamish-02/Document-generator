## AI Summary

A file named proj3d.py.


### Function: world_transformation(xmin, xmax, ymin, ymax, zmin, zmax, pb_aspect)

**Description:** Produce a matrix that scales homogeneous coords in the specified ranges
to [0, 1], or [0, pb_aspect[i]] if the plotbox aspect ratio is specified.

### Function: _rotation_about_vector(v, angle)

**Description:** Produce a rotation matrix for an angle in radians about a vector.

### Function: _view_axes(E, R, V, roll)

**Description:** Get the unit viewing axes in data coordinates.

Parameters
----------
E : 3-element numpy array
    The coordinates of the eye/camera.
R : 3-element numpy array
    The coordinates of the center of the view box.
V : 3-element numpy array
    Unit vector in the direction of the vertical axis.
roll : float
    The roll angle in radians.

Returns
-------
u : 3-element numpy array
    Unit vector pointing towards the right of the screen.
v : 3-element numpy array
    Unit vector pointing towards the top of the screen.
w : 3-element numpy array
    Unit vector pointing out of the screen.

### Function: _view_transformation_uvw(u, v, w, E)

**Description:** Return the view transformation matrix.

Parameters
----------
u : 3-element numpy array
    Unit vector pointing towards the right of the screen.
v : 3-element numpy array
    Unit vector pointing towards the top of the screen.
w : 3-element numpy array
    Unit vector pointing out of the screen.
E : 3-element numpy array
    The coordinates of the eye/camera.

### Function: _persp_transformation(zfront, zback, focal_length)

### Function: _ortho_transformation(zfront, zback)

### Function: _proj_transform_vec(vec, M)

### Function: _proj_transform_vec_clip(vec, M, focal_length)

### Function: inv_transform(xs, ys, zs, invM)

**Description:** Transform the points by the inverse of the projection matrix, *invM*.

### Function: _vec_pad_ones(xs, ys, zs)

### Function: proj_transform(xs, ys, zs, M)

**Description:** Transform the points by the projection matrix *M*.

### Function: proj_transform_clip(xs, ys, zs, M)

### Function: _proj_transform_clip(xs, ys, zs, M, focal_length)

**Description:** Transform the points by the projection matrix
and return the clipping result
returns txs, tys, tzs, tis

### Function: _proj_points(points, M)

### Function: _proj_trans_points(points, M)
