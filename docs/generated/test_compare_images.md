## AI Summary

A file named test_compare_images.py.


### Function: test_image_comparison_expect_rms(im1, im2, tol, expect_rms, tmp_path, monkeypatch)

**Description:** Compare two images, expecting a particular RMS error.

im1 and im2 are filenames relative to the baseline_dir directory.

tol is the tolerance to pass to compare_images.

expect_rms is the expected RMS value, or None. If None, the test will
succeed if compare_images succeeds. Otherwise, the test will succeed if
compare_images fails and returns an RMS error almost equal to this value.
