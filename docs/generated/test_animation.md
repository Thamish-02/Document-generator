## AI Summary

A file named test_animation.py.


### Function: anim(request)

**Description:** Create a simple animation (with options).

## Class: NullMovieWriter

**Description:** A minimal MovieWriter.  It doesn't actually write anything.
It just saves the arguments that were given to the setup() and
grab_frame() methods as attributes, and counts how many times
grab_frame() is called.

This class doesn't have an __init__ method with the appropriate
signature, and it doesn't define an isAvailable() method, so
it cannot be added to the 'writers' registry.

### Function: test_null_movie_writer(anim)

### Function: test_animation_delete(anim)

### Function: test_movie_writer_dpi_default()

## Class: RegisteredNullMovieWriter

### Function: gen_writers()

### Function: test_save_animation_smoketest(tmpdir, writer, frame_format, output, anim)

### Function: test_grabframe(tmpdir, writer, frame_format, output)

### Function: test_animation_repr_html(writer, html, want, anim)

### Function: test_no_length_frames(anim)

### Function: test_movie_writer_registry()

### Function: test_embed_limit(method_name, caplog, tmpdir, anim)

### Function: test_cleanup_temporaries(method_name, tmpdir, anim)

### Function: test_failing_ffmpeg(tmpdir, monkeypatch, anim)

**Description:** Test that we correctly raise a CalledProcessError when ffmpeg fails.

To do so, mock ffmpeg using a simple executable shell script that
succeeds when called with no arguments (so that it gets registered by
`isAvailable`), but fails otherwise, and add it to the $PATH.

### Function: test_funcanimation_cache_frame_data(cache_frame_data)

### Function: test_draw_frame(return_value)

### Function: test_exhausted_animation(tmpdir)

### Function: test_no_frame_warning(tmpdir)

### Function: test_animation_frame(tmpdir, fig_test, fig_ref)

### Function: test_save_count_override_warnings_has_length(anim)

### Function: test_save_count_override_warnings_scaler(anim)

### Function: test_disable_cache_warning(anim)

### Function: test_movie_writer_invalid_path(anim)

### Function: test_animation_with_transparency()

**Description:** Test animation exhaustion with transparency using PillowWriter directly

### Function: init()

### Function: animate(i)

### Function: setup(self, fig, outfile, dpi)

### Function: grab_frame(self)

### Function: finish(self)

## Class: DummyMovieWriter

### Function: __init__(self, fps, codec, bitrate, extra_args, metadata)

### Function: isAvailable(cls)

## Class: Frame

### Function: init()

### Function: animate(frame)

### Function: frames_generator()

### Function: animate(i)

### Function: update(frame)

### Function: update(frame)

### Function: init()

### Function: animate(i)

### Function: _run(self)
