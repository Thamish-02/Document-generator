## AI Summary

A file named animation.py.


### Function: adjusted_figsize(w, h, dpi, n)

**Description:** Compute figure size so that pixels are a multiple of n.

Parameters
----------
w, h : float
    Size in inches.

dpi : float
    The dpi.

n : int
    The target multiple.

Returns
-------
wnew, hnew : float
    The new figure size in inches.

## Class: MovieWriterRegistry

**Description:** Registry of available writer classes by human readable name.

## Class: AbstractMovieWriter

**Description:** Abstract base class for writing movies, providing a way to grab frames by
calling `~AbstractMovieWriter.grab_frame`.

`setup` is called to start the process and `finish` is called afterwards.
`saving` is provided as a context manager to facilitate this process as ::

    with moviewriter.saving(fig, outfile='myfile.mp4', dpi=100):
        # Iterate over frames
        moviewriter.grab_frame(**savefig_kwargs)

The use of the context manager ensures that `setup` and `finish` are
performed as necessary.

An instance of a concrete subclass of this class can be given as the
``writer`` argument of `Animation.save()`.

## Class: MovieWriter

**Description:** Base class for writing movies.

This is a base class for MovieWriter subclasses that write a movie frame
data to a pipe. You cannot instantiate this class directly.
See examples for how to use its subclasses.

Attributes
----------
frame_format : str
    The format used in writing frame data, defaults to 'rgba'.
fig : `~matplotlib.figure.Figure`
    The figure to capture data from.
    This must be provided by the subclasses.

## Class: FileMovieWriter

**Description:** `MovieWriter` for writing to individual files and stitching at the end.

This must be sub-classed to be useful.

## Class: PillowWriter

## Class: FFMpegBase

**Description:** Mixin class for FFMpeg output.

This is a base class for the concrete `FFMpegWriter` and `FFMpegFileWriter`
classes.

## Class: FFMpegWriter

**Description:** Pipe-based ffmpeg writer.

Frames are streamed directly to ffmpeg via a pipe and written in a single pass.

This effectively works as a slideshow input to ffmpeg with the fps passed as
``-framerate``, so see also `their notes on frame rates`_ for further details.

.. _their notes on frame rates: https://trac.ffmpeg.org/wiki/Slideshow#Framerates

## Class: FFMpegFileWriter

**Description:** File-based ffmpeg writer.

Frames are written to temporary files on disk and then stitched together at the end.

This effectively works as a slideshow input to ffmpeg with the fps passed as
``-framerate``, so see also `their notes on frame rates`_ for further details.

.. _their notes on frame rates: https://trac.ffmpeg.org/wiki/Slideshow#Framerates

## Class: ImageMagickBase

**Description:** Mixin class for ImageMagick output.

This is a base class for the concrete `ImageMagickWriter` and
`ImageMagickFileWriter` classes, which define an ``input_names`` attribute
(or property) specifying the input names passed to ImageMagick.

## Class: ImageMagickWriter

**Description:** Pipe-based animated gif writer.

Frames are streamed directly to ImageMagick via a pipe and written
in a single pass.

## Class: ImageMagickFileWriter

**Description:** File-based animated gif writer.

Frames are written to temporary files on disk and then stitched
together at the end.

### Function: _included_frames(frame_count, frame_format, frame_dir)

### Function: _embedded_frames(frame_list, frame_format)

**Description:** frame_list should be a list of base64-encoded png files

## Class: HTMLWriter

**Description:** Writer for JavaScript-based HTML movies.

## Class: Animation

**Description:** A base class for Animations.

This class is not usable as is, and should be subclassed to provide needed
behavior.

.. note::

    You must store the created Animation in a variable that lives as long
    as the animation should run. Otherwise, the Animation object will be
    garbage-collected and the animation stops.

Parameters
----------
fig : `~matplotlib.figure.Figure`
    The figure object used to get needed events, such as draw or resize.

event_source : object, optional
    A class that can run a callback when desired events
    are generated, as well as be stopped and started.

    Examples include timers (see `TimedAnimation`) and file
    system notifications.

blit : bool, default: False
    Whether blitting is used to optimize drawing.  If the backend does not
    support blitting, then this parameter has no effect.

See Also
--------
FuncAnimation,  ArtistAnimation

## Class: TimedAnimation

**Description:** `Animation` subclass for time-based animation.

A new frame is drawn every *interval* milliseconds.

.. note::

    You must store the created Animation in a variable that lives as long
    as the animation should run. Otherwise, the Animation object will be
    garbage-collected and the animation stops.

Parameters
----------
fig : `~matplotlib.figure.Figure`
    The figure object used to get needed events, such as draw or resize.
interval : int, default: 200
    Delay between frames in milliseconds.
repeat_delay : int, default: 0
    The delay in milliseconds between consecutive animation runs, if
    *repeat* is True.
repeat : bool, default: True
    Whether the animation repeats when the sequence of frames is completed.
blit : bool, default: False
    Whether blitting is used to optimize drawing.

## Class: ArtistAnimation

**Description:** `TimedAnimation` subclass that creates an animation by using a fixed
set of `.Artist` objects.

Before creating an instance, all plotting should have taken place
and the relevant artists saved.

.. note::

    You must store the created Animation in a variable that lives as long
    as the animation should run. Otherwise, the Animation object will be
    garbage-collected and the animation stops.

Parameters
----------
fig : `~matplotlib.figure.Figure`
    The figure object used to get needed events, such as draw or resize.
artists : list
    Each list entry is a collection of `.Artist` objects that are made
    visible on the corresponding frame.  Other artists are made invisible.
interval : int, default: 200
    Delay between frames in milliseconds.
repeat_delay : int, default: 0
    The delay in milliseconds between consecutive animation runs, if
    *repeat* is True.
repeat : bool, default: True
    Whether the animation repeats when the sequence of frames is completed.
blit : bool, default: False
    Whether blitting is used to optimize drawing.

## Class: FuncAnimation

**Description:** `TimedAnimation` subclass that makes an animation by repeatedly calling
a function *func*.

.. note::

    You must store the created Animation in a variable that lives as long
    as the animation should run. Otherwise, the Animation object will be
    garbage-collected and the animation stops.

Parameters
----------
fig : `~matplotlib.figure.Figure`
    The figure object used to get needed events, such as draw or resize.

func : callable
    The function to call at each frame.  The first argument will
    be the next value in *frames*.   Any additional positional
    arguments can be supplied using `functools.partial` or via the *fargs*
    parameter.

    The required signature is::

        def func(frame, *fargs) -> iterable_of_artists

    It is often more convenient to provide the arguments using
    `functools.partial`. In this way it is also possible to pass keyword
    arguments. To pass a function with both positional and keyword
    arguments, set all arguments as keyword arguments, just leaving the
    *frame* argument unset::

        def func(frame, art, *, y=None):
            ...

        ani = FuncAnimation(fig, partial(func, art=ln, y='foo'))

    If ``blit == True``, *func* must return an iterable of all artists
    that were modified or created. This information is used by the blitting
    algorithm to determine which parts of the figure have to be updated.
    The return value is unused if ``blit == False`` and may be omitted in
    that case.

frames : iterable, int, generator function, or None, optional
    Source of data to pass *func* and each frame of the animation

    - If an iterable, then simply use the values provided.  If the
      iterable has a length, it will override the *save_count* kwarg.

    - If an integer, then equivalent to passing ``range(frames)``

    - If a generator function, then must have the signature::

         def gen_function() -> obj

    - If *None*, then equivalent to passing ``itertools.count``.

    In all of these cases, the values in *frames* is simply passed through
    to the user-supplied *func* and thus can be of any type.

init_func : callable, optional
    A function used to draw a clear frame. If not given, the results of
    drawing from the first item in the frames sequence will be used. This
    function will be called once before the first frame.

    The required signature is::

        def init_func() -> iterable_of_artists

    If ``blit == True``, *init_func* must return an iterable of artists
    to be re-drawn. This information is used by the blitting algorithm to
    determine which parts of the figure have to be updated.  The return
    value is unused if ``blit == False`` and may be omitted in that case.

fargs : tuple or None, optional
    Additional arguments to pass to each call to *func*. Note: the use of
    `functools.partial` is preferred over *fargs*. See *func* for details.

save_count : int, optional
    Fallback for the number of values from *frames* to cache. This is
    only used if the number of frames cannot be inferred from *frames*,
    i.e. when it's an iterator without length or a generator.

interval : int, default: 200
    Delay between frames in milliseconds.

repeat_delay : int, default: 0
    The delay in milliseconds between consecutive animation runs, if
    *repeat* is True.

repeat : bool, default: True
    Whether the animation repeats when the sequence of frames is completed.

blit : bool, default: False
    Whether blitting is used to optimize drawing.  Note: when using
    blitting, any animated artists will be drawn according to their zorder;
    however, they will be drawn on top of any previous artists, regardless
    of their zorder.

cache_frame_data : bool, default: True
    Whether frame data is cached.  Disabling cache might be helpful when
    frames contain large objects.

### Function: _validate_grabframe_kwargs(savefig_kwargs)

### Function: correct_roundoff(x, dpi, n)

### Function: __init__(self)

### Function: register(self, name)

**Description:** Decorator for registering a class under a name.

Example use::

    @registry.register(name)
    class Foo:
        pass

### Function: is_available(self, name)

**Description:** Check if given writer is available by name.

Parameters
----------
name : str

Returns
-------
bool

### Function: __iter__(self)

**Description:** Iterate over names of available writer class.

### Function: list(self)

**Description:** Get a list of available MovieWriters.

### Function: __getitem__(self, name)

**Description:** Get an available writer class from its name.

### Function: __init__(self, fps, metadata, codec, bitrate)

### Function: setup(self, fig, outfile, dpi)

**Description:** Setup for writing the movie file.

Parameters
----------
fig : `~matplotlib.figure.Figure`
    The figure object that contains the information for frames.
outfile : str
    The filename of the resulting movie file.
dpi : float, default: ``fig.dpi``
    The DPI (or resolution) for the file.  This controls the size
    in pixels of the resulting movie file.

### Function: frame_size(self)

**Description:** A tuple ``(width, height)`` in pixels of a movie frame.

### Function: _supports_transparency(self)

**Description:** Whether this writer supports transparency.

Writers may consult output file type and codec to determine this at runtime.

### Function: grab_frame(self)

**Description:** Grab the image information from the figure and save as a movie frame.

All keyword arguments in *savefig_kwargs* are passed on to the
`~.Figure.savefig` call that saves the figure.  However, several
keyword arguments that are supported by `~.Figure.savefig` may not be
passed as they are controlled by the MovieWriter:

- *dpi*, *bbox_inches*:  These may not be passed because each frame of the
   animation much be exactly the same size in pixels.
- *format*: This is controlled by the MovieWriter.

### Function: finish(self)

**Description:** Finish any processing for writing the movie.

### Function: saving(self, fig, outfile, dpi)

**Description:** Context manager to facilitate writing the movie file.

``*args, **kw`` are any parameters that should be passed to `setup`.

### Function: __init__(self, fps, codec, bitrate, extra_args, metadata)

**Description:** Parameters
----------
fps : int, default: 5
    Movie frame rate (per second).
codec : str or None, default: :rc:`animation.codec`
    The codec to use.
bitrate : int, default: :rc:`animation.bitrate`
    The bitrate of the movie, in kilobits per second.  Higher values
    means higher quality movies, but increase the file size.  A value
    of -1 lets the underlying movie encoder select the bitrate.
extra_args : list of str or None, optional
    Extra command-line arguments passed to the underlying movie encoder. These
    arguments are passed last to the encoder, just before the filename. The
    default, None, means to use :rc:`animation.[name-of-encoder]_args` for the
    builtin writers.
metadata : dict[str, str], default: {}
    A dictionary of keys and values for metadata to include in the
    output file. Some keys that may be of use include:
    title, artist, genre, subject, copyright, srcform, comment.

### Function: _adjust_frame_size(self)

### Function: setup(self, fig, outfile, dpi)

### Function: _run(self)

### Function: finish(self)

**Description:** Finish any processing for writing the movie.

### Function: grab_frame(self)

### Function: _args(self)

**Description:** Assemble list of encoder-specific command-line arguments.

### Function: bin_path(cls)

**Description:** Return the binary path to the commandline tool used by a specific
subclass. This is a class method so that the tool can be looked for
before making a particular MovieWriter subclass available.

### Function: isAvailable(cls)

**Description:** Return whether a MovieWriter subclass is actually available.

### Function: __init__(self)

### Function: setup(self, fig, outfile, dpi, frame_prefix)

**Description:** Setup for writing the movie file.

Parameters
----------
fig : `~matplotlib.figure.Figure`
    The figure to grab the rendered frames from.
outfile : str
    The filename of the resulting movie file.
dpi : float, default: ``fig.dpi``
    The dpi of the output file. This, with the figure size,
    controls the size in pixels of the resulting movie file.
frame_prefix : str, optional
    The filename prefix to use for temporary files.  If *None* (the
    default), files are written to a temporary directory which is
    deleted by `finish`; if not *None*, no temporary files are
    deleted.

### Function: __del__(self)

### Function: frame_format(self)

**Description:** Format (png, jpeg, etc.) to use for saving the frames, which can be
decided by the individual subclasses.

### Function: frame_format(self, frame_format)

### Function: _base_temp_name(self)

### Function: grab_frame(self)

### Function: finish(self)

### Function: _supports_transparency(self)

### Function: isAvailable(cls)

### Function: setup(self, fig, outfile, dpi)

### Function: grab_frame(self)

### Function: finish(self)

### Function: _supports_transparency(self)

### Function: output_args(self)

### Function: _args(self)

### Function: _args(self)

### Function: _supports_transparency(self)

### Function: _args(self)

### Function: bin_path(cls)

### Function: isAvailable(cls)

### Function: isAvailable(cls)

### Function: __init__(self, fps, codec, bitrate, extra_args, metadata, embed_frames, default_mode, embed_limit)

### Function: setup(self, fig, outfile, dpi, frame_dir)

### Function: grab_frame(self)

### Function: finish(self)

### Function: __init__(self, fig, event_source, blit)

### Function: __del__(self)

### Function: _start(self)

**Description:** Starts interactive animation. Adds the draw frame command to the GUI
handler, calls show to start the event loop.

### Function: _stop(self)

### Function: save(self, filename, writer, fps, dpi, codec, bitrate, extra_args, metadata, extra_anim, savefig_kwargs)

**Description:** Save the animation as a movie file by drawing every frame.

Parameters
----------
filename : str
    The output filename, e.g., :file:`mymovie.mp4`.

writer : `MovieWriter` or str, default: :rc:`animation.writer`
    A `MovieWriter` instance to use or a key that identifies a
    class to use, such as 'ffmpeg'.

fps : int, optional
    Movie frame rate (per second).  If not set, the frame rate from the
    animation's frame interval.

dpi : float, default: :rc:`savefig.dpi`
    Controls the dots per inch for the movie frames.  Together with
    the figure's size in inches, this controls the size of the movie.

codec : str, default: :rc:`animation.codec`.
    The video codec to use.  Not all codecs are supported by a given
    `MovieWriter`.

bitrate : int, default: :rc:`animation.bitrate`
    The bitrate of the movie, in kilobits per second.  Higher values
    means higher quality movies, but increase the file size.  A value
    of -1 lets the underlying movie encoder select the bitrate.

extra_args : list of str or None, optional
    Extra command-line arguments passed to the underlying movie encoder. These
    arguments are passed last to the encoder, just before the output filename.
    The default, None, means to use :rc:`animation.[name-of-encoder]_args` for
    the builtin writers.

metadata : dict[str, str], default: {}
    Dictionary of keys and values for metadata to include in
    the output file. Some keys that may be of use include:
    title, artist, genre, subject, copyright, srcform, comment.

extra_anim : list, default: []
    Additional `Animation` objects that should be included
    in the saved movie file. These need to be from the same
    `.Figure` instance. Also, animation frames will
    just be simply combined, so there should be a 1:1 correspondence
    between the frames from the different animations.

savefig_kwargs : dict, default: {}
    Keyword arguments passed to each `~.Figure.savefig` call used to
    save the individual frames.

progress_callback : function, optional
    A callback function that will be called for every frame to notify
    the saving progress. It must have the signature ::

        def func(current_frame: int, total_frames: int) -> Any

    where *current_frame* is the current frame number and *total_frames* is the
    total number of frames to be saved. *total_frames* is set to None, if the
    total number of frames cannot be determined. Return values may exist but are
    ignored.

    Example code to write the progress to stdout::

        progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')

Notes
-----
*fps*, *codec*, *bitrate*, *extra_args* and *metadata* are used to
construct a `.MovieWriter` instance and can only be passed if
*writer* is a string.  If they are passed as non-*None* and *writer*
is a `.MovieWriter`, a `RuntimeError` will be raised.

### Function: _step(self)

**Description:** Handler for getting events. By default, gets the next frame in the
sequence and hands the data off to be drawn.

### Function: new_frame_seq(self)

**Description:** Return a new sequence of frame information.

### Function: new_saved_frame_seq(self)

**Description:** Return a new sequence of saved/cached frame information.

### Function: _draw_next_frame(self, framedata, blit)

### Function: _init_draw(self)

### Function: _pre_draw(self, framedata, blit)

### Function: _draw_frame(self, framedata)

### Function: _post_draw(self, framedata, blit)

### Function: _blit_draw(self, artists)

### Function: _blit_clear(self, artists)

### Function: _setup_blit(self)

### Function: _on_resize(self, event)

### Function: _end_redraw(self, event)

### Function: to_html5_video(self, embed_limit)

**Description:** Convert the animation to an HTML5 ``<video>`` tag.

This saves the animation as an h264 video, encoded in base64
directly into the HTML5 video tag. This respects :rc:`animation.writer`
and :rc:`animation.bitrate`. This also makes use of the
*interval* to control the speed, and uses the *repeat*
parameter to decide whether to loop.

Parameters
----------
embed_limit : float, optional
    Limit, in MB, of the returned animation. No animation is created
    if the limit is exceeded.
    Defaults to :rc:`animation.embed_limit` = 20.0.

Returns
-------
str
    An HTML5 video tag with the animation embedded as base64 encoded
    h264 video.
    If the *embed_limit* is exceeded, this returns the string
    "Video too large to embed."

### Function: to_jshtml(self, fps, embed_frames, default_mode)

**Description:** Generate HTML representation of the animation.

Parameters
----------
fps : int, optional
    Movie frame rate (per second). If not set, the frame rate from
    the animation's frame interval.
embed_frames : bool, optional
default_mode : str, optional
    What to do when the animation ends. Must be one of ``{'loop',
    'once', 'reflect'}``. Defaults to ``'loop'`` if the *repeat*
    parameter is True, otherwise ``'once'``.

Returns
-------
str
    An HTML representation of the animation embedded as a js object as
    produced with the `.HTMLWriter`.

### Function: _repr_html_(self)

**Description:** IPython display hook for rendering.

### Function: pause(self)

**Description:** Pause the animation.

### Function: resume(self)

**Description:** Resume the animation.

### Function: __init__(self, fig, interval, repeat_delay, repeat, event_source)

### Function: _step(self)

**Description:** Handler for getting events.

### Function: __init__(self, fig, artists)

### Function: _init_draw(self)

### Function: _pre_draw(self, framedata, blit)

**Description:** Clears artists from the last frame.

### Function: _draw_frame(self, artists)

### Function: __init__(self, fig, func, frames, init_func, fargs, save_count)

### Function: new_frame_seq(self)

### Function: new_saved_frame_seq(self)

### Function: _init_draw(self)

### Function: _draw_frame(self, framedata)

### Function: wrapper(writer_cls)

### Function: _pre_composite_to_white(color)

### Function: gen()

### Function: iter_frames(frames)
