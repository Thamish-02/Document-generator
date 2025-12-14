## AI Summary

A file named editorhooks.py.


### Function: install_editor(template, wait)

**Description:** Installs the editor that is called by IPython for the %edit magic.

This overrides the default editor, which is generally set by your EDITOR
environment variable or is notepad (windows) or vi (linux). By supplying a
template string `run_template`, you can control how the editor is invoked
by IPython -- (e.g. the format in which it accepts command line options)

Parameters
----------
template : basestring
    run_template acts as a template for how your editor is invoked by
    the shell. It should contain '{filename}', which will be replaced on
    invocation with the file name, and '{line}', $line by line number
    (or 0) to invoke the file with.
wait : bool
    If `wait` is true, wait until the user presses enter before returning,
    to facilitate non-blocking editors that exit immediately after
    the call.

### Function: komodo(exe)

**Description:** Activestate Komodo [Edit] 

### Function: scite(exe)

**Description:** SciTE or Sc1 

### Function: notepadplusplus(exe)

**Description:** Notepad++ http://notepad-plus.sourceforge.net 

### Function: jed(exe)

**Description:** JED, the lightweight emacsish editor 

### Function: idle(exe)

**Description:** Idle, the editor bundled with python

Parameters
----------
exe : str, None
    If none, should be pretty smart about finding the executable.

### Function: mate(exe)

**Description:** TextMate, the missing editor

### Function: emacs(exe)

### Function: gnuclient(exe)

### Function: crimson_editor(exe)

### Function: kate(exe)

### Function: call_editor(self, filename, line)
