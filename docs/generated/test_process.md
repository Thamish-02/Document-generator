## AI Summary

A file named test_process.py.


### Function: test_find_cmd_ls()

**Description:** Make sure we can find the full path to ls.

### Function: test_find_cmd_pythonw()

**Description:** Try to find pythonw on Windows.

### Function: test_find_cmd_fail()

**Description:** Make sure that FindCmdError is raised if we can't find the cmd.

### Function: test_arg_split(argstr, argv)

**Description:** Ensure that argument lines are correctly split like in a shell.

### Function: test_arg_split_win32(argstr, argv)

**Description:** Ensure that argument lines are correctly split like in a shell.

## Class: SubProcessTestCase

### Function: setUp(self)

**Description:** Make a valid python temp file.

### Function: test_system(self)

### Function: test_system_quotes(self)

### Function: assert_interrupts(self, command)

**Description:** Interrupt a subprocess after a second.

### Function: test_system_interrupt(self)

**Description:** When interrupted in the way ipykernel interrupts IPython, the
subprocess is interrupted.

### Function: test_getoutput(self)

### Function: test_getoutput_quoted(self)

### Function: test_getoutput_quoted2(self)

### Function: test_getoutput_error(self)

### Function: test_get_output_error_code(self)

### Function: interrupt()

### Function: command()
