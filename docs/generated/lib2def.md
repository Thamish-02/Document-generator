## AI Summary

A file named lib2def.py.


### Function: parse_cmd()

**Description:** Parses the command-line arguments.

libfile, deffile = parse_cmd()

### Function: getnm(nm_cmd, shell)

**Description:** Returns the output of nm_cmd via a pipe.

nm_output = getnm(nm_cmd = 'nm -Cs py_lib')

### Function: parse_nm(nm_output)

**Description:** Returns a tuple of lists: dlist for the list of data
symbols and flist for the list of function symbols.

dlist, flist = parse_nm(nm_output)

### Function: output_def(dlist, flist, header, file)

**Description:** Outputs the final DEF file to a file defaulting to stdout.

output_def(dlist, flist, header, file = sys.stdout)
