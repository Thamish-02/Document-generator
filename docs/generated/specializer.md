## AI Summary

A file named specializer.py.


### Function: stringToProgram(string)

### Function: programToString(program)

### Function: programToCommands(program, getNumRegions)

**Description:** Takes a T2CharString program list and returns list of commands.
Each command is a two-tuple of commandname,arg-list.  The commandname might
be empty string if no commandname shall be emitted (used for glyph width,
hintmask/cntrmask argument, as well as stray arguments at the end of the
program (🤷).
'getNumRegions' may be None, or a callable object. It must return the
number of regions. 'getNumRegions' takes a single argument, vsindex. It
returns the numRegions for the vsindex.
The Charstring may or may not start with a width value. If the first
non-blend operator has an odd number of arguments, then the first argument is
a width, and is popped off. This is complicated with blend operators, as
there may be more than one before the first hint or moveto operator, and each
one reduces several arguments to just one list argument. We have to sum the
number of arguments that are not part of the blend arguments, and all the
'numBlends' values. We could instead have said that by definition, if there
is a blend operator, there is no width value, since CFF2 Charstrings don't
have width values. I discussed this with Behdad, and we are allowing for an
initial width value in this case because developers may assemble a CFF2
charstring from CFF Charstrings, which could have width values.

### Function: _flattenBlendArgs(args)

### Function: commandsToProgram(commands)

**Description:** Takes a commands list as returned by programToCommands() and converts
it back to a T2CharString program list.

### Function: _everyN(el, n)

**Description:** Group the list el into groups of size n

## Class: _GeneralizerDecombinerCommandsMap

### Function: _convertBlendOpToArgs(blendList)

### Function: generalizeCommands(commands, ignoreErrors)

### Function: generalizeProgram(program, getNumRegions)

### Function: _categorizeVector(v)

**Description:** Takes X,Y vector v and returns one of r, h, v, or 0 depending on which
of X and/or Y are zero, plus tuple of nonzero ones.  If both are zero,
it returns a single zero still.

>>> _categorizeVector((0,0))
('0', (0,))
>>> _categorizeVector((1,0))
('h', (1,))
>>> _categorizeVector((0,2))
('v', (2,))
>>> _categorizeVector((1,2))
('r', (1, 2))

### Function: _mergeCategories(a, b)

### Function: _negateCategory(a)

### Function: _convertToBlendCmds(args)

### Function: _addArgs(a, b)

### Function: _argsStackUse(args)

### Function: specializeCommands(commands, ignoreErrors, generalizeFirst, preserveTopology, maxstack)

### Function: specializeProgram(program, getNumRegions)

### Function: rmoveto(args)

### Function: hmoveto(args)

### Function: vmoveto(args)

### Function: rlineto(args)

### Function: hlineto(args)

### Function: vlineto(args)

### Function: rrcurveto(args)

### Function: hhcurveto(args)

### Function: vvcurveto(args)

### Function: hvcurveto(args)

### Function: vhcurveto(args)

### Function: rcurveline(args)

### Function: rlinecurve(args)
