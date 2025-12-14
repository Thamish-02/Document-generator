## AI Summary

A file named _type1font.py.


## Class: _Token

**Description:** A token in a PostScript stream.

Attributes
----------
pos : int
    Position, i.e. offset from the beginning of the data.
raw : str
    Raw text of the token.
kind : str
    Description of the token (for debugging or testing).

## Class: _NameToken

## Class: _BooleanToken

## Class: _KeywordToken

## Class: _DelimiterToken

## Class: _WhitespaceToken

## Class: _StringToken

## Class: _BinaryToken

## Class: _NumberToken

### Function: _tokenize(data, skip_ws)

**Description:** A generator that produces _Token instances from Type-1 font code.

The consumer of the generator may send an integer to the tokenizer to
indicate that the next token should be _BinaryToken of the given length.

Parameters
----------
data : bytes
    The data of the font to tokenize.

skip_ws : bool
    If true, the generator will drop any _WhitespaceTokens from the output.

## Class: _BalancedExpression

### Function: _expression(initial, tokens, data)

**Description:** Consume some number of tokens and return a balanced PostScript expression.

Parameters
----------
initial : _Token
    The token that triggered parsing a balanced expression.
tokens : iterator of _Token
    Following tokens.
data : bytes
    Underlying data that the token positions point to.

Returns
-------
_BalancedExpression

## Class: Type1Font

**Description:** A class representing a Type-1 font, for use by backends.

Attributes
----------
parts : tuple
    A 3-tuple of the cleartext part, the encrypted part, and the finale of
    zeros.

decrypted : bytes
    The decrypted form of ``parts[1]``.

prop : dict[str, Any]
    A dictionary of font properties. Noteworthy keys include:

    - FontName: PostScript name of the font
    - Encoding: dict from numeric codes to glyph names
    - FontMatrix: bytes object encoding a matrix
    - UniqueID: optional font identifier, dropped when modifying the font
    - CharStrings: dict from glyph names to byte code
    - Subrs: array of byte code subroutines
    - OtherSubrs: bytes object encoding some PostScript code

### Function: __init__(self, pos, raw)

### Function: __str__(self)

### Function: endpos(self)

**Description:** Position one past the end of the token

### Function: is_keyword(self)

**Description:** Is this a name token with one of the names?

### Function: is_slash_name(self)

**Description:** Is this a name token that starts with a slash?

### Function: is_delim(self)

**Description:** Is this a delimiter token?

### Function: is_number(self)

**Description:** Is this a number token?

### Function: value(self)

### Function: is_slash_name(self)

### Function: value(self)

### Function: value(self)

### Function: is_keyword(self)

### Function: is_delim(self)

### Function: opposite(self)

### Function: _escape(cls, match)

### Function: value(self)

### Function: value(self)

### Function: is_number(self)

### Function: value(self)

### Function: __init__(self, input)

**Description:** Initialize a Type-1 font.

Parameters
----------
input : str or 3-tuple
    Either a pfb file name, or a 3-tuple of already-decoded Type-1
    font `~.Type1Font.parts`.

### Function: _read(self, file)

**Description:** Read the font from a file, decoding into usable parts.

### Function: _split(self, data)

**Description:** Split the Type 1 font into its three main parts.

The three parts are: (1) the cleartext part, which ends in a
eexec operator; (2) the encrypted part; (3) the fixed part,
which contains 512 ASCII zeros possibly divided on various
lines, a cleartomark operator, and possibly something else.

### Function: _decrypt(ciphertext, key, ndiscard)

**Description:** Decrypt ciphertext using the Type-1 font algorithm.

The algorithm is described in Adobe's "Adobe Type 1 Font Format".
The key argument can be an integer, or one of the strings
'eexec' and 'charstring', which map to the key specified for the
corresponding part of Type-1 fonts.

The ndiscard argument should be an integer, usually 4.
That number of bytes is discarded from the beginning of plaintext.

### Function: _encrypt(plaintext, key, ndiscard)

**Description:** Encrypt plaintext using the Type-1 font algorithm.

The algorithm is described in Adobe's "Adobe Type 1 Font Format".
The key argument can be an integer, or one of the strings
'eexec' and 'charstring', which map to the key specified for the
corresponding part of Type-1 fonts.

The ndiscard argument should be an integer, usually 4. That
number of bytes is prepended to the plaintext before encryption.
This function prepends NUL bytes for reproducibility, even though
the original algorithm uses random bytes, presumably to avoid
cryptanalysis.

### Function: _parse(self)

**Description:** Find the values of various font properties. This limited kind
of parsing is described in Chapter 10 "Adobe Type Manager
Compatibility" of the Type-1 spec.

### Function: _parse_subrs(self, tokens, _data)

### Function: _parse_charstrings(tokens, _data)

### Function: _parse_encoding(tokens, _data)

### Function: _parse_othersubrs(tokens, data)

### Function: transform(self, effects)

**Description:** Return a new font that is slanted and/or extended.

Parameters
----------
effects : dict
    A dict with optional entries:

    - 'slant' : float, default: 0
        Tangent of the angle that the font is to be slanted to the
        right. Negative values slant to the left.
    - 'extend' : float, default: 1
        Scaling factor for the font width. Values less than 1 condense
        the glyphs.

Returns
-------
`Type1Font`
