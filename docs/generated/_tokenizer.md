## AI Summary

A file named _tokenizer.py.


## Class: HTMLTokenizer

**Description:** This class takes care of tokenizing HTML.

* self.currentToken
  Holds the token that is currently being processed.

* self.state
  Holds a reference to the method to be invoked... XXX

* self.stream
  Points to HTMLInputStream object.

### Function: __init__(self, stream, parser)

### Function: __iter__(self)

**Description:** This is where the magic happens.

We do our usually processing through the states and when we have a token
to return we yield the token which pauses processing until the next token
is requested.

### Function: consumeNumberEntity(self, isHex)

**Description:** This function returns either U+FFFD or the character based on the
decimal or hexadecimal representation. It also discards ";" if present.
If not present self.tokenQueue.append({"type": tokenTypes["ParseError"]}) is invoked.

### Function: consumeEntity(self, allowedChar, fromAttribute)

### Function: processEntityInAttribute(self, allowedChar)

**Description:** This method replaces the need for "entityInAttributeValueState".
        

### Function: emitCurrentToken(self)

**Description:** This method is a generic handler for emitting the tags. It also sets
the state to "data" because that's what's needed after a token has been
emitted.

### Function: dataState(self)

### Function: entityDataState(self)

### Function: rcdataState(self)

### Function: characterReferenceInRcdata(self)

### Function: rawtextState(self)

### Function: scriptDataState(self)

### Function: plaintextState(self)

### Function: tagOpenState(self)

### Function: closeTagOpenState(self)

### Function: tagNameState(self)

### Function: rcdataLessThanSignState(self)

### Function: rcdataEndTagOpenState(self)

### Function: rcdataEndTagNameState(self)

### Function: rawtextLessThanSignState(self)

### Function: rawtextEndTagOpenState(self)

### Function: rawtextEndTagNameState(self)

### Function: scriptDataLessThanSignState(self)

### Function: scriptDataEndTagOpenState(self)

### Function: scriptDataEndTagNameState(self)

### Function: scriptDataEscapeStartState(self)

### Function: scriptDataEscapeStartDashState(self)

### Function: scriptDataEscapedState(self)

### Function: scriptDataEscapedDashState(self)

### Function: scriptDataEscapedDashDashState(self)

### Function: scriptDataEscapedLessThanSignState(self)

### Function: scriptDataEscapedEndTagOpenState(self)

### Function: scriptDataEscapedEndTagNameState(self)

### Function: scriptDataDoubleEscapeStartState(self)

### Function: scriptDataDoubleEscapedState(self)

### Function: scriptDataDoubleEscapedDashState(self)

### Function: scriptDataDoubleEscapedDashDashState(self)

### Function: scriptDataDoubleEscapedLessThanSignState(self)

### Function: scriptDataDoubleEscapeEndState(self)

### Function: beforeAttributeNameState(self)

### Function: attributeNameState(self)

### Function: afterAttributeNameState(self)

### Function: beforeAttributeValueState(self)

### Function: attributeValueDoubleQuotedState(self)

### Function: attributeValueSingleQuotedState(self)

### Function: attributeValueUnQuotedState(self)

### Function: afterAttributeValueState(self)

### Function: selfClosingStartTagState(self)

### Function: bogusCommentState(self)

### Function: markupDeclarationOpenState(self)

### Function: commentStartState(self)

### Function: commentStartDashState(self)

### Function: commentState(self)

### Function: commentEndDashState(self)

### Function: commentEndState(self)

### Function: commentEndBangState(self)

### Function: doctypeState(self)

### Function: beforeDoctypeNameState(self)

### Function: doctypeNameState(self)

### Function: afterDoctypeNameState(self)

### Function: afterDoctypePublicKeywordState(self)

### Function: beforeDoctypePublicIdentifierState(self)

### Function: doctypePublicIdentifierDoubleQuotedState(self)

### Function: doctypePublicIdentifierSingleQuotedState(self)

### Function: afterDoctypePublicIdentifierState(self)

### Function: betweenDoctypePublicAndSystemIdentifiersState(self)

### Function: afterDoctypeSystemKeywordState(self)

### Function: beforeDoctypeSystemIdentifierState(self)

### Function: doctypeSystemIdentifierDoubleQuotedState(self)

### Function: doctypeSystemIdentifierSingleQuotedState(self)

### Function: afterDoctypeSystemIdentifierState(self)

### Function: bogusDoctypeState(self)

### Function: cdataSectionState(self)
