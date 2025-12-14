## AI Summary

A file named md.py.


## Class: MessDetectorPlugin

**Description:** Base abstract class used for mess detection plugins.
All detectors MUST extend and implement given methods.

## Class: TooManySymbolOrPunctuationPlugin

## Class: TooManyAccentuatedPlugin

## Class: UnprintablePlugin

## Class: SuspiciousDuplicateAccentPlugin

## Class: SuspiciousRange

## Class: SuperWeirdWordPlugin

## Class: CjkUncommonPlugin

**Description:** Detect messy CJK text that probably means nothing.

## Class: ArchaicUpperLowerPlugin

## Class: ArabicIsolatedFormPlugin

### Function: is_suspiciously_successive_range(unicode_range_a, unicode_range_b)

**Description:** Determine if two Unicode range seen next to each other can be considered as suspicious.

### Function: mess_ratio(decoded_sequence, maximum_threshold, debug)

**Description:** Compute a mess ratio given a decoded bytes sequence. The maximum threshold does stop the computation earlier.

### Function: eligible(self, character)

**Description:** Determine if given character should be fed in.

### Function: feed(self, character)

**Description:** The main routine to be executed upon character.
Insert the logic in witch the text would be considered chaotic.

### Function: reset(self)

**Description:** Permit to reset the plugin to the initial state.

### Function: ratio(self)

**Description:** Compute the chaos ratio based on what your feed() has seen.
Must NOT be lower than 0.; No restriction gt 0.

### Function: __init__(self)

### Function: eligible(self, character)

### Function: feed(self, character)

### Function: reset(self)

### Function: ratio(self)

### Function: __init__(self)

### Function: eligible(self, character)

### Function: feed(self, character)

### Function: reset(self)

### Function: ratio(self)

### Function: __init__(self)

### Function: eligible(self, character)

### Function: feed(self, character)

### Function: reset(self)

### Function: ratio(self)

### Function: __init__(self)

### Function: eligible(self, character)

### Function: feed(self, character)

### Function: reset(self)

### Function: ratio(self)

### Function: __init__(self)

### Function: eligible(self, character)

### Function: feed(self, character)

### Function: reset(self)

### Function: ratio(self)

### Function: __init__(self)

### Function: eligible(self, character)

### Function: feed(self, character)

### Function: reset(self)

### Function: ratio(self)

### Function: __init__(self)

### Function: eligible(self, character)

### Function: feed(self, character)

### Function: reset(self)

### Function: ratio(self)

### Function: __init__(self)

### Function: eligible(self, character)

### Function: feed(self, character)

### Function: reset(self)

### Function: ratio(self)

### Function: __init__(self)

### Function: reset(self)

### Function: eligible(self, character)

### Function: feed(self, character)

### Function: ratio(self)
