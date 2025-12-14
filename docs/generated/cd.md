## AI Summary

A file named cd.py.


### Function: encoding_unicode_range(iana_name)

**Description:** Return associated unicode ranges in a single byte code page.

### Function: unicode_range_languages(primary_range)

**Description:** Return inferred languages used with a unicode range.

### Function: encoding_languages(iana_name)

**Description:** Single-byte encoding language association. Some code page are heavily linked to particular language(s).
This function does the correspondence.

### Function: mb_encoding_languages(iana_name)

**Description:** Multi-byte encoding language association. Some code page are heavily linked to particular language(s).
This function does the correspondence.

### Function: get_target_features(language)

**Description:** Determine main aspects from a supported language if it contains accents and if is pure Latin.

### Function: alphabet_languages(characters, ignore_non_latin)

**Description:** Return associated languages associated to given characters.

### Function: characters_popularity_compare(language, ordered_characters)

**Description:** Determine if a ordered characters list (by occurrence from most appearance to rarest) match a particular language.
The result is a ratio between 0. (absolutely no correspondence) and 1. (near perfect fit).
Beware that is function is not strict on the match in order to ease the detection. (Meaning close match is 1.)

### Function: alpha_unicode_split(decoded_sequence)

**Description:** Given a decoded text sequence, return a list of str. Unicode range / alphabet separation.
Ex. a text containing English/Latin with a bit a Hebrew will return two items in the resulting list;
One containing the latin letters and the other hebrew.

### Function: merge_coherence_ratios(results)

**Description:** This function merge results previously given by the function coherence_ratio.
The return type is the same as coherence_ratio.

### Function: filter_alt_coherence_matches(results)

**Description:** We shall NOT return "English—" in CoherenceMatches because it is an alternative
of "English". This function only keeps the best match and remove the em-dash in it.

### Function: coherence_ratio(decoded_sequence, threshold, lg_inclusion)

**Description:** Detect ANY language that can be identified in given sequence. The sequence will be analysed by layers.
A layer = Character extraction by alphabets/ranges.
