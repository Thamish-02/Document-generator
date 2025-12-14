## AI Summary

A file named locales.py.


### Function: get_locale(name)

**Description:** Returns an appropriate :class:`Locale <arrow.locales.Locale>`
corresponding to an input locale name.

:param name: the name of the locale.

### Function: get_locale_by_class_name(name)

**Description:** Returns an appropriate :class:`Locale <arrow.locales.Locale>`
corresponding to an locale class name.

:param name: the name of the locale class.

## Class: Locale

**Description:** Represents locale-specific data and functionality.

## Class: EnglishLocale

## Class: ItalianLocale

## Class: SpanishLocale

## Class: FrenchBaseLocale

## Class: FrenchLocale

## Class: FrenchCanadianLocale

## Class: GreekLocale

## Class: JapaneseLocale

## Class: SwedishLocale

## Class: FinnishLocale

## Class: ChineseCNLocale

## Class: ChineseTWLocale

## Class: HongKongLocale

## Class: KoreanLocale

## Class: DutchLocale

## Class: SlavicBaseLocale

## Class: BelarusianLocale

## Class: PolishLocale

## Class: RussianLocale

## Class: AfrikaansLocale

## Class: BulgarianLocale

## Class: UkrainianLocale

## Class: MacedonianLocale

## Class: MacedonianLatinLocale

## Class: GermanBaseLocale

## Class: GermanLocale

## Class: SwissLocale

## Class: AustrianLocale

## Class: NorwegianLocale

## Class: NewNorwegianLocale

## Class: PortugueseLocale

## Class: BrazilianPortugueseLocale

## Class: TagalogLocale

## Class: VietnameseLocale

## Class: TurkishLocale

## Class: AzerbaijaniLocale

## Class: ArabicLocale

## Class: LevantArabicLocale

## Class: AlgeriaTunisiaArabicLocale

## Class: MauritaniaArabicLocale

## Class: MoroccoArabicLocale

## Class: IcelandicLocale

## Class: DanishLocale

## Class: MalayalamLocale

## Class: HindiLocale

## Class: CzechLocale

## Class: SlovakLocale

## Class: FarsiLocale

## Class: HebrewLocale

## Class: MarathiLocale

## Class: CatalanLocale

## Class: BasqueLocale

## Class: HungarianLocale

## Class: EsperantoLocale

## Class: ThaiLocale

## Class: LaotianLocale

## Class: BengaliLocale

## Class: RomanshLocale

## Class: RomanianLocale

## Class: SlovenianLocale

## Class: IndonesianLocale

## Class: NepaliLocale

## Class: EstonianLocale

## Class: LatvianLocale

## Class: SwahiliLocale

## Class: CroatianLocale

## Class: LatinLocale

## Class: LithuanianLocale

## Class: MalayLocale

## Class: MalteseLocale

## Class: SamiLocale

## Class: OdiaLocale

## Class: SerbianLocale

## Class: LuxembourgishLocale

## Class: ZuluLocale

## Class: TamilLocale

## Class: AlbanianLocale

## Class: GeorgianLocale

## Class: SinhalaLocale

## Class: UrduLocale

## Class: KazakhLocale

## Class: AmharicLocale

## Class: ArmenianLocale

## Class: UzbekLocale

### Function: __init_subclass__(cls)

### Function: __init__(self)

### Function: describe(self, timeframe, delta, only_distance)

**Description:** Describes a delta within a timeframe in plain language.

:param timeframe: a string representing a timeframe.
:param delta: a quantity representing a delta in a timeframe.
:param only_distance: return only distance eg: "11 seconds" without "in" or "ago" keywords

### Function: describe_multi(self, timeframes, only_distance)

**Description:** Describes a delta within multiple timeframes in plain language.

:param timeframes: a list of string, quantity pairs each representing a timeframe and delta.
:param only_distance: return only distance eg: "2 hours and 11 seconds" without "in" or "ago" keywords

### Function: day_name(self, day)

**Description:** Returns the day name for a specified day of the week.

:param day: the ``int`` day of the week (1-7).

### Function: day_abbreviation(self, day)

**Description:** Returns the day abbreviation for a specified day of the week.

:param day: the ``int`` day of the week (1-7).

### Function: month_name(self, month)

**Description:** Returns the month name for a specified month of the year.

:param month: the ``int`` month of the year (1-12).

### Function: month_abbreviation(self, month)

**Description:** Returns the month abbreviation for a specified month of the year.

:param month: the ``int`` month of the year (1-12).

### Function: month_number(self, name)

**Description:** Returns the month number for a month specified by name or abbreviation.

:param name: the month name or abbreviation.

### Function: year_full(self, year)

**Description:** Returns the year for specific locale if available

:param year: the ``int`` year (4-digit)

### Function: year_abbreviation(self, year)

**Description:** Returns the year for specific locale if available

:param year: the ``int`` year (4-digit)

### Function: meridian(self, hour, token)

**Description:** Returns the meridian indicator for a specified hour and format token.

:param hour: the ``int`` hour of the day.
:param token: the format token.

### Function: ordinal_number(self, n)

**Description:** Returns the ordinal format of a given integer

:param n: an integer

### Function: _ordinal_number(self, n)

### Function: _name_to_ordinal(self, lst)

### Function: _format_timeframe(self, timeframe, delta)

### Function: _format_relative(self, humanized, timeframe, delta)

### Function: _ordinal_number(self, n)

### Function: describe(self, timeframe, delta, only_distance)

**Description:** Describes a delta within a timeframe in plain language.

:param timeframe: a string representing a timeframe.
:param delta: a quantity representing a delta in a timeframe.
:param only_distance: return only distance eg: "11 seconds" without "in" or "ago" keywords

### Function: _ordinal_number(self, n)

### Function: _ordinal_number(self, n)

### Function: _ordinal_number(self, n)

### Function: _format_timeframe(self, timeframe, delta)

### Function: _ordinal_number(self, n)

### Function: _ordinal_number(self, n)

### Function: _format_relative(self, humanized, timeframe, delta)

### Function: _format_timeframe(self, timeframe, delta)

### Function: _ordinal_number(self, n)

### Function: describe(self, timeframe, delta, only_distance)

**Description:** Describes a delta within a timeframe in plain language.

:param timeframe: a string representing a timeframe.
:param delta: a quantity representing a delta in a timeframe.
:param only_distance: return only distance eg: "11 seconds" without "in" or "ago" keywords

### Function: _ordinal_number(self, n)

### Function: _ordinal_number(self, n)

### Function: _ordinal_number(self, n)

### Function: _format_timeframe(self, timeframe, delta)

### Function: _format_timeframe(self, timeframe, delta)

### Function: _ordinal_number(self, n)

### Function: _format_timeframe(self, timeframe, delta)

**Description:** Czech aware time frame format function, takes into account
the differences between past and future forms.

### Function: _format_timeframe(self, timeframe, delta)

**Description:** Slovak aware time frame format function, takes into account
the differences between past and future forms.

### Function: _format_timeframe(self, timeframe, delta)

### Function: describe_multi(self, timeframes, only_distance)

**Description:** Describes a delta within multiple timeframes in plain language.
In Hebrew, the and word behaves a bit differently.

:param timeframes: a list of string, quantity pairs each representing a timeframe and delta.
:param only_distance: return only distance eg: "2 hours and 11 seconds" without "in" or "ago" keywords

### Function: _format_timeframe(self, timeframe, delta)

### Function: _ordinal_number(self, n)

### Function: year_full(self, year)

**Description:** Thai always use Buddhist Era (BE) which is CE + 543

### Function: year_abbreviation(self, year)

**Description:** Thai always use Buddhist Era (BE) which is CE + 543

### Function: _format_relative(self, humanized, timeframe, delta)

**Description:** Thai normally doesn't have any space between words

### Function: year_full(self, year)

**Description:** Lao always use Buddhist Era (BE) which is CE + 543

### Function: year_abbreviation(self, year)

**Description:** Lao always use Buddhist Era (BE) which is CE + 543

### Function: _format_relative(self, humanized, timeframe, delta)

**Description:** Lao normally doesn't have any space between words

### Function: _ordinal_number(self, n)

### Function: _format_timeframe(self, timeframe, delta)

### Function: _format_timeframe(self, timeframe, delta)

### Function: _format_timeframe(self, timeframe, delta)

### Function: _ordinal_number(self, n)

### Function: _format_timeframe(self, timeframe, delta)

### Function: _ordinal_number(self, n)

### Function: describe(self, timeframe, delta, only_distance)

### Function: _format_timeframe(self, timeframe, delta)

**Description:** Zulu aware time frame format function, takes into account
the differences between past and future forms.

### Function: _ordinal_number(self, n)

### Function: _format_timeframe(self, timeframe, delta)

**Description:** Sinhala awares time frame format function, takes into account
the differences between general, past, and future forms (three different suffixes).

### Function: describe(self, timeframe, delta, only_distance)

**Description:** Describes a delta within a timeframe in plain language.

:param timeframe: a string representing a timeframe.
:param delta: a quantity representing a delta in a timeframe.
:param only_distance: return only distance eg: "11 seconds" without "in" or "ago" keywords

### Function: _ordinal_number(self, n)

### Function: _format_timeframe(self, timeframe, delta)

**Description:** Amharic awares time frame format function, takes into account
the differences between general, past, and future forms (three different suffixes).

### Function: describe(self, timeframe, delta, only_distance)

**Description:** Describes a delta within a timeframe in plain language.

:param timeframe: a string representing a timeframe.
:param delta: a quantity representing a delta in a timeframe.
:param only_distance: return only distance eg: "11 seconds" without "in" or "ago" keywords
