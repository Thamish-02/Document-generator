## AI Summary

A file named semver.py.


## Class: _R

## Class: Extendlist

### Function: list_get(xs, i)

### Function: parse(version, loose)

### Function: valid(version, loose)

### Function: clean(version, loose)

### Function: semver(version, loose)

## Class: SemVer

### Function: inc(version, release, loose, identifier)

### Function: compare_identifiers(a, b)

### Function: rcompare_identifiers(a, b)

### Function: compare(a, b, loose)

### Function: compare_loose(a, b)

### Function: rcompare(a, b, loose)

### Function: make_key_function(loose)

### Function: sort(list_, loose)

### Function: rsort(list_, loose)

### Function: gt(a, b, loose)

### Function: lt(a, b, loose)

### Function: eq(a, b, loose)

### Function: neq(a, b, loose)

### Function: gte(a, b, loose)

### Function: lte(a, b, loose)

### Function: cmp(a, op, b, loose)

### Function: comparator(comp, loose)

## Class: Comparator

### Function: make_range(range_, loose)

## Class: Range

### Function: to_comparators(range_, loose)

### Function: parse_comparator(comp, loose)

### Function: is_x(id_)

### Function: replace_tildes(comp, loose)

### Function: replace_tilde(comp, loose)

### Function: replace_carets(comp, loose)

### Function: replace_caret(comp, loose)

### Function: replace_xranges(comp, loose)

### Function: replace_xrange(comp, loose)

### Function: replace_stars(comp, loose)

### Function: hyphen_replace(mob)

### Function: test_set(set_, version)

### Function: satisfies(version, range_, loose)

### Function: max_satisfying(versions, range_, loose)

### Function: valid_range(range_, loose)

### Function: ltr(version, range_, loose)

### Function: rtr(version, range_, loose)

### Function: outside(version, range_, hilo, loose)

### Function: __init__(self, i)

### Function: __call__(self)

### Function: value(self)

### Function: __setitem__(self, i, v)

### Function: __init__(self, version, loose)

### Function: format(self)

### Function: __repr__(self)

### Function: __str__(self)

### Function: compare(self, other)

### Function: compare_main(self, other)

### Function: compare_pre(self, other)

### Function: inc(self, release, identifier)

### Function: key_function(version)

### Function: __init__(self, comp, loose)

### Function: parse(self, comp)

### Function: __repr__(self)

### Function: __str__(self)

### Function: test(self, version)

### Function: __init__(self, range_, loose)

### Function: __repr__(self)

### Function: format(self)

### Function: __str__(self)

### Function: parse_range(self, range_)

### Function: test(self, version)

### Function: repl(mob)

### Function: repl(mob)

### Function: repl(mob)
