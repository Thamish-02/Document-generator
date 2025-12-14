## AI Summary

A file named symbolic.py.


## Class: Language

**Description:** Used as Expr.tostring language argument.

## Class: Op

**Description:** Used as Expr op attribute.

## Class: RelOp

**Description:** Used in Op.RELATIONAL expression to specify the function part.

## Class: ArithOp

**Description:** Used in Op.APPLY expression to specify the function part.

## Class: OpError

## Class: Precedence

**Description:** Used as Expr.tostring precedence argument.

### Function: _pairs_add(d, k, v)

## Class: ExprWarning

### Function: ewarn(message)

## Class: Expr

**Description:** Represents a Fortran expression as a op-data pair.

Expr instances are hashable and sortable.

### Function: normalize(obj)

**Description:** Normalize Expr and apply basic evaluation methods.
    

### Function: as_expr(obj)

**Description:** Convert non-Expr objects to Expr objects.
    

### Function: as_symbol(obj)

**Description:** Return object as SYMBOL expression (variable or unparsed expression).
    

### Function: as_number(obj, kind)

**Description:** Return object as INTEGER or REAL constant.
    

### Function: as_integer(obj, kind)

**Description:** Return object as INTEGER constant.
    

### Function: as_real(obj, kind)

**Description:** Return object as REAL constant.
    

### Function: as_string(obj, kind)

**Description:** Return object as STRING expression (string literal constant).
    

### Function: as_array(obj)

**Description:** Return object as ARRAY expression (array constant).
    

### Function: as_complex(real, imag)

**Description:** Return object as COMPLEX expression (complex literal constant).
    

### Function: as_apply(func)

**Description:** Return object as APPLY expression (function call, constructor, etc.)
    

### Function: as_ternary(cond, expr1, expr2)

**Description:** Return object as TERNARY expression (cond?expr1:expr2).
    

### Function: as_ref(expr)

**Description:** Return object as referencing expression.
    

### Function: as_deref(expr)

**Description:** Return object as dereferencing expression.
    

### Function: as_eq(left, right)

### Function: as_ne(left, right)

### Function: as_lt(left, right)

### Function: as_le(left, right)

### Function: as_gt(left, right)

### Function: as_ge(left, right)

### Function: as_terms(obj)

**Description:** Return expression as TERMS expression.
    

### Function: as_factors(obj)

**Description:** Return expression as FACTORS expression.
    

### Function: as_term_coeff(obj)

**Description:** Return expression as term-coefficient pair.
    

### Function: as_numer_denom(obj)

**Description:** Return expression as numer-denom pair.
    

### Function: _counter()

### Function: eliminate_quotes(s)

**Description:** Replace quoted substrings of input string.

Return a new string and a mapping of replacements.

### Function: insert_quotes(s, d)

**Description:** Inverse of eliminate_quotes.
    

### Function: replace_parenthesis(s)

**Description:** Replace substrings of input that are enclosed in parenthesis.

Return a new string and a mapping of replacements.

### Function: _get_parenthesis_kind(s)

### Function: unreplace_parenthesis(s, d)

**Description:** Inverse of replace_parenthesis.
    

### Function: fromstring(s, language)

**Description:** Create an expression from a string.

This is a "lazy" parser, that is, only arithmetic operations are
resolved, non-arithmetic operations are treated as symbols.

## Class: _Pair

## Class: _FromStringWorker

### Function: fromstring(cls, s, language)

### Function: tostring(self, language)

### Function: parse(s, language)

**Description:** Parse a Fortran expression to a Expr.
        

### Function: __init__(self, op, data)

### Function: __eq__(self, other)

### Function: __hash__(self)

### Function: __lt__(self, other)

### Function: __le__(self, other)

### Function: __gt__(self, other)

### Function: __ge__(self, other)

### Function: __repr__(self)

### Function: __str__(self)

### Function: tostring(self, parent_precedence, language)

**Description:** Return a string representation of Expr.
        

### Function: __pos__(self)

### Function: __neg__(self)

### Function: __add__(self, other)

### Function: __radd__(self, other)

### Function: __sub__(self, other)

### Function: __rsub__(self, other)

### Function: __mul__(self, other)

### Function: __rmul__(self, other)

### Function: __pow__(self, other)

### Function: __truediv__(self, other)

### Function: __rtruediv__(self, other)

### Function: __floordiv__(self, other)

### Function: __rfloordiv__(self, other)

### Function: __call__(self)

### Function: __getitem__(self, index)

### Function: substitute(self, symbols_map)

**Description:** Recursively substitute symbols with values in symbols map.

Symbols map is a dictionary of symbol-expression pairs.

### Function: traverse(self, visit)

**Description:** Traverse expression tree with visit function.

The visit function is applied to an expression with given args
and kwargs.

Traverse call returns an expression returned by visit when not
None, otherwise return a new normalized expression with
traverse-visit sub-expressions.

### Function: contains(self, other)

**Description:** Check if self contains other.
        

### Function: symbols(self)

**Description:** Return a set of symbols contained in self.
        

### Function: polynomial_atoms(self)

**Description:** Return a set of expressions used as atoms in polynomial self.
        

### Function: linear_solve(self, symbol)

**Description:** Return a, b such that a * symbol + b == self.

If self is not linear with respect to symbol, raise RuntimeError.

### Function: repl(m)

### Function: __init__(self, left, right)

### Function: substitute(self, symbols_map)

### Function: __repr__(self)

### Function: __init__(self, language)

### Function: finalize_string(self, s)

### Function: parse(self, inp)

### Function: process(self, s, context)

**Description:** Parse string within the given context.

The context may define the result in case of ambiguous
expressions. For instance, consider expressions `f(x, y)` and
`(x, y) + (a, b)` where `f` is a function and pair `(x, y)`
denotes complex number. Specifying context as "args" or
"expr", the subexpression `(x, y)` will be parse to an
argument list or to a complex number, respectively.

### Function: visit(expr, found)

### Function: visit(expr, found)

### Function: visit(expr, found)

### Function: restore(r)
