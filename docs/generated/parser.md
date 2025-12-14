## AI Summary

A file named parser.py.


## Class: Parser

### Function: __init__(self, msg, fname, pos)

### Function: parse(self, global_vars)

### Function: _err_str(self)

### Function: _err_offsets(self)

### Function: _succeed(self, v, newpos)

### Function: _fail(self)

### Function: _rewind(self, newpos)

### Function: _bind(self, rule, var)

### Function: _not(self, rule)

### Function: _opt(self, rule)

### Function: _plus(self, rule)

### Function: _star(self, rule, vs)

### Function: _seq(self, rules)

### Function: _choose(self, rules)

### Function: _ch(self, ch)

### Function: _str(self, s)

### Function: _range(self, i, j)

### Function: _push(self, name)

### Function: _pop(self, name)

### Function: _get(self, var)

### Function: _set(self, var, val)

### Function: _is_unicat(self, var, cat)

### Function: _join(self, s, vs)

### Function: _xtou(self, s)

### Function: _grammar_(self)

### Function: _trailing_(self)

### Function: _trailing__c0_(self)

### Function: _trailing__c0__s0_(self)

### Function: _trailing__c1_(self)

### Function: _trailing__c1_n_(self)

### Function: _sp_(self)

### Function: _ws_(self)

### Function: _ws__c0_(self)

### Function: _ws__c3_(self)

### Function: _ws__c4_(self)

### Function: _ws__c5_(self)

### Function: _ws__c6_(self)

### Function: _ws__c7_(self)

### Function: _ws__c8_(self)

### Function: _ws__c8__s0_(self)

### Function: _ws__c8__s0_n_n_(self)

### Function: _ws__c8__s0_n_n_g__c0_(self)

### Function: _ws__c8__s0_n_n_g__c0__s1_(self)

### Function: _eol_(self)

### Function: _eol__c0_(self)

### Function: _eol__c1_(self)

### Function: _eol__c2_(self)

### Function: _eol__c3_(self)

### Function: _eol__c4_(self)

### Function: _comment_(self)

### Function: _comment__c0_(self)

### Function: _comment__c0__s1_p_(self)

### Function: _comment__c1_(self)

### Function: _comment__c1__s1_(self)

### Function: _comment__c1__s1_p__s0_(self)

### Function: _value_(self)

### Function: _value__c0_(self)

### Function: _value__c1_(self)

### Function: _value__c2_(self)

### Function: _value__c3_(self)

### Function: _value__c4_(self)

### Function: _value__c5_(self)

### Function: _value__c6_(self)

### Function: _object_(self)

### Function: _object__c0_(self)

### Function: _object__c1_(self)

### Function: _array_(self)

### Function: _array__c0_(self)

### Function: _array__c1_(self)

### Function: _string_(self)

### Function: _string__c0_(self)

### Function: _string__c0__s1_(self)

### Function: _string__c1_(self)

### Function: _string__c1__s1_(self)

### Function: _sqchar_(self)

### Function: _sqchar__c0_(self)

### Function: _sqchar__c1_(self)

### Function: _sqchar__c2_(self)

### Function: _sqchar__c3_(self)

### Function: _sqchar__c3__s0_n_(self)

### Function: _dqchar_(self)

### Function: _dqchar__c0_(self)

### Function: _dqchar__c1_(self)

### Function: _dqchar__c2_(self)

### Function: _dqchar__c3_(self)

### Function: _dqchar__c3__s0_n_(self)

### Function: _bslash_(self)

### Function: _squote_(self)

### Function: _dquote_(self)

### Function: _esc_char_(self)

### Function: _esc_char__c0_(self)

### Function: _esc_char__c1_(self)

### Function: _esc_char__c10_(self)

### Function: _esc_char__c11_(self)

### Function: _esc_char__c12_(self)

### Function: _esc_char__c2_(self)

### Function: _esc_char__c3_(self)

### Function: _esc_char__c4_(self)

### Function: _esc_char__c5_(self)

### Function: _esc_char__c6_(self)

### Function: _esc_char__c7_(self)

### Function: _esc_char__c8_(self)

### Function: _esc_char__c9_(self)

### Function: _esc_char__c9__s0_(self)

### Function: _esc_char__c9__s0_n_g_(self)

### Function: _esc_char__c9__s0_n_g__c0_(self)

### Function: _esc_char__c9__s0_n_g__c1_(self)

### Function: _hex_esc_(self)

### Function: _unicode_esc_(self)

### Function: _element_list_(self)

### Function: _element_list__s1_(self)

### Function: _element_list__s1_l_p_(self)

### Function: _element_list__s3_(self)

### Function: _member_list_(self)

### Function: _member_list__s1_(self)

### Function: _member_list__s1_l_p_(self)

### Function: _member_list__s3_(self)

### Function: _member_(self)

### Function: _member__c0_(self)

### Function: _member__c1_(self)

### Function: _ident_(self)

### Function: _ident__s1_(self)

### Function: _id_start_(self)

### Function: _id_start__c2_(self)

### Function: _ascii_id_start_(self)

### Function: _ascii_id_start__c0_(self)

### Function: _ascii_id_start__c1_(self)

### Function: _ascii_id_start__c2_(self)

### Function: _ascii_id_start__c3_(self)

### Function: _other_id_start_(self)

### Function: _other_id_start__c0_(self)

### Function: _other_id_start__c0__s1_(self)

### Function: _other_id_start__c1_(self)

### Function: _other_id_start__c1__s1_(self)

### Function: _other_id_start__c2_(self)

### Function: _other_id_start__c2__s1_(self)

### Function: _other_id_start__c3_(self)

### Function: _other_id_start__c3__s1_(self)

### Function: _other_id_start__c4_(self)

### Function: _other_id_start__c4__s1_(self)

### Function: _other_id_start__c5_(self)

### Function: _other_id_start__c5__s1_(self)

### Function: _id_continue_(self)

### Function: _id_continue__c3_(self)

### Function: _id_continue__c3__s1_(self)

### Function: _id_continue__c4_(self)

### Function: _id_continue__c4__s1_(self)

### Function: _id_continue__c5_(self)

### Function: _id_continue__c5__s1_(self)

### Function: _id_continue__c6_(self)

### Function: _id_continue__c6__s1_(self)

### Function: _id_continue__c7_(self)

### Function: _id_continue__c8_(self)

### Function: _id_continue__c9_(self)

### Function: _num_literal_(self)

### Function: _num_literal__c0_(self)

### Function: _num_literal__c1_(self)

### Function: _num_literal__c2_(self)

### Function: _unsigned_lit_(self)

### Function: _unsigned_lit__c0_(self)

### Function: _unsigned_lit__c2_(self)

### Function: _unsigned_lit__c3_(self)

### Function: _dec_literal_(self)

### Function: _dec_literal__c0_(self)

### Function: _dec_literal__c1_(self)

### Function: _dec_literal__c2_(self)

### Function: _dec_literal__c3_(self)

### Function: _dec_literal__c4_(self)

### Function: _dec_literal__c5_(self)

### Function: _dec_int_lit_(self)

### Function: _dec_int_lit__c0_(self)

### Function: _dec_int_lit__c1_(self)

### Function: _dec_int_lit__c1__s1_(self)

### Function: _digit_(self)

### Function: _nonzerodigit_(self)

### Function: _hex_literal_(self)

### Function: _hex_literal__s0_(self)

### Function: _hex_literal__s1_(self)

### Function: _hex_(self)

### Function: _hex__c0_(self)

### Function: _hex__c1_(self)

### Function: _frac_(self)

### Function: _frac__s1_(self)

### Function: _exp_(self)

### Function: _exp__c0_(self)

### Function: _exp__c0__s0_(self)

### Function: _exp__c0__s1_l_(self)

### Function: _exp__c0__s2_(self)

### Function: _exp__c1_(self)

### Function: _exp__c1__s0_(self)

### Function: _exp__c1__s1_(self)

### Function: _anything_(self)

### Function: _end_(self)
