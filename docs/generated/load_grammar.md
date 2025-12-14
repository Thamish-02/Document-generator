## AI Summary

A file named load_grammar.py.


## Class: FindRuleSize

## Class: EBNF_to_BNF

## Class: SimplifyRule_Visitor

## Class: RuleTreeToText

## Class: PrepareAnonTerminals

**Description:** Create a unique list of anonymous terminals. Attempt to give meaningful names to them when we add them

## Class: _ReplaceSymbols

**Description:** Helper for ApplyTemplates

## Class: ApplyTemplates

**Description:** Apply the templates, creating new rules that represent the used templates

### Function: _rfind(s, choices)

### Function: eval_escaping(s)

### Function: _literal_to_pattern(literal)

## Class: PrepareLiterals

### Function: _make_joined_pattern(regexp, flags_set)

## Class: TerminalTreeToPattern

## Class: ValidateSymbols

### Function: nr_deepcopy_tree(t)

**Description:** Deepcopy tree `t` without recursion

## Class: Grammar

## Class: FromPackageLoader

**Description:** Provides a simple way of creating custom import loaders that load from packages via ``pkgutil.get_data`` instead of using `open`.
This allows them to be compatible even from within zip files.

Relative imports are handled, so you can just freely use them.

pkg_name: The name of the package. You can probably provide `__name__` most of the time
search_paths: All the path that will be search on absolute imports.

### Function: resolve_term_references(term_dict)

### Function: symbol_from_strcase(s)

## Class: PrepareGrammar

### Function: _find_used_symbols(tree)

### Function: _get_parser()

### Function: _translate_parser_exception(parse, e)

### Function: _parse_grammar(text, name, start)

### Function: _error_repr(error)

### Function: _search_interactive_parser(interactive_parser, predicate)

### Function: find_grammar_errors(text, start)

### Function: _get_mangle(prefix, aliases, base_mangle)

### Function: _mangle_definition_tree(exp, mangle)

### Function: _make_rule_tuple(modifiers_tree, name, params, priority_tree, expansions)

## Class: Definition

## Class: GrammarBuilder

### Function: verify_used_files(file_hashes)

### Function: list_grammar_imports(grammar, import_paths)

**Description:** Returns a list of paths to the lark grammars imported by the given grammar (recursively)

### Function: load_grammar(grammar, source, import_paths, global_keep_all_tokens)

### Function: sha256_digest(s)

**Description:** Get the sha256 digest of a string

Supports the `usedforsecurity` argument for Python 3.9+ to allow running on
a FIPS-enabled system.

### Function: __init__(self, keep_all_tokens)

### Function: _will_not_get_removed(self, sym)

### Function: _args_as_int(self, args)

### Function: expansion(self, args)

### Function: expansions(self, args)

### Function: __init__(self)

### Function: _name_rule(self, inner)

### Function: _add_rule(self, key, name, expansions)

### Function: _add_recurse_rule(self, type_, expr)

### Function: _add_repeat_rule(self, a, b, target, atom)

**Description:** Generate a rule that repeats target ``a`` times, and repeats atom ``b`` times.

When called recursively (into target), it repeats atom for x(n) times, where:
    x(0) = 1
    x(n) = a(n) * x(n-1) + b

Example rule when a=3, b=4:

    new_rule: target target target atom atom atom atom

### Function: _add_repeat_opt_rule(self, a, b, target, target_opt, atom)

**Description:** Creates a rule that matches atom 0 to (a*n+b)-1 times.

When target matches n times atom, and target_opt 0 to n-1 times target_opt,

First we generate target * i followed by target_opt, for i from 0 to a-1
These match 0 to n*a - 1 times atom

Then we generate target * a followed by atom * i, for i from 0 to b-1
These match n*a to n*a + b-1 times atom

The created rule will not have any shift/reduce conflicts so that it can be used with lalr

Example rule when a=3, b=4:

    new_rule: target_opt
            | target target_opt
            | target target target_opt

            | target target target
            | target target target atom
            | target target target atom atom
            | target target target atom atom atom

### Function: _generate_repeats(self, rule, mn, mx)

**Description:** Generates a rule tree that repeats ``rule`` exactly between ``mn`` to ``mx`` times.
        

### Function: expr(self, rule, op)

### Function: maybe(self, rule)

### Function: _flatten(tree)

### Function: expansion(self, tree)

### Function: alias(self, tree)

### Function: expansions(self, tree)

### Function: expansions(self, x)

### Function: expansion(self, symbols)

### Function: alias(self, x)

### Function: __init__(self, terminals)

### Function: pattern(self, p)

### Function: __init__(self)

### Function: value(self, c)

### Function: template_usage(self, c)

### Function: __init__(self, rule_defs)

### Function: template_usage(self, c)

### Function: literal(self, literal)

### Function: range(self, start, end)

### Function: pattern(self, ps)

### Function: expansion(self, items)

### Function: expansions(self, exps)

### Function: expr(self, args)

### Function: maybe(self, expr)

### Function: alias(self, t)

### Function: value(self, v)

### Function: value(self, v)

### Function: __init__(self, rule_defs, term_defs, ignore)

### Function: compile(self, start, terminals_to_keep)

### Function: __init__(self, pkg_name, search_paths)

### Function: __repr__(self)

### Function: __call__(self, base_path, grammar_path)

### Function: terminal(self, name)

### Function: nonterminal(self, name)

### Function: expand(node)

### Function: on_error(e)

### Function: mangle(s)

### Function: __init__(self, is_term, tree, params, options)

### Function: __init__(self, global_keep_all_tokens, import_paths, used_files)

### Function: _grammar_error(self, is_term, msg)

### Function: _check_options(self, is_term, options)

### Function: _define(self, name, is_term, exp, params, options)

### Function: _extend(self, name, is_term, exp, params, options)

### Function: _ignore(self, exp_or_name)

### Function: _unpack_import(self, stmt, grammar_name)

### Function: _unpack_definition(self, tree, mangle)

### Function: load_grammar(self, grammar_text, grammar_name, mangle)

### Function: _remove_unused(self, used)

### Function: do_import(self, dotted_path, base_path, aliases, base_mangle)

### Function: validate(self)

### Function: build(self)

### Function: rule_dependencies(symbol)
