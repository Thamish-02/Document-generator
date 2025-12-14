## AI Summary

A file named parser_frontends.py.


### Function: _wrap_lexer(lexer_class)

### Function: _deserialize_parsing_frontend(data, memo, lexer_conf, callbacks, options)

## Class: ParsingFrontend

### Function: _validate_frontend_args(parser, lexer)

### Function: _get_lexer_callbacks(transformer, terminals)

## Class: PostLexConnector

### Function: create_basic_lexer(lexer_conf, parser, postlex, options)

### Function: create_contextual_lexer(lexer_conf, parser, postlex, options)

### Function: create_lalr_parser(lexer_conf, parser_conf, options)

## Class: EarleyRegexpMatcher

### Function: create_earley_parser__dynamic(lexer_conf, parser_conf)

### Function: _match_earley_basic(term, token)

### Function: create_earley_parser__basic(lexer_conf, parser_conf)

### Function: create_earley_parser(lexer_conf, parser_conf, options)

## Class: CYK_FrontEnd

### Function: _construct_parsing_frontend(parser_type, lexer_type, lexer_conf, parser_conf, options)

### Function: __init__(self, lexer_conf, parser_conf, options, parser)

### Function: _verify_start(self, start)

### Function: _make_lexer_thread(self, text)

### Function: parse(self, text, start, on_error)

### Function: parse_interactive(self, text, start)

### Function: __init__(self, lexer, postlexer)

### Function: lex(self, lexer_state, parser_state)

### Function: __init__(self, lexer_conf)

### Function: match(self, term, text, index)

### Function: __init__(self, lexer_conf, parser_conf, options)

### Function: parse(self, lexer_thread, start)

### Function: _transform(self, tree)

### Function: _apply_callback(self, tree)

## Class: CustomLexerWrapper1

### Function: __init__(self, lexer_conf)

### Function: lex(self, lexer_state, parser_state)

## Class: CustomLexerWrapper0

### Function: __init__(self, lexer_conf)

### Function: lex(self, lexer_state, parser_state)
