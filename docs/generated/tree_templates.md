## AI Summary

A file named tree_templates.py.


## Class: TemplateConf

**Description:** Template Configuration

Allows customization for different uses of Template

parse() must return a Tree instance.

## Class: _ReplaceVars

## Class: Template

**Description:** Represents a tree template, tied to a specific configuration

A tree template is a tree that contains nodes that are template variables.
Those variables will match any tree.
(future versions may support annotations on the variables, to allow more complex templates)

### Function: translate(t1, t2, tree)

**Description:** Search tree and translate each occurrence of t1 into t2.
    

## Class: TemplateTranslator

**Description:** Utility class for translating a collection of patterns
    

### Function: _get_template_name(value)

### Function: __init__(self, parse)

### Function: test_var(self, var)

**Description:** Given a tree node, if it is a template variable return its name. Otherwise, return None.

This method may be overridden for customization

Parameters:
    var: Tree | str - The tree node to test

### Function: _get_tree(self, template)

### Function: __call__(self, template)

### Function: _match_tree_template(self, template, tree)

**Description:** Returns dict of {var: match} if found a match, else None
        

### Function: __init__(self, conf, vars)

### Function: __default__(self, data, children, meta)

### Function: __init__(self, tree, conf)

### Function: match(self, tree)

**Description:** Match a tree template to a tree.

A tree template without variables will only match ``tree`` if it is equal to the template.

Parameters:
    tree (Tree): The tree to match to the template

Returns:
    Optional[Dict[str, Tree]]: If match is found, returns a dictionary mapping
        template variable names to their matching tree nodes.
        If no match was found, returns None.

### Function: search(self, tree)

**Description:** Search for all occurrences of the tree template inside ``tree``.
        

### Function: apply_vars(self, vars)

**Description:** Apply vars to the template tree
        

### Function: __init__(self, translations)

### Function: translate(self, tree)
