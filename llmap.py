#!/usr/bin/env python3

import sys
from pathlib import Path
from tree_sitter_languages import get_language, get_parser

def _process_class(child, query, indent):
    """Process a class definition node"""
    parts = []
    class_body = None
    for n, t in query.captures(child):
        if t == 'name.definition.class':
            parts.append('class ' + n.text.decode('utf8').strip())
            continue
        if t in ('class.modifiers', 'class.superclass', 'class.interfaces'):
            if n.text:
                parts.append(t + '->' + n.text.decode('utf8').strip())
            continue
        if t == 'class.body':
            class_body = n
    print(parts)
    return ' '.join(parts), class_body

def _process_interface(child, query, indent):
    """Process an interface definition node"""
    parts = []
    interface_body = None
    for n, t in query.captures(child):
        if t in ('interface.modifiers', 'name.definition.interface', 'interface.extends'):
            if n.text:
                parts.append(n.text.decode('utf8').strip())
        if t == 'interface.body':
            interface_body = n
    
    return ' '.join(parts), interface_body

def _process_method(child, query, indent):
    """Process a method definition node"""
    parts = []
    for n, t in query.captures(child):
        if t in ('method.modifiers', 'method.type', 'name.definition.method', 'method.params'):
            if n.text:
                parts.append(n.text.decode('utf8').strip())
    return ' '.join(parts) + ' {...}'

def _process_field(child, indent):
    """Process a field definition node"""
    return child.text.decode('utf8')

def _process_node(node, query, indent_level=0):
    """Process a node recursively and return its skeleton parts"""
    skeleton = []
    indent = "  " * indent_level

    for child, tag in query.captures(node):
        if tag == 'definition.class':
            signature, class_body = _process_class(child, query, indent)
            skeleton.append(indent + signature)
            if class_body:
                skeleton.extend(_process_node(class_body, query, indent_level + 1))
            
        elif tag == 'definition.interface':
            signature, interface_body = _process_interface(child, query, indent)
            skeleton.append(indent + signature)
            skeleton.extend(_process_node(interface_body, query, indent_level + 1))
            
        elif tag == 'definition.method':
            signature = _process_method(child, query, indent)
            skeleton.append(indent + signature)
            
        elif tag == 'definition.field':
            text = _process_field(child, indent)
            skeleton.append(indent + text)

    return skeleton

def extract_skeleton(java_file):
    """Extract class/method signatures from a Java file"""
    # Load Java parser
    language = get_language('java')
    parser = get_parser('java')
    
    # Load tags query
    query_path = Path(__file__).parent / "languages" / "tree-sitter-java-tags.scm" 
    query = language.query(query_path.read_text())

    # Parse file
    code = Path(java_file).read_text()
    tree = parser.parse(bytes(code, "utf8"))

    # Extract definitions recursively
    skeleton = _process_node(tree.root_node, query)
    return '\n\n'.join(skeleton)

def main():
    if len(sys.argv) < 2:
        print("Usage: llmap.py <java_file> [java_file...]")
        sys.exit(1)
        
    for fname in sys.argv[1:]:
        print(f"\n# {fname}\n")
        print(extract_skeleton(fname))

if __name__ == '__main__':
    main()
