#!/usr/bin/env python3

import sys
from pathlib import Path
from tree_sitter_languages import get_language, get_parser

def _process_node(node, query, indent_level=0):
    """Process a node recursively and return its skeleton parts"""
    skeleton = []
    indent = "  " * indent_level

    for child, tag in query.captures(node):
        if tag == 'definition.class':
            # Find all parts of the class declaration
            parts = []
            class_body = None
            for n, t in query.captures(child):
                if t == 'name.definition.class':
                    parts.append('class ' + n.text.decode('utf8').strip())
                    continue
                if t in ('class.modifiers', 'class.superclass', 'class.interfaces'):
                    if n.text:
                        parts.append(n.text.decode('utf8').strip())
                if t == 'class.body':
                    class_body = n
            
            skeleton.append(indent + ' '.join(parts))
            # Recursively process the class body
            if class_body:
                skeleton.extend(_process_node(class_body, query, indent_level + 1))
            
        elif tag == 'definition.interface':
            # Find all parts of the interface declaration
            parts = []
            class_body = None
            for n, t in query.captures(child):
                if t in ('interface.modifiers', 'name.definition.interface', 'interface.extends'):
                    if n.text:
                        parts.append(n.text.decode('utf8').strip())
                if t == 'interface.body':
                    class_body = n

            skeleton.append(indent + ' '.join(parts))
            # Recursively process the interface body
            skeleton.extend(_process_node(class_body, query, indent_level + 1))
            
        elif tag == 'definition.method':
            # Find all parts of the method declaration
            parts = []
            for n, t in query.captures(child):
                if t in ('method.modifiers', 'method.type', 'name.definition.method', 'method.params'):
                    if n.text:
                        parts.append(n.text.decode('utf8').strip())
            skeleton.append(indent + ' '.join(parts) + ' {...}')
            
        elif tag == 'definition.field':
            text = child.text.decode('utf8')
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
