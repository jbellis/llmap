#!/usr/bin/env python3

import sys
from pathlib import Path
from tree_sitter_languages import get_language, get_parser

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

    # Extract definitions 
    skeleton = []
    for node, tag in query.captures(tree.root_node):
        if tag == 'definition.class':
            # Find all parts of the class declaration
            parts = []
            for n, t in query.captures(node):
                if t in ('class.modifiers', 'name.definition.class', 'class.superclass', 'class.interfaces'):
                    if n.text:
                        parts.append(n.text.decode('utf8').strip())
            skeleton.append(' '.join(parts))
        elif tag == 'definition.interface':
            # Find all parts of the interface declaration
            parts = []
            for n, t in query.captures(node):
                if t in ('interface.modifiers', 'name.definition.interface', 'interface.extends'):
                    if n.text:
                        parts.append(n.text.decode('utf8').strip())
            skeleton.append(' '.join(parts))
        elif tag == 'definition.method':
            # Find all parts of the method declaration
            parts = []
            for n, t in query.captures(node):
                if t in ('method.modifiers', 'method.type', 'name.definition.method', 'method.params'):
                    if n.text:
                        parts.append(n.text.decode('utf8').strip())
            skeleton.append('  ' + ' '.join(parts) + ' {...}')
        elif tag == 'definition.field':
            text = node.text.decode('utf8')
            skeleton.append('  ' + text)

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
