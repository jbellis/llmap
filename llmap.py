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
        if tag in ('definition.class', 'definition.interface'):
            text = node.text.decode('utf8')
            if '{' in text:
                text = text[:text.index('{')] + ' {'
            skeleton.append(text + '\n}')
        elif tag == 'definition.method':
            text = node.text.decode('utf8')
            if '{' in text:
                text = text[:text.index('{')] + ' {'
            skeleton.append('    ' + text + '\n    }')

    return '\n\n'.join(skeleton)

def main():
    if len(sys.argv) < 2:
        print("Usage: skeleton.py <java_file> [java_file...]")
        sys.exit(1)
        
    for fname in sys.argv[1:]:
        print(f"\n# {fname}\n")
        print("```java")
        print(extract_skeleton(fname))
        print("```")

if __name__ == '__main__':
    main()
