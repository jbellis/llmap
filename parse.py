#!/usr/bin/env python3

import sys
from pathlib import Path
from tree_sitter import Query
from tree_sitter_languages import get_language, get_parser

# Load queries
QUERIES = {
    '.java': Path(__file__).parent / "queries" / "java" / "skeleton.scm",
    '.py': Path(__file__).parent / "queries" / "python" / "skeleton.scm"
}

def get_query(file_path: str):
    """Get the appropriate query for the file extension"""
    ext = Path(file_path).suffix
    if ext not in QUERIES:
        raise ValueError(f"Unsupported file extension: {ext}")
    return QUERIES[ext].read_text()

def _format_node(node, indent_level=0):
    """Format a node's text with proper indentation"""
    return "  " * indent_level + node.text.decode('utf8')

def _process_captures(captures, code_bytes, file_ext):
    """Process query captures into skeleton structure"""
    skeleton = []
    
    for capture in captures:
        node = capture[0]
        capture_name = capture[1]
        
        # Skip Java annotations
        if capture_name == 'annotation':
            continue
            
        # Calculate indent based on node's start position
        start_byte = node.start_byte
        line_start = code_bytes.rfind(b'\n', 0, start_byte) + 1
        indent = "  " * ((start_byte - line_start) // 2)
        
        if capture_name in ('class.declaration', 'interface.declaration', 'annotation.declaration'):
            # Get text up to body start, skipping annotations
            body_node = node.child_by_field_name('body')
            start_pos = node.start_byte
            # Skip past any annotations
            for child in node.children:
                if child.type == 'annotation':
                    start_pos = child.end_byte
                elif child.type in ('class_declaration', 'interface_declaration'):
                    break
            text = node.text[start_pos - node.start_byte:body_node.start_byte - node.start_byte].decode('utf8')
            skeleton.append(f"{indent}{text.strip()} {{")
            
        elif capture_name == 'method.declaration':
            # Get text up to body (if it exists)
            body_node = node.child_by_field_name('body')
            # Find return type node to start from
            return_type = node.child_by_field_name('type')
            start_pos = return_type.start_byte if return_type else node.start_byte
            if body_node:
                text = node.text[start_pos - node.start_byte:body_node.start_byte - node.start_byte].decode('utf8')
                skeleton.append(f"{indent}{text.strip()} {{...}}")
            else:
                text = node.text[start_pos - node.start_byte:].decode('utf8')
                skeleton.append(f"{indent}{text.strip()}")
                
        elif capture_name == 'field.declaration':
            # skip annotations (only found in Java)
            start_pos = node.start_byte
            for child in node.children:
                if child.type == 'annotation':
                    start_pos = child.end_byte
                elif child.type == 'field_declaration':
                    break
            text = node.text[start_pos - node.start_byte:].decode('utf8')
            skeleton.append(f"{indent}{text.strip()}")
            
        elif capture_name == 'enum.declaration':
            # Get text up to body, skipping annotations
            body_node = node.child_by_field_name('body')
            start_pos = node.start_byte
            # Skip past any annotations
            for child in node.children:
                if child.type == 'annotation':
                    start_pos = child.end_byte
                elif child.type == 'enum_declaration':
                    break
            text = node.text[start_pos - node.start_byte:body_node.start_byte - node.start_byte].decode('utf8')
            skeleton.append(f"{indent}{text.strip()} {{")
            
            # Add constants if present
            constants_node = node.child_by_field_name('constants')
            if constants_node:
                skeleton.append(f"{indent}  {constants_node.text.decode('utf8')}")
    
    # Add closing braces for containers
    result = []
    open_braces = []
    
    for line in skeleton:
        if line.rstrip().endswith(" {"):
            # Store the actual indentation string by finding leading spaces
            indent = line[:len(line) - len(line.lstrip())]
            open_braces.append(indent)
            result.append(line)
        else:
            if open_braces and not line.strip():
                continue  # Skip empty lines between braces
            result.append(line)
            
    # Add remaining closing braces
    while open_braces:
        indent = open_braces.pop()
        result.append("")  # Add blank line before closing brace
        result.append(f"{indent}}}")
    
    return '\n'.join(result)

def extract_skeleton(source_file):
    """Extract class/method signatures from source file"""
    # Get file extension and load appropriate parser
    file_ext = Path(source_file).suffix.lower()
    lang_name = 'java' if file_ext == '.java' else 'python'
    
    parser = get_parser(lang_name)
    language = get_language(lang_name)
    query = language.query(get_query(source_file))
    
    # Parse file
    code = Path(source_file).read_text()
    code_bytes = bytes(code, "utf8")
    tree = parser.parse(code_bytes)
    
    # Execute query and process captures
    captures = query.captures(tree.root_node)
    skeleton = _process_captures(captures, code_bytes, file_ext)
    
    return skeleton

def main():
    if len(sys.argv) < 2:
        print("Usage: parse.py <source_file> [source_file...]")
        sys.exit(1)
        
    for fname in sys.argv[1:]:
        print(f"\n# {fname}\n")
        print(extract_skeleton(fname))

if __name__ == '__main__':
    main()
