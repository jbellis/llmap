#!/usr/bin/env python3

import sys
from pathlib import Path
from tree_sitter_languages import get_language, get_parser

def _process_class(node, indent):
    """Process a class definition node"""
    name_node = node.child_by_field_name('name')
    class_name = name_node.text.decode('utf8') if name_node else 'Anonymous'

    modifiers = []
    for t in ('class.modifiers', 'class.superclass', 'class.interfaces'):
        child = node.child_by_field_name('t')
        if not child:
            continue
        modifiers.extend(m.text.decode('utf8') for m in child)

    body = node.child_by_field_name('body')
    return f"{' '.join(modifiers)}class {class_name}", body

def _process_interface(node, indent):
    """Process an interface definition node"""
    name_node = node.child_by_field_name('name')
    interface_name = name_node.text.decode('utf8') if name_node else 'Anonymous'
    
    modifiers = node.child_by_field_name('modifiers')
    mod_text = modifiers.text.decode('utf8') + ' ' if modifiers else ''
    
    body = node.child_by_field_name('body')
    return f"{mod_text}interface {interface_name}", body

def _process_method(node, indent):
    """Process a method definition node"""
    name = node.child_by_field_name('name').text.decode('utf8')
    type_node = node.child_by_field_name('type')
    type_text = type_node.text.decode('utf8') if type_node else 'void'
    params = node.child_by_field_name('parameters').text.decode('utf8')
    
    modifiers = node.child_by_field_name('modifiers')
    mod_text = modifiers.text.decode('utf8') + ' ' if modifiers else ''
    
    return f"{mod_text}{type_text} {name}{params} {{...}}"

def _process_field(node, indent):
    """Process a field definition node"""
    return node.text.decode('utf8')

def _process_enum(node, indent):
    """Process an enum definition node"""
    name_node = node.child_by_field_name('name')
    enum_name = name_node.text.decode('utf8') if name_node else 'Anonymous'
    
    modifiers = node.child_by_field_name('modifiers')
    mod_text = modifiers.text.decode('utf8') + ' ' if modifiers else ''
    
    body = node.child_by_field_name('body')
    
    # Start with the enum declaration
    enum_parts = [f"{mod_text}enum {enum_name}"]
    
    if body:
        # First process enum constants
        constants = []
        methods = []
        
        for child in body.children:
            if child.type == 'enum_constant':
                # Just get the identifier part for enum constants
                const_name = child.child_by_field_name('name')
                if const_name:
                    constants.append(indent + "  " + const_name.text.decode('utf8'))
            elif child.type == 'method_declaration':
                # Add method signature
                method_sig = _process_method(child, indent)
                methods.append(indent + "  " + method_sig)
            elif child.type == 'enum_body_declarations':
                # Process methods inside the enum body declarations
                for method in child.children:
                    if method.type == 'method_declaration':
                        method_sig = _process_method(method, indent)
                        methods.append(indent + "  " + method_sig)
        
        # Add constants first, then methods
        enum_parts.extend(constants)
        if methods:
            enum_parts.extend(methods)
                
    return "\n".join(enum_parts), None

def _process_node(node, indent_level=0):
    """Process a node recursively and return its skeleton parts"""
    skeleton = []
    indent = "  " * indent_level

    for child in node.children:
        if child.type == 'class_declaration':
            signature, body = _process_class(child, indent)
            skeleton.append(indent + signature)
            if body:
                skeleton.extend(_process_node(body, indent_level + 1))
            
        elif child.type == 'interface_declaration':
            signature, body = _process_interface(child, indent)
            skeleton.append(indent + signature)
            if body:
                skeleton.extend(_process_node(body, indent_level + 1))
            
        elif child.type == 'method_declaration':
            signature = _process_method(child, indent)
            skeleton.append(indent + signature)
            
        elif child.type == 'field_declaration':
            text = _process_field(child, indent)
            skeleton.append(indent + text)
            
        elif child.type == 'enum_declaration':
            signature, body = _process_enum(child, indent)
            skeleton.append(indent + signature)
            if body:
                skeleton.extend(_process_node(body, indent_level + 1))

    return skeleton

def extract_skeleton(source_file):
    """Extract class/method signatures from a Java file"""
    # Load Java parser
    parser = get_parser('java')
    
    # Parse file
    code = Path(source_file).read_text()
    tree = parser.parse(bytes(code, "utf8"))

    # Extract definitions recursively
    skeleton = _process_node(tree.root_node)
    return '\n\n'.join(skeleton)

def main():
    if len(sys.argv) < 2:
        print("Usage: llmap.py <source_file> [source_file...]")
        sys.exit(1)
        
    for fname in sys.argv[1:]:
        print(f"\n# {fname}\n")
        print(extract_skeleton(fname))

if __name__ == '__main__':
    main()
