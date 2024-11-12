import sys
import warnings
from pathlib import Path
from tree_sitter_languages import get_parser, get_language

# Suppress FutureWarning from tree_sitter
warnings.simplefilter("ignore", category=FutureWarning)

def get_skeleton(fname):
    """Extract class/method signatures from a source file using tree-sitter"""
    
    code = Path(fname).read_text()
    if not code.endswith("\n"):
        code += "\n"
        
    lang = Path(fname).suffix.lstrip('.')
    if lang == 'py':
        lang = 'python'
        
    try:
        language = get_language(lang)
        parser = get_parser(lang)
    except Exception as err:
        print(f"Error processing {fname}: {err}", file=sys.stderr)
        return ""

    tree = parser.parse(bytes(code, "utf-8"))
    
    def extract_signature(node):
        """Get just the signature portion of a node"""
        if node.type in ("class_declaration", "interface_declaration"):
            # For classes/interfaces, find end of declaration before body
            class_body = next((c for c in node.children if c.type.endswith('_body')), None)
            if class_body:
                end_byte = class_body.start_byte
            else:
                end_byte = node.end_byte
        else:
            # For methods, find end of declaration before body
            body = next((c for c in node.children if c.type == 'block'), None)
            if body:
                end_byte = body.start_byte
            else:
                end_byte = node.end_byte
                
        sig = code[node.start_byte:end_byte].strip()
        return sig

    def traverse(node, depth=0):
        """Walk the AST and collect signatures with proper indentation"""
        sigs = []
        
        # Check if this is a definition node we care about
        if node.type in ("class_declaration", "interface_declaration", 
                        "method_declaration", "constructor_declaration",
                        "function_declaration"):
            sig = extract_signature(node)
            if sig:
                indent = "    " * depth
                sigs.append(f"{indent}{sig}")
                depth += 1
                
        # Recurse into children
        for child in node.children:
            sigs.extend(traverse(child, depth))
            
        return sigs

    signatures = traverse(tree.root_node)
    return "\n".join(signatures)

def main():
    if len(sys.argv) < 2:
        print("Usage: python llmap.py <source_files...>", file=sys.stderr)
        sys.exit(1)
        
    for fname in sys.argv[1:]:
        if not Path(fname).is_file():
            print(f"Error: {fname} is not a file", file=sys.stderr)
            continue
            
        print(f"\n# {fname}\n")
        skeleton = get_skeleton(fname)
        if skeleton:
            print("```" + Path(fname).suffix.lstrip('.'))
            print(skeleton)
            print("```")

if __name__ == "__main__":
    main()
