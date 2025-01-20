#!/usr/bin/env python3

import sys
from pathlib import Path

from tree_sitter_languages import get_language, get_parser

from .deepseek_v3_tokenizer import tokenizer


MAX_DEEPSEEK_TOKENS = 62000 - 8000 # output 8k counts towards 64k limit. Headroom for scaffolding.

QUERIES = {
    '.java': Path(__file__).parent / "queries" / "java" / "skeleton.scm",
    '.py': Path(__file__).parent / "queries" / "python" / "skeleton.scm"
}

def token_count(text: str) -> int:
    return len(tokenizer.encode(text))

def maybe_truncate(text, max_tokens, what):
    """Truncate 'text' to 'max_tokens' tokens if needed and log to stderr."""
    encoded = tokenizer.encode(text)
    if len(encoded) <= max_tokens:
        return text
    print(f"[WARN] {what} exceeds {max_tokens} tokens; truncating.", file=sys.stderr)
    return tokenizer.decode(encoded[:max_tokens])

def get_query(file_path: str) -> str:
    """Load the correct .scm query based on extension."""
    ext = Path(file_path).suffix
    if ext not in QUERIES:
        raise ValueError(f"Unsupported file extension: {ext}")
    return QUERIES[ext].read_text()

def parse_code(source_file: str):
    """
    Parse 'source_file' with Tree-sitter, run the appropriate query,
    and build IR (list of {type, start, end, text, node}).
    """
    code_str = Path(source_file).read_text()
    code_bytes = code_str.encode("utf8")

    ext = Path(source_file).suffix.lower()
    if ext not in ('.java', '.py'):
        raise ValueError(f"Unsupported extension: {ext}")
    lang_name = 'java' if ext == '.java' else 'python'
    parser = get_parser(lang_name)
    language = get_language(lang_name)
    tree = parser.parse(code_bytes)

    captures = language.query(get_query(source_file)).captures(tree.root_node)
    ir = []
    for node, capture_name in captures:
        # Skip annotation nodes in IR
        if capture_name == 'annotation':
            continue
        snippet = code_bytes[node.start_byte: node.end_byte].decode("utf8")
        ir.append({
            'type': capture_name,
            'start': node.start_byte,
            'end': node.end_byte,
            'text': snippet,
            'node': node,
        })
    ir.sort(key=lambda x: x['start'])
    return code_str, code_bytes, tree, ir

def compute_indentation(node, code_bytes):
    """Compute leading spaces for 'node' based on the nearest preceding newline."""
    start_byte = node.start_byte
    newline_pos = code_bytes.rfind(b'\n', 0, start_byte)
    line_start = 0 if newline_pos < 0 else newline_pos + 1
    return " " * (start_byte - line_start)

def leading_whitespace_of_snippet(text):
    """Return the leading whitespace of 'text'."""
    idx = 0
    while idx < len(text) and text[idx] in (' ', '\t'):
        idx += 1
    return text[:idx]

def gather_head(ir, root_node, code_bytes):
    """
    Return (head_items, body_items, top_level_class_count).
    Head items are top-level class signatures (+ '{') plus any top-level fields.
    Body items are everything else. Also track how many top-level classes have bodies.
    """
    head, body = [], []
    top_level_class_count = 0

    for item in ir:
        node, snippet = item['node'], item['text']
        # Find the containing top-level class
        p, top_level_class = node, None
        while p and p != root_node:
            if p.type in ('class_declaration','interface_declaration','annotation_declaration','enum_declaration'):
                if p.parent == root_node:
                    top_level_class = p
                break
            p = p.parent

        if top_level_class and node == top_level_class and item['type'] == 'class.declaration':
            body_node = node.child_by_field_name('body')
            indent = leading_whitespace_of_snippet(snippet)
            if body_node:
                top_level_class_count += 1
                sig_len = body_node.start_byte - node.start_byte
                partial = snippet[:sig_len].rstrip()
                head_text = partial + " {"
                if not head_text.startswith(indent):
                    head_text = indent + head_text.lstrip()
                head.append({**item, 'text': head_text})
            else:
                head.append(item)
        elif top_level_class and item['type'] == 'field.declaration':
            # Add one level of indentation for fields inside classes
            field_text = item['text']
            indent = leading_whitespace_of_snippet(field_text)
            field_text = indent + "    " + field_text.lstrip()
            head.append({**item, 'text': field_text})
        else:
            body.append(item)
    return head, body, top_level_class_count

def build_body_blocks(body_ir, code_str, root_node):
    """Group IR items so that nested classes remain intact and top-level items are not split."""
    code_bytes = code_str.encode('utf8')
    used, blocks = set(), []

    def fully_in(a, b):
        return a.start_byte >= b.start_byte and a.end_byte <= b.end_byte

    for item in body_ir:
        if (item['start'], item['end']) in used:
            continue
        node = item['node']
        if node.type in ('class_declaration', 'interface_declaration','annotation_declaration','enum_declaration'):
            snippet = code_bytes[node.start_byte: node.end_byte].decode('utf8')
            blocks.append({'start': node.start_byte,'end': node.end_byte,'text': snippet})
            for sub in body_ir:
                if fully_in(sub['node'], node):
                    used.add((sub['start'], sub['end']))
        else:
            blocks.append({'start': item['start'],'end': item['end'],'text': item['text']})
            used.add((item['start'], item['end']))
    return sorted(blocks, key=lambda b: b['start'])

def chunk_from_ir_with_head(ir, root_node, code_str, max_tokens=65536):
    """
    Build code chunks under 'max_tokens'. The 'head' is repeated in each chunk
    if it fits. Each nested class or method is kept intact.
    """
    code_bytes = code_str.encode("utf8")
    head_items, body_items, top_level_count = gather_head(ir, root_node, code_bytes)
    head_block = "\n".join(i['text'].rstrip('\r\n') for i in head_items).rstrip()
    head_tokens = token_count(head_block) if head_block else 0
    head_usable = head_block and (head_tokens <= (max_tokens // 2))
    body_budget = max_tokens - head_tokens if head_usable else max_tokens

    blocks = build_body_blocks(body_items, code_str, root_node)
    chunks, current_texts, current_tokens = [], [], 0

    def flush():
        if not current_texts and not head_usable:
            return
        chunk_body = "\n\n".join(current_texts).rstrip()
        if head_usable:
            chunk = head_block + ("\n\n" + chunk_body if chunk_body else "")
            if top_level_count > 0:
                chunk += "\n" + "\n".join("}" for _ in range(top_level_count))
        else:
            chunk = chunk_body
        chunks.append(chunk)

    for b in blocks:
        snippet = b['text']
        tcount = token_count(snippet)
        if tcount > body_budget:
            snippet = maybe_truncate(snippet, body_budget, "Large IR block")
            tcount = token_count(snippet)
        if current_tokens + tcount > body_budget:
            flush()
            current_texts, current_tokens = [snippet], tcount
        else:
            current_texts.append(snippet)
            current_tokens += tcount
    if current_texts:
        flush()
    return chunks

def extract_skeleton(source_file: str) -> str:
    """
    Return a concise structural outline of the code: classes, methods, fields,
    with indentation and { ... } placeholders.
    """
    code_str, code_bytes, tree, ir = parse_code(source_file)
    lines, open_braces = [], []

    def text_slice(s, e):
        return code_bytes[s:e].decode('utf8')

    for item in ir:
        ctype, node = item['type'], item['node']
        indent = compute_indentation(node, code_bytes)
        if ctype in ('class.declaration','interface.declaration','annotation.declaration','enum.declaration'):
            body = node.child_by_field_name('body')
            if body:
                sig_part = text_slice(node.start_byte, body.start_byte).rstrip()
                lines.append(f"{indent}{sig_part} {{")
                open_braces.append(indent)
            else:
                snippet = text_slice(node.start_byte, node.end_byte).rstrip()
                lines.append(f"{indent}{snippet}")
        elif ctype == 'method.declaration':
            body = node.child_by_field_name('body')
            ret_node = node.child_by_field_name('type')
            start_pos = ret_node.start_byte if ret_node else node.start_byte
            if body:
                sig_head = text_slice(start_pos, body.start_byte).rstrip()
                lines.append(f"{indent}{sig_head} {{...}}")
            else:
                snippet = text_slice(start_pos, node.end_byte).rstrip()
                lines.append(f"{indent}{snippet}")
        elif ctype == 'field.declaration':
            snippet = text_slice(node.start_byte, node.end_byte).rstrip()
            lines.append(f"{indent}{snippet}")

    while open_braces:
        lines.append(f"{open_braces.pop()}}}")
    return "\n".join(lines)

def chunk(source_file: str, max_tokens=MAX_DEEPSEEK_TOKENS):
    """
    Break the file's code into chunks that do not exceed 'max_tokens',
    preserving the top-level head block and grouping items sensibly.
    """
    if source_file.endswith('.py'):
        # chunking not yet supported for Python
        truncated = maybe_truncate(Path(source_file).read_text(), max_tokens, source_file)
        return [truncated]
    code_str, code_bytes, tree, ir = parse_code(source_file)
    return chunk_from_ir_with_head(ir, tree.root_node, code_str, max_tokens)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: parse.py <skeleton|chunk> <source_file> [source_file...]")
        sys.exit(1)

    cmd = sys.argv[1]
    fnames = sys.argv[2:]

    for fname in fnames:
        print(f"\n# {fname}\n")
        if cmd == 'skeleton':
            print("Skeleton:")
            print("--------------------------------------")
            print(extract_skeleton(fname))
        elif cmd == 'chunk':
            chs = chunk(fname)  # smaller max for demo
            print("Chunks:")
            print("--------------------------------------")
            for i, ch in enumerate(chs, 1):
                print(f"\n--- Chunk {i} (length={token_count(ch)})---")
                print(ch + "\n")
        else:
            print("First argument must be either 'skeleton' or 'chunk'")
            sys.exit(1)
