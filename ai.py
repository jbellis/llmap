from openai import OpenAI, BadRequestError
from deepseek_v2_tokenizer import tokenizer
from llmap import extract_skeleton

MAX_TOKENS = 64000  # Leave some headroom for message scaffolding while staying under 64k token limit

def maybe_truncate(text, max_tokens):
    """Truncate skeleton to stay under token limit"""
    # Count tokens
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
        
    # If over limit, truncate skeleton while preserving structure
    while len(tokens) > max_tokens:
        # Cut skeleton in half
        lines = text.split('\n')
        text = '\n'.join(lines[:len(lines) // 2])
        tokens = tokenizer.encode(text)
        
    return text

def check_full_source(file_path, question, client):
    """Check the full source of a file for relevance"""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
    except Exception as e:
        return file_path, f"Error reading file: {e}"

    # Truncate if needed
    source = maybe_truncate(source, MAX_TOKENS)  # Reuse truncate function
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that responds with only a single word without elaboration, not even punctuation."},
        {"role": "user",
         "content": f"Given this Java source code:\n\n{source}\n\nIs this code relevant to the question: {question}? Answer with only yes or no"}
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
    except BadRequestError as e:
        return file_path, e

    return file_path, response.choices[0].message.content.lower().strip()

def check_relevance(file_path, question, client):
    """Check if a Java file is relevant to the question using DeepSeek."""
    skeleton = extract_skeleton(file_path)
    
    # Truncate if needed
    skeleton = maybe_truncate(skeleton, MAX_TOKENS)
    
    # Create messages 
    messages = [
        {"role": "system", "content": "You are a helpful assistant that responds with only a single word without elaboration, not even punctuation."},
        {"role": "user",
         "content": f"Given this Java code skeleton:\n\n{skeleton}\n\nIs this code relevant to the question: {question}? Answer with yes or no, or source if you can't tell from the skeleton but the full source could clarify."}
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
    except BadRequestError as e:
        return file_path, e

    answer = response.choices[0].message.content.lower().strip()
    return file_path, answer

def create_client(api_key):
    """Create OpenAI client configured for DeepSeek"""
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
