import json
import datetime
from openai import OpenAI, BadRequestError
from deepseek_v2_tokenizer import tokenizer
from llmap import extract_skeleton

MAX_TOKENS = 64000  # Leave some headroom for message scaffolding while staying under 64k token limit

# TODO split up large files into declaration + state + methods and run multiple evaluations
# against different sets of methods for very large files instead of throwing data away
def maybe_truncate(text: str, max_tokens: int) -> str:
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

def check_full_source(file_path: str, question: str, client: OpenAI) -> tuple[str, str]:
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
         "content": f"Given this source code:\n\n{source}\n\nIs this code relevant to the problem or question: {question}? Answer with only yes or no"}
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

def generate_relevance(file_path: str, question: str, client: OpenAI) -> tuple[str, str]:
    """Check if a Java file is relevant to the question using DeepSeek."""
    skeleton = extract_skeleton(file_path)
    
    # Truncate if needed
    skeleton = maybe_truncate(skeleton, MAX_TOKENS)
    
    # Create messages 
    messages = [
        {"role": "system", "content": "You are a helpful assistant designed to analyze and explain source code."},
        {"role": "user",
         "content": f"Given this code skeleton:\n\n{skeleton}\n\nIs this code relevant to the following problem or question: {question}? Give your reasoning, then a final verdict of Relevant, Irrelevant, or Unclear."}
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
    
    # Log all evaluations in JSONL format
    eval_data = {
        "file": file_path,
        "response": answer,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open('evaluation.jsonl', 'a') as f:
        f.write(json.dumps(eval_data) + '\n')
        
    # Also log unclear cases separately in JSONL
    if 'unclear' in answer:
        with open('unclear.jsonl', 'a') as f:
            f.write(json.dumps(eval_data) + '\n')
            
    return file_path, answer

def create_client(api_key: str) -> OpenAI:
    """Create OpenAI client configured for DeepSeek"""
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def evaluate_relevance(client, full_path: str, evaluation_text: str) -> tuple[str, bool]:
    """Convert LLM's evaluation text into a boolean relevance decision"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that responds with only a single word without elaboration, not even punctuation."},
        {"role": "user", "content": f"Based on this evaluation:\n\n{evaluation_text}\n\nIs the related file relevant? Answer with only Relevant or Irrelevant"}
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    verdict = response.choices[0].message.content.lower().strip()
    is_relevant = verdict == "relevant"
    return full_path, is_relevant
