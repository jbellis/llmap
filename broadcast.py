import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI, BadRequestError
from llmap import extract_skeleton
from deepseek_v2_tokenizer.deepseek_v2_tokenizer import tokenizer

MAX_TOKENS = 64000  # Leave some headroom for message scaffolding while staying under 64k token limit

def truncate_skeleton(skeleton, max_tokens):
    """Truncate skeleton to stay under token limit"""
    # Count tokens
    tokens = tokenizer.encode(skeleton)
    if len(tokens) <= max_tokens:
        return skeleton
        
    # If over limit, truncate skeleton while preserving structure
    while len(tokens) > max_tokens:
        # Cut skeleton in half
        lines = skeleton.split('\n')
        skeleton = '\n'.join(lines[:len(lines)//2])
        tokens = tokenizer.encode(skeleton)
        
    return skeleton

def check_relevance(file_path, question, client):
    """Check if a Java file is relevant to the question using DeepSeek."""
    skeleton = extract_skeleton(file_path)
    
    # Truncate if needed
    skeleton = truncate_skeleton(skeleton, MAX_TOKENS)
    
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
    if answer == 'yes':
        return file_path
    return None


def main():
    parser = argparse.ArgumentParser(description='Check Java files for relevance to a question')
    parser.add_argument('directory', help='Directory containing Java files')
    parser.add_argument('question', help='Question to check relevance against')
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        return 1
        
    # Initialize OpenAI client with DeepSeek endpoint
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    # Get all Java files in the directory
    java_files = glob.glob(os.path.join(args.directory, "**/*.java"), recursive=True)
    
    # Create thread pool and process files
    with ThreadPoolExecutor(max_workers=500) as executor:
        futures = []
        for file_path in java_files:
            future = executor.submit(check_relevance, file_path, args.question, client)
            futures.append(future)
        
        # Process results with progress bar
        results = []
        for future in tqdm(futures, total=len(futures), desc="Processing files"):
            results.append(future.result())
    
    # Print results
    for file, result in results:
        print(f'{file}: {result}')

if __name__ == "__main__":
    main()
