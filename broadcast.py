import argparse
import glob
import os
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ai import generate_relevance, create_client, evaluate_relevance


def main():
    parser = argparse.ArgumentParser(description='Check Java files for relevance to a question')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--directory', help='Directory containing Java files')
    group.add_argument('--file', help='Single Java file to check')
    parser.add_argument('question', help='Question to check relevance against')
    parser.add_argument('--sample', type=int, help='Number of random files to sample')
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        return 1
        
    # Initialize client
    client = create_client(api_key)
    
    # Get Java files based on input
    if args.file:
        if not args.file.endswith('.java'):
            print("Error: File must be a Java file")
            return 1
        if not os.path.isfile(args.file):
            print(f"Error: File {args.file} does not exist")
            return 1
        java_files = [args.file]
    else:
        # Get all Java files in the directory
        java_files = glob.glob(os.path.join(args.directory, "**/*.java"), recursive=True)
        
        # Sample files if requested
        if args.sample and args.sample < len(java_files):
            java_files = random.sample(java_files, args.sample)
    
    # Create thread pool and process files
    with ThreadPoolExecutor(max_workers=500) as executor:
        futures = []
        for file_path in java_files:
            future = executor.submit(generate_relevance, file_path, args.question, client)
            futures.append(future)
        
        # Process results with progress bars
        results = []
        relevant_files = []
        
        # First progress bar for generating relevance
        for future in tqdm(futures, total=len(futures), desc="Generating relevance"):
            results.append(future.result())
            
        # Second progress bar for evaluating relevance
        for file, result in tqdm(results, desc="Evaluating relevance"):
            _, is_relevant = evaluate_relevance(client, file, result)
            if is_relevant:
                relevant_files.append(file)
    
    for file in relevant_files:
        print(file)

if __name__ == "__main__":
    main()
