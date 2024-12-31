import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from llmap import extract_skeleton
from ai import check_relevance, check_full_source, create_client


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
        
    # Initialize client
    client = create_client(api_key)
    
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
    
    # Separate files that need full source check
    source_files = [file for file, result in results if result == 'source']
    
    # Second pass for source files
    if source_files:
        source_futures = []
        with ThreadPoolExecutor(max_workers=500) as executor:
            for file_path in source_files:
                future = executor.submit(check_full_source, file_path, args.question, client)
                source_futures.append(future)
            
            # Process source results with progress bar
            source_results = []
            for future in tqdm(source_futures, total=len(source_futures), desc="Processing full sources"):
                file, r = future.result()
                source_results.append((file, 'source -> ' + r))
        
        # Update results for source files
        results = [(f, r) for f, r in results if r != 'source'] + source_results

    # Print final results
    for file, result in results:
        print(f'{file}: {result}')

if __name__ == "__main__":
    main()
