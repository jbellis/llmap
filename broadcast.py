import argparse
import glob
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ai import AI
from exceptions import AIException


def main():
    parser = argparse.ArgumentParser(description='Check Java files for relevance to a question')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--directory', help='Directory containing Java files')
    group.add_argument('--file', help='Single Java file to check')
    parser.add_argument('question', help='Question to check relevance against')
    parser.add_argument('--sample', type=int, help='Number of random files to sample')
    args = parser.parse_args()
    
    # Initialize client
    client = AI()
    
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
    
    def process_batch(executor, files, process_fn, desc):
        """Process a batch of files and return results, tracking errors"""
        futures = [executor.submit(process_fn, f) for f in files]
        results = []
        errors = []
        
        for future in tqdm(futures, desc=desc):
            try:
                results.append(future.result())
            except AIException as e:
                errors.append(e)
        return results, errors

    # Create thread pool and process files
    errors = []
    relevant_files = []
    with ThreadPoolExecutor(max_workers=500) as executor:
        # Phase 1: Generate initial relevance
        gen_fn = lambda f: client.generate_relevance(f, args.question)
        initial_results, phase1_errors = process_batch(
            executor, java_files, gen_fn, "Generating relevance")
        errors.extend(phase1_errors)
        
        # Phase 2: Evaluate initial results
        eval_fn = lambda r: client.evaluate_relevance(r[0], r[1], args.question, True)
        eval_results, phase2_errors = process_batch(
            executor, initial_results, eval_fn, "Evaluating relevance")
        errors.extend(phase2_errors)
        
        # Sort results into relevant files and those needing full source
        needs_full_source = []
        for file_path, verdict in eval_results:
            if verdict == "relevant":
                relevant_files.append(file_path)
            elif verdict == "source":
                needs_full_source.append(file_path)

        # Phase 3: Process files needing full source check
        if needs_full_source:
            # Generate full source relevance
            gen_full_fn = lambda f: client.generate_relevance_full_source(f, args.question)
            full_results, phase3_errors = process_batch(
                executor, needs_full_source, gen_full_fn, "Checking full source")
            errors.extend(phase3_errors)
            
            # Evaluate full source results
            eval_full_fn = lambda r: client.evaluate_relevance(r[0], r[1], args.question, False)
            full_eval_results, phase4_errors = process_batch(
                executor, full_results, eval_full_fn, "Evaluating full source")
            errors.extend(phase4_errors)
            
            # Add relevant files from full source check
            for file_path, verdict in full_eval_results:
                if verdict == "relevant":
                    relevant_files.append(file_path)

    # Print any errors to stderr
    if errors:
        print("\nErrors encountered:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        print("", file=sys.stderr)

    # No extra headers so the output can easily be used in xargs et al
    for file in relevant_files:
        print(file)

if __name__ == "__main__":
    main()
