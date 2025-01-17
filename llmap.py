import argparse
import glob
import hashlib
import json
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
    parser.add_argument('--save-cache', action='store_true', help='Keep cache directory after completion')
    args = parser.parse_args()
    
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

    # Setup cache directory
    cache_key = f"{args.question}_{args.directory or args.file}_{args.sample or 'all'}"
    # Hash it to get a safe directory name
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_dir = Path(".llmap_cache") / cache_hash
    if cache_dir.exists():
        print(f"Using cache directory: {cache_dir}")
    else:
        print(f"Creating cache directory: {cache_dir}")

    # Initialize client
    client = AI(cache_dir)

    def load_cached_results(cache_path, phase):
        """Load cached results and errors for a phase"""
        results_file = cache_path / f"{phase}_results.jsonl"
        errors_file = cache_path / f"{phase}_errors.jsonl"
    
        results = []
        errors = []
    
        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    results.append(tuple(json.loads(line)))
                
        if errors_file.exists():
            with open(errors_file) as f:
                for line in f:
                    error_data = json.loads(line)
                    errors.append(AIException(
                        message=error_data['message'],
                        filename=error_data['filename'],
                        original_exception=None
                    ))
                
        return results, errors

    def save_results(cache_path, phase, results, errors):
        """Save results and errors for a phase"""
        results_file = cache_path / f"{phase}_results.jsonl"
        errors_file = cache_path / f"{phase}_errors.jsonl"
    
        with open(results_file, 'w') as f:
            for result in results:
                json.dump(list(result), f)
                f.write('\n')
            
        with open(errors_file, 'w') as f:
            for error in errors:
                error_dict = {
                    'message': str(error),
                    'filename': error.filename
                }
                json.dump(error_dict, f)
                f.write('\n')

    def process_batch(executor, files, process_fn, desc, cache_path=None, phase=None):
        """Process a batch of files and return results, tracking errors"""
        # Check cache first if path provided
        if cache_path and phase:
            if (cache_path / f"{phase}_results.jsonl").exists():
                print(f"Using cached results for {desc}", file=sys.stderr)
                return load_cached_results(cache_path, phase)
        
        futures = [executor.submit(process_fn, f) for f in files]
        results = []
        errors = []
        
        for future in tqdm(futures, desc=desc):
            try:
                results.append(future.result())
            except AIException as e:
                errors.append(e)
                
        # Save results if cache path provided
        if cache_path and phase:
            cache_path.mkdir(parents=True, exist_ok=True)
            save_results(cache_path, phase, results, errors)
            
        return results, errors

    # Create thread pool and process files
    errors = []
    relevant_files = []
    with ThreadPoolExecutor(max_workers=500) as executor:
        # Phase 1: Generate initial relevance
        gen_fn = lambda f: client.generate_relevance(f, args.question)
        initial_results, phase1_errors = process_batch(
            executor, java_files, gen_fn, "Generating relevance",
            cache_path=cache_dir, phase="gen1")
        errors.extend(phase1_errors)
        
        # Phase 2: Evaluate initial results
        eval_fn = lambda r: client.evaluate_relevance(r[0], r[1], args.question, True)
        eval_results, phase2_errors = process_batch(
            executor, initial_results, eval_fn, "Evaluating relevance",
            cache_path=cache_dir, phase="eval1")
        errors.extend(phase2_errors)
        
        # Initialize results dictionary with initial analysis
        full_results_dict = {file_path: analysis for file_path, analysis in initial_results}

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
            full_results_list, phase3_errors = process_batch(
                executor, needs_full_source, gen_full_fn, "Checking full source",
                cache_path=cache_dir, phase="gen2")
            errors.extend(phase3_errors)
            
            # Update results dictionary with full source analysis
            full_results_dict.update({file_path: analysis for file_path, analysis in full_results_list})
            
            # Evaluate full source results
            eval_full_fn = lambda r: client.evaluate_relevance(r[0], r[1], args.question, False)
            full_eval_results, phase4_errors = process_batch(
                executor, full_results_list, eval_full_fn, "Evaluating full source",
                cache_path=cache_dir, phase="eval2")
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

    # Print relevant files with their analysis
    # TODO our approach results in a cluttered context (it includes the full "thinking" of the simple model)
    # it would be nice if we could strip that out without adding yet another pass
    for file_path in relevant_files:
        print(f"{file_path}:\n{full_results_dict[file_path]}\n")
        
    # Clean up cache unless --save-cache was specified
    if not args.save_cache:
        shutil.rmtree(cache_dir)

if __name__ == "__main__":
    main()
