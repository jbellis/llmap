import os
import sys
import google.generativeai as genai
from pathlib import Path

def find_relevant_files(query, source_dir):
    """Find files relevant to a query using Claude AI."""
    # Get list of all files recursively
    all_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), source_dir)
            all_files.append(rel_path)
    
    # Format files list for Claude
    files_list = "\n".join(all_files)
    
    # Configure Gemini
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    
    # Create Gemini client
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Construct prompt
    prompt = f"""Given this list of files:

{files_list}

Which of these files might be relevant to this query: "{query}"?
Return ONLY the filenames, one per line.
Don't explain your choices, just list the filenames."""

    # Get response from Gemini
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0
        )
    )
    
    # Parse response
    relevant_files = response.text.strip().split('\n')
    
    # Check existence and print results
    results = []
    for file in relevant_files:
        file = file.strip()
        if file:  # Skip empty lines
            exists = os.path.exists(os.path.join(source_dir, file))
            print(f"{file}: {exists}")
            results.append((file, exists))
    
    return results

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ask.py <source_dir> <query>")
        sys.exit(1)
        
    source_dir = sys.argv[1]
    query = sys.argv[2]
    
    if not os.path.isdir(source_dir):
        print(f"Error: {source_dir} is not a directory")
        sys.exit(1)
        
    find_relevant_files(query, source_dir)
