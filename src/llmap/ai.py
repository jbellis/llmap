import datetime
import hashlib
import json
import os
import os.path
import sys
import threading
from textwrap import dedent

from openai import OpenAI, BadRequestError

from .deepseek_v2_tokenizer import tokenizer
from .exceptions import AIException
from .parse import extract_skeleton

MAX_DEEPSEEK_TOKENS = 50_000  # deepseek v2 tokenizer undercounts v3 tokens, this keeps it under the actual 64k limit
MAX_GEMINI_TOKENS = 900_000   # Gemini limit is 1M but we're using the wrong tokenizer so be conservative


def collate(analyses: list[tuple[str, str]]) -> tuple[list[list[tuple[str, str]]], list[tuple[str, str]]]:
    """
    Group analyses into batches that fit under token limit, and separate out large files.
    
    Args:
        analyses: List of (file_path, analysis_text) tuples
    
    Returns:
        Tuple of (grouped_analyses, large_files) where:
        - grouped_analyses is a list of lists of (file_path, analysis) tuples, each group under MAX_DEEPSEEK_TOKENS
        - large_files is a list of (file_path, analysis) tuples that individually exceed MAX_DEEPSEEK_TOKENS
    """
    large_files = []
    small_files = []
    
    # Separate large and small files
    for file_path, analysis in analyses:
        tokens = len(tokenizer.encode(analysis))
        if tokens > MAX_DEEPSEEK_TOKENS:
            large_files.append((file_path, analysis))
        else:
            small_files.append((file_path, analysis, tokens))
    
    # Group small files
    groups = []
    current_group = []
    current_tokens = 0
    
    for file_path, analysis, tokens in small_files:
        if current_tokens + tokens > MAX_DEEPSEEK_TOKENS:
            if current_group:  # Only append if group has items
                groups.append(current_group)
            current_group = [(file_path, analysis)]
            current_tokens = tokens
        else:
            current_group.append((file_path, analysis))
            current_tokens += tokens
    
    if current_group:  # Add final group if it exists
        groups.append(current_group)
        
    return groups, large_files


def clean_response(text: str) -> str:
    """Keep only alphanumeric characters and convert to lowercase"""
    return ''.join(c for c in text.lower() if c.isalnum())


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


class AI:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

        # deepseek client
        deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        if not deepseek_api_key:
            raise Exception("DEEPSEEK_API_KEY environment variable not set")
        self.deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

        # gemini client
        gemini_api_key = os.getenv('GOOGLE_API_KEY')
        if not gemini_api_key:
            raise Exception("GOOGLE_API_KEY environment variable not set")
        self.gemini_client = OpenAI(api_key=gemini_api_key,
                                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",)
        self.gemini_lock = threading.Lock()

    def ask_deepseek(self, messages, file_path=None):
        """Helper method to make requests to DeepSeek API with error handling"""
        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            if os.getenv('LLMAP_VERBOSE'):
                print(f"DeepSeek response for {file_path}:", file=sys.stderr)
                print("\t" + response.choices[0].message.content, file=sys.stderr)
            return response
        except BadRequestError as e:
            raise AIException("Error evaluating source code", file_path, e)

    def skeleton_relevance(self, full_path: str, question: str) -> tuple[str, str]:
        """
        Check if a source file is relevant to the question using DeepSeek.
        Raises AIException if a recoverable error occurs.
        """
        skeleton = extract_skeleton(full_path)
        
        # Truncate if needed
        skeleton = maybe_truncate(skeleton, MAX_DEEPSEEK_TOKENS)
        
        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to analyze and explain source code."},
            {"role": "user", "content": skeleton},
            {"role": "user", "content": dedent(f"""
                Evaluate the above source code skeleton for relevance to the following question:
                ```
                {question}
                ```

                Think about whether the skeleton provides sufficient information to determine relevance:
                - If the skeleton clearly indicates irrelevance to the question, conclude LLMAP_IRRELEVANT.
                - If the skeleton clearly shows that the code is relevant to the question,
                  OR if implementation details are needed to determine relevance, conclude LLMAP_RELEVANT.
            """)}
        ]

        for _ in range(3):
            # try up to 3 times to get a valid response
            response = self.ask_deepseek(messages, full_path)
            if any(choice in response.choices[0].message.content
                   for choice in {'LLMAP_RELEVANT', 'LLMAP_IRRELEVANT', 'LLMAP_SOURCE'}):
                break
        else:
            raise AIException("Failed to get a valid response from DeepSeek", full_path)

        answer = response.choices[0].message.content
        return full_path, answer

    def full_source_relevance(self, source: str, question: str, file_path: str = None) -> tuple[str, str]:
        """
        Check source code for relevance
        Args:
            source: The source code to analyze
            question: The question to check relevance against
            file_path: Optional file path for error reporting
        Returns tuple of (file_path, evaluation_text)
        Raises AIException if a recoverable error occurs.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to analyze and explain source code."},
            {"role": "user", "content": source},
            {"role": "user", "content": dedent(f"""
                Evaluate the above source code for relevance to the following question:
                ```
                {question}
                ```

                Give an overall summary, then give the most relevant section(s) of code, if any.
                ```
                {source}
                ```
            """)}
        ]

        # Create cache key from messages
        cache_key = hashlib.sha256(json.dumps(messages).encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        # Try to load from cache
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                return file_path, cached_data['answer']

        # Call LLM if not in cache
        response = self.ask_deepseek(messages, file_path)
        
        # Save successful response to cache
        answer = response.choices[0].message.content
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump({
                'answer': answer,
                'timestamp': datetime.datetime.now().isoformat()
            }, f)
        return file_path, answer

    def sift_context(self, file_group: list[tuple[str, str]], question: str) -> str:
        """
        Process groups of file analyses to extract only the relevant context.

        Args:
            file_groups: List of lists of (file_path, analysis) tuples
            question: The original question being analyzed

        Returns:
            List of processed contexts, one per group
        """
        combined = "\n\n".join(f"File: {path}\n{analysis}" for path, analysis in file_group)

        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to collate source code."},
            {"role": "user", "content": combined},
            {"role": "user", "content": dedent(f"""
                The above text contains analysis of multiple source files related to this question:
                ```
                {question}
                ```

                Extract only the most relevant context and code sections that help answer the question.
                Remove any irrelevant files completely, but preserve file paths for the relevant code fragments.
                
                Do not include additional commentary or analysis of the provided text.
            """)}
        ]

        response = self.ask_deepseek(messages)
        return response.choices[0].message.content
