import hashlib
import json
import os
import time
from textwrap import dedent
from typing import NamedTuple
import httpx

from openai import OpenAI, BadRequestError, APITimeoutError, APIConnectionError, APIStatusError, RateLimitError
class FakeInternalServerError(Exception):
    pass


class SourceAnalysis(NamedTuple):
    file_path: str
    analysis: str

from .deepseek_v3_tokenizer import tokenizer
from .exceptions import AIException
from .parse import extract_skeleton, MAX_DEEPSEEK_TOKENS
from .cache import Cache


def collate(analyses: list[SourceAnalysis]) -> tuple[list[list[SourceAnalysis]], list[SourceAnalysis]]:
    """
    Group analyses into batches that fit under token limit, and separate out large files.
    
    Args:
        analyses: List of SourceAnalysis objects
    
    Returns:
        Tuple of (grouped_analyses, large_files) where:
        - grouped_analyses is a list of lists of SourceAnalysis objects, each group under MAX_DEEPSEEK_TOKENS
        - large_files is a list of SourceAnalysis objects that individually exceed MAX_DEEPSEEK_TOKENS
    """
    large_files = []
    small_files = []
    
    # Separate large and small files
    for analysis in analyses:
        tokens = len(tokenizer.encode(analysis.analysis))
        if tokens > MAX_DEEPSEEK_TOKENS:
            large_files.append(analysis)
        else:
            small_files.append((analysis, tokens))
    
    # Group small files
    groups = []
    current_group = []
    current_tokens = 0
    
    for analysis, tokens in small_files:
        if current_tokens + tokens > MAX_DEEPSEEK_TOKENS:
            if current_group:  # Only append if group has items
                groups.append(current_group)
            current_group = [analysis]
            current_tokens = tokens
        else:
            current_group.append(analysis)
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
    def __init__(self, cache_dir=None):  # cache_dir kept for backwards compatibility
        # Progress callback will be set per-phase
        self.progress_callback = None
        # Set up caching based on LLMAP_CACHE env var
        cache_mode = os.getenv('LLMAP_CACHE', 'read/write').lower()
        if cache_mode not in ['none', 'read', 'write', 'read/write']:
            raise ValueError("LLMAP_CACHE must be one of: none, read, write, read/write")
        self.cache_mode = cache_mode
        self.cache = None if cache_mode == 'none' else Cache()
        
        # Get environment variables
        deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        if not deepseek_api_key:
            raise Exception("DEEPSEEK_API_KEY environment variable not set")
            
        # Validate model names
        valid_models = {'deepseek-chat', 'deepseek-reasoner'}
        
        self.analyze_model = os.getenv('LLMAP_ANALYZE_MODEL', 'deepseek-chat')
        if self.analyze_model not in valid_models:
            raise ValueError(f"LLMAP_ANALYZE_MODEL must be one of: {', '.join(valid_models)}")
            
        self.refine_model = os.getenv('LLMAP_REFINE_MODEL', 'deepseek-reasoner')
        if self.refine_model not in valid_models:
            raise ValueError(f"LLMAP_REFINE_MODEL must be one of: {', '.join(valid_models)}")
        
        self.deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    def ask_deepseek(self, messages, model, file_path=None):
        """Helper method to make requests to DeepSeek API with error handling, retries and caching"""
        # Create cache key from messages and model
        cache_key = _make_cache_key(messages, model)
        
        # Try to load from cache if reading enabled
        if self.cache and self.cache_mode in ['read', 'read/write']:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return type('Response', (), {
                    'choices': [type('Choice', (), {
                        'message': type('Message', (), {
                            'content': cached_data['answer']
                        })
                    })]
                })

        # Call API if not in cache or cache read disabled
        for attempt in range(5):
            try:
                stream = self.deepseek_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,  # Enable streaming
                    max_tokens=8000,
                )
                
                full_content = []
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        delta = chunk.choices[0].delta.content
                        full_content.append(delta)
                        
                        # Update progress based on newlines received
                        if self.progress_callback:
                            new_lines = delta.count('\n')
                            if new_lines > 0:
                                self.progress_callback(new_lines)
                
                content = ''.join(full_content)
                if not content.strip():
                    raise FakeInternalServerError()
                
                # Save to cache if enabled
                if self.cache and self.cache_mode in ['write', 'read/write']:
                    self.cache.set(cache_key, {'answer': content})
                
                # Return mock response object
                return type('Response', (), {
                    'choices': [type('Choice', (), {
                        'message': type('Message', (), {
                            'content': content
                        })
                    })]
                })
            except BadRequestError as e:
                # log the request to /tmp/deepseek_error.log
                with open('/tmp/deepseek_error.log', 'a') as f:
                    print(f"{messages}\n\n->\n{e}", file=f)
                raise AIException("Error evaluating source code", file_path, e)
            except RateLimitError:
                time.sleep(5)
            except (httpx.RemoteProtocolError, APITimeoutError, APIConnectionError, APIStatusError, FakeInternalServerError):
                time.sleep(1)  # Wait 1 second before retrying
        else:
            raise AIException("Repeated timeouts evaluating source code", file_path)

    def skeleton_relevance(self, full_path: str, question: str) -> SourceAnalysis:
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
            {"role": "assistant", "content": "Thank you for providing your source code skeleton for analysis."},
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

        response = self.ask_deepseek(messages, self.analyze_model, full_path)
        content = response.choices[0].message.content
        # if the response doesn't contain any of the expected choices, try again
        if not any(choice in content
                   for choice in {'LLMAP_RELEVANT', 'LLMAP_IRRELEVANT'}):
            messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": dedent(f"""
                    - If the skeleton clearly indicates irrelevance to the question, conclude LLMAP_IRRELEVANT.
                    - If the skeleton clearly shows that the code is relevant to the question,
                      OR if implementation details are needed to determine relevance, conclude LLMAP_RELEVANT.
                """)}
            ]
            response = self.ask_deepseek(messages, self.analyze_model, full_path)
            content = response.choices[0].message.content
        # if it still doesn't contain any of the expected choices, raise an exception
        if not any(choice in content
                   for choice in {'LLMAP_RELEVANT', 'LLMAP_IRRELEVANT'}):
            if self.cache:
                cache_key = _make_cache_key(messages, self.analyze_model)
                self.cache.delete(cache_key)
            raise AIException("Failed to get a valid response from DeepSeek", full_path)
        return SourceAnalysis(full_path, content)

    def full_source_relevance(self, source: str, question: str, file_path: str = None) -> SourceAnalysis:
        """
        Check source code for relevance
        Args:
            source: The source code to analyze
            question: The question to check relevance against
            file_path: Optional file path for error reporting
        Returns SourceAnalysis containing file path and evaluation text
        Raises AIException if a recoverable error occurs.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to analyze and explain source code."},
            {"role": "user", "content": source},
            {"role": "assistant", "content": "Thank you for providing your source code for analysis."},
            {"role": "user", "content": dedent(f"""
                Evaluate the above source code for relevance to the following question:
                ```
                {question}
                ```

                Give an overall summary, then give the most relevant section(s) of code, if any.
                Prefer to give relevant code in units of functions, classes, or methods, rather
                than isolated lines.
            """)}
        ]

        response = self.ask_deepseek(messages, self.analyze_model, file_path)
        return SourceAnalysis(file_path, response.choices[0].message.content)

    def sift_context(self, file_group: list[SourceAnalysis], question: str) -> str:
        """
        Process groups of file analyses to extract only the relevant context.

        Args:
            file_groups: List of lists of (file_path, analysis) tuples
            question: The original question being analyzed

        Returns:
            List of processed contexts, one per group
        """
        combined = "\n\n".join(f"File: {analysis.file_path}\n{analysis.analysis}" for analysis in file_group)

        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to collate source code."},
            {"role": "user", "content": combined},
            {"role": "assistant", "content": "Thank you for providing your source code fragments."},
            {"role": "user", "content": dedent(f"""
                The above text contains analysis of multiple source files related to this question:
                ```
                {question}
                ```

                Extract only the most relevant context and code sections that help answer the question.
                Remove any irrelevant files completely, but preserve file paths for the relevant code fragments.
                Include the relevant code fragments as-is; do not truncate, summarize, or modify them.
                
                Do not include additional commentary or analysis of the provided text.
            """)}
        ]

        response = self.ask_deepseek(messages, self.refine_model)
        content1 = response.choices[0].message.content
        messages += [
            {"role": "assistant", "content": content1},
            {"role": "user", "content": dedent(f"""
                Take one more look and make sure you didn't miss anything important for answering
                the question:
                ```
                {question}
                ```
            """)}
        ]
        response = self.ask_deepseek(messages, self.refine_model)
        content2 = response.choices[0].message.content

        return content1 + '\n\n' + content2


def _make_cache_key(messages: list, model: str) -> str:
    return hashlib.sha256(json.dumps([messages, model]).encode()).hexdigest()

