import hashlib
import json
import os
import sys
import time
from random import random
from textwrap import dedent
from typing import NamedTuple
import httpx

from openai import OpenAI, BadRequestError, AuthenticationError, PermissionDeniedError, UnprocessableEntityError, RateLimitError, APIError
class FakeInternalServerError(Exception):
    pass

from .deepseek_v3_tokenizer import tokenizer
from .exceptions import AIException
from .cache import Cache


class SourceText(NamedTuple):
    file_path: str
    text: str


class AI:
    def max_tokens(self) -> int:
        """Return the maximum tokens allowed for the current API"""
        if self.api_base_url == "https://api.deepseek.com":
            return 62000 - 8000  # output 8k counts towards 64k limit. Headroom for scaffolding
        else:
            return 500000

    def __init__(self, cache_dir=None):  # cache_dir kept for backwards compatibility
        # Progress callback will be set per-phase
        self.progress_callback = None
        # Set up caching based on LLMAP_CACHE env var
        cache_mode = os.getenv('LLMAP_CACHE', 'read/write').lower()
        if cache_mode not in ['none', 'read', 'write', 'read/write']:
            raise ValueError("LLMAP_CACHE must be one of: none, read, write, read/write")
        self.cache_mode = cache_mode
        self.cache = None if cache_mode == 'none' else Cache()
        
        # Get environment variables and set up API client
        deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not (deepseek_api_key or gemini_api_key):
            raise Exception("Either DEEPSEEK_API_KEY or GEMINI_API_KEY environment variable must be set")
            
        # Configure based on available API key
        if gemini_api_key:
            valid_models = {'gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-2.0-pro-exp-02-05'}
            self.api_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            self.api_key = gemini_api_key
            self.analyze_model = os.getenv('LLMAP_ANALYZE_MODEL', 'gemini-2.0-flash')
            self.refine_model = os.getenv('LLMAP_REFINE_MODEL', 'gemini-2.0-pro-exp-02-05')
            print("Using Gemini API", file=sys.stderr)
        else:
            valid_models = {'deepseek-chat', 'deepseek-reasoner'}
            self.api_base_url = "https://api.deepseek.com"
            self.api_key = deepseek_api_key
            self.analyze_model = os.getenv('LLMAP_ANALYZE_MODEL', 'deepseek-chat')
            self.refine_model = os.getenv('LLMAP_REFINE_MODEL', 'deepseek-reasoner')
            print("Using DeepSeek API", file=sys.stderr)

        # Validate model names
        if self.analyze_model not in valid_models:
            raise ValueError(f"LLMAP_ANALYZE_MODEL must be one of: {', '.join(valid_models)}")
            
        if self.refine_model not in valid_models:
            raise ValueError(f"LLMAP_REFINE_MODEL must be one of: {', '.join(valid_models)}")
        
        self.llm_client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)

    def ask_llm(self, messages, model, file_path=None):
        """Helper method to make requests to the API with error handling, retries and caching"""
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
            stream = None
            try:
                stream = self.llm_client.chat.completions.create(
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
            except (BadRequestError, AuthenticationError, PermissionDeniedError, UnprocessableEntityError) as e:
                # log the request to /tmp/deepseek_error.log
                with open('/tmp/deepseek_error.log', 'a') as f:
                    print(f"{messages}\n\n->\n{e}", file=f)
                raise AIException("Error evaluating source code", file_path, e)
            except RateLimitError:
                time.sleep(5 * random() + 2**attempt) # 2, 4, 8, 16, 32
            except (httpx.RemoteProtocolError, APIError, FakeInternalServerError):
                time.sleep(1)  # Wait 1 second before retrying
            finally:
                if stream:
                    stream.close()
        else:
            raise AIException("Repeated timeouts evaluating source code", file_path)

    def multi_skeleton_relevance(self, skeletons: list[SourceText], question: str) -> str:
        """
        Evaluate multiple skeletons for relevance.
        Skeletons is a list of SourceText objects containing file paths and skeleton text.
        Returns a string containing only the relevant file paths (one per line),
        or no paths if none are relevant.
        """
        # Combine all skeletons into a single message, labeling each with its file path
        combined = []
        for skeleton in skeletons:
            combined.append(f"### FILE: {skeleton.file_path}\n{skeleton.text}\n")
        combined_text = "\n\n".join(combined)

        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to analyze and explain source code."},
            {"role": "user", "content": combined_text},
            {"role": "assistant", "content": "Thank you for providing your source code skeletons for analysis."},
            {"role": "user", "content": dedent(f"""
                I have given you multiple file skeletons, each labeled with "### FILE: path".
                Evaluate each skeleton for relevance to the following question:
                ```
                {question}
                ```

                Think about whether the skeleton provides sufficient information to determine relevance:
                - If the skeleton clearly indicates irrelevance to the question, eliminate it from consideration.
                - If the skeleton clearly shows that the code is relevant to the question,
                  OR if implementation details are needed to determine relevance, output its FULL path.
                List ONLY the file paths that appear relevant to answering the question. 
                Output one path per line. If a file is not relevant, do not list it at all.
            """)},
            {"role": "assistant", "content": "Understood."},
        ]
        response = self.ask_llm(messages, self.analyze_model)
        return response.choices[0].message.content

    def full_source_relevance(self, source: str, question: str, file_path: str = None) -> SourceText:
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

        response = self.ask_llm(messages, self.analyze_model, file_path)
        return SourceText(file_path, response.choices[0].message.content)

    def sift_context(self, file_group: list[SourceText], question: str) -> str:
        """
        Process groups of file analyses to extract only the relevant context.

        Args:
            file_groups: List of lists of (file_path, analysis) tuples
            question: The original question being analyzed

        Returns:
            List of processed contexts, one per group
        """
        combined = "\n\n".join(f"File: {analysis.file_path}\n{analysis.text}" for analysis in file_group)

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
                
                DO NOT include additional commentary or analysis of the provided text.
            """)}
        ]

        response = self.ask_llm(messages, self.refine_model)
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
        response = self.ask_llm(messages, self.refine_model)
        content2 = response.choices[0].message.content

        return content1 + '\n\n' + content2


def _make_cache_key(messages: list, model: str) -> str:
    return hashlib.sha256(json.dumps([messages, model]).encode()).hexdigest()

