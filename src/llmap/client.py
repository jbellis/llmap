import hashlib
import json
import os
import sys
import time
from random import random
from typing import NamedTuple

import httpx
from openai import OpenAI, BadRequestError, AuthenticationError, PermissionDeniedError, UnprocessableEntityError, \
    RateLimitError, APIError

from .cache import Cache
from .exceptions import AIRequestException, AITimeoutException


class FakeInternalServerError(Exception):
    pass


class SourceText(NamedTuple):
    file_path: str
    text: str


class CachingClient:
    def __init__(self):
        # Set up caching based on LLMAP_CACHE env var
        cache_mode = os.getenv('LLMAP_CACHE', 'read/write').lower()
        if cache_mode not in ['none', 'read', 'write', 'read/write']:
            raise ValueError("LLMAP_CACHE must be one of: none, read, write, read/write")
        self.cache_mode = cache_mode
        self.cache = None if cache_mode == 'none' else Cache()

        # Initialize API configuration
        self._setup_api()

        # Progress callback will be set per-phase
        self.progress_callback = None

    def _setup_api(self):
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        gemini_api_key = os.getenv('GEMINI_API_KEY')

        if not (deepseek_api_key or gemini_api_key or openrouter_api_key):
            raise Exception("Either DEEPSEEK_API_KEY or GEMINI_API_KEY environment variable must be set")

        if gemini_api_key:
            valid_models = {'gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-2.0-pro-exp-02-05'}
            self.api_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            self.api_key = gemini_api_key
            self.analyze_model = os.getenv('LLMAP_ANALYZE_MODEL', 'gemini-2.0-flash')
            self.refine_model = os.getenv('LLMAP_REFINE_MODEL', 'gemini-2.0-pro-exp-02-05')
            print("Using Gemini API", file=sys.stderr)
        elif deepseek_api_key:
            valid_models = {'deepseek-chat', 'deepseek-reasoner'}
            self.api_base_url = "https://api.deepseek.com"
            self.api_key = deepseek_api_key
            self.analyze_model = os.getenv('LLMAP_ANALYZE_MODEL', 'deepseek-chat')
            self.refine_model = os.getenv('LLMAP_REFINE_MODEL', 'deepseek-reasoner')
            print("Using DeepSeek API", file=sys.stderr)
        else: # Open Router
            valid_models = {'deepseek/deepseek-chat', 'deepseek/deepseek-r1'}
            self.api_base_url = "https://openrouter.ai/api/v1"
            self.api_key = openrouter_api_key
            self.analyze_model = os.getenv('LLMAP_ANALYZE_MODEL', 'deepseek/deepseek-chat')
            self.refine_model = os.getenv('LLMAP_REFINE_MODEL', 'deepseek/deepseek-r1')
            print("Using OpenRouter API", file=sys.stderr)
        
        if self.analyze_model not in valid_models:
            raise ValueError(f"LLMAP_ANALYZE_MODEL must be one of: {', '.join(valid_models)}")
        if self.refine_model not in valid_models:
            raise ValueError(f"LLMAP_REFINE_MODEL must be one of: {', '.join(valid_models)}")

        self.llm_client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)

    def max_tokens(self) -> int:
        """Return the maximum tokens allowed for the current API"""
        if self.api_base_url == "https://api.deepseek.com":
            return 62000 - 8000  # output 8k counts towards 64k limit. Headroom for scaffolding
        else:
            return 500000

    def ask(self, messages, model, file_path=None):
        """Helper method to make requests to the API with error handling, retries and caching"""
        # Try to load from cache if reading enabled
        cache_key = _make_cache_key(messages, model)
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

        for attempt in range(10):
            stream = None
            try:
                stream = self.llm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
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
                with open('/tmp/deepseek_error.log', 'a') as f:
                    print(f"{messages}\n\n->\n{e}", file=f)
                raise AIRequestException("Error evaluating source code", file_path, e)
            except RateLimitError:
                # print("Rate limited, waiting", file=sys.stderr)
                time.sleep(5 * random() + 2 ** attempt)
            except (httpx.RemoteProtocolError, APIError, FakeInternalServerError):
                time.sleep(1)
            finally:
                if stream:
                    stream.close()
        else:
            raise AITimeoutException("Repeated timeouts evaluating source code", file_path)


def _make_cache_key(messages: list, model: str) -> str:
    return hashlib.sha256(json.dumps([messages, model]).encode()).hexdigest()