import os
import sys
import json
import datetime
import threading

from openai import OpenAI, BadRequestError
from deepseek_v2_tokenizer import tokenizer
from exceptions import AIException
from parse import extract_skeleton
import re
from textwrap import dedent

MAX_DEEPSEEK_TOKENS = 64_000  # Leave some headroom for message scaffolding while staying under 64k token limit
MAX_GEMINI_TOKENS = 900_000   # Gemini limit is 1M but we're using the wrong tokenizer so be conservative

def clean_response(text: str) -> str:
    """Keep only alphanumeric characters and convert to lowercase"""
    return ''.join(c for c in text.lower() if c.isalnum())


def log_evaluation(file_path: str, answer: str) -> None:
    """Log an evaluation result to the evaluation.jsonl file"""
    eval_data = {
        "file": file_path,
        "response": answer,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open('evaluation.jsonl', 'a') as f:
        f.write(json.dumps(eval_data) + '\n')


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
    def __init__(self):
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

    def ask_gemini(self, messages, file_path=None):
        """Helper method to make requests to Gemini API with error handling"""
        # TODO upgrade to gemini-2.0-flash when available for production
        with self.gemini_lock:
            try:
                response = self.gemini_client.chat.completions.create(
                    model="gemini-1.5-flash",
                    messages=messages,
                    stream=False
                )
                if os.getenv('LLMAP_VERBOSE'):
                    print(f"Gemini response for {file_path}:", file=sys.stderr)
                    print("\t" + response.choices[0].message.content, file=sys.stderr)
                return response
            except BadRequestError as e:
                raise AIException("Error evaluating source code with Gemini", file_path, e)

    def generate_relevance(self, full_path: str, question: str) -> tuple[str, str]:
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
            {"role": "user", "content": dedent(f"""
                Given the following question,
                
                ```
                {question}
                ```

                Evaluate this source code skeleton for relevance to the question.  Give an overall summary, then give
                the most relevant section(s) of code, if any.  If the implementation details OF THIS FILE are key to
                determining relevance, respond with "Full Source Required" after explaining your reasoning.  It is not
                appropriate to respond with "Full Source Required" if it is clear from the skeleton that this file is not
                relevant.

                ```
                {skeleton}
                ```
            """)}
        ]

        response = self.ask_deepseek(messages, full_path)
        answer = response.choices[0].message.content
        log_evaluation(full_path, answer)
        return full_path, answer

    def generate_relevance_full_source(self, file_path: str, question: str) -> tuple[str, str]:
        """
        Check the full source of a file for relevance
        Returns tuple of (file_path, evaluation_text)
        Raises AIException if a recoverable error occurs.
        """
        try:
            with open(file_path, 'r') as f:
                source = f.read()
        except Exception as e:
            raise AIException(f"Error reading file: {e}", file_path)

        if len(tokenizer.encode(source)) > MAX_DEEPSEEK_TOKENS:
            source = maybe_truncate(source, MAX_GEMINI_TOKENS)
            f = self.ask_gemini
        else:
            f = self.ask_deepseek

        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to analyze and explain source code."},
            {"role": "user", "content": dedent(f"""
                Given the following question,
                
                ```
                {question}
                ```

                Evaluate this source code for relevance to the question.  Give an overall summary, then give
                the most relevant section(s) of code, if any.

                ```
                {source}
                ```
            """)}
        ]

        response = f(messages, file_path)
        answer = response.choices[0].message.content
        log_evaluation(file_path, answer)
        return file_path, answer

    def evaluate_relevance(self, full_path: str, evaluation_text: str, question: str, source_fallback: bool) -> tuple[str, str]:
        """
        Convert LLM's evaluation text into a relevance decision
        
        Args:
            full_path: Path to the source file
            evaluation_text: Text from LLM evaluation
            question: Original question being evaluated
            source_fallback: True if we should allow the LLM to request a second evaluation with the full source
            
        Returns:
            String verdict: "relevant", "irrelevant" or "source"
            
        Raises:
            AIException if a recoverable error occurs
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant that responds with only a single word without elaboration, not even punctuation."},
            {"role": "user", "content": dedent(f"""
                Based on this evaluation:
                
                ```
                {evaluation_text}
                ```
                
                Was the evaluated file relevant to the following problem or question?
        
                    {question}
        
                Answer with exactly one of these words:
                - "Relevant" if the file was definitely relevant
                - "Irrelevant" if the file was definitely not relevant
                {'- "Source" if the evaluation states or implies that the implementation details of the full source code are needed' if source_fallback else ''}
            """)}
        ]

        response = self.ask_deepseek(messages, full_path)
        return full_path, clean_response(response.choices[0].message.content)
