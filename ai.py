import os
import json
import datetime
from openai import OpenAI, BadRequestError
from deepseek_v2_tokenizer import tokenizer
from exceptions import AIException
from llmap import extract_skeleton
from textwrap import dedent

MAX_TOKENS = 64000  # Leave some headroom for message scaffolding while staying under 64k token limit

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

def check_full_source(file_path: str, question: str, client: 'AI') -> tuple[str, str]:
    """
    Check the full source of a file for relevance
    Raises AIException if a recoverable error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            source = f.read()
    except Exception as e:
        return file_path, f"Error reading file: {e}"

    # Truncate if needed
    source = maybe_truncate(source, MAX_TOKENS)  # Reuse truncate function
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that responds with only a single word without elaboration, not even punctuation."},
        {"role": "user",
         "content": f"Given this source code:\n\n{source}\n\nIs this code relevant to the problem or question: {question}? Answer with only yes or no"}
    ]

    try:
        response = client.deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
    except BadRequestError as e:
        raise AIException("Error evaluating source code", file_path, e)

    return file_path, response.choices[0].message.content.lower().strip()

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

    def generate_relevance(self, full_path: str, question: str) -> tuple[str, str]:
        """
        Check if a source file is relevant to the question using DeepSeek.
        Raises AIException if a recoverable error occurs.
        """
        skeleton = extract_skeleton(full_path)
        
        # Truncate if needed
        skeleton = maybe_truncate(skeleton, MAX_TOKENS)
        
        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to analyze and explain source code."},
            {"role": "user", "content": dedent(f"""
                Given this code skeleton:

                ```
                {skeleton}
                ```

                Is this code relevant to the following problem or question?

                    {question}

                Give your reasoning, then a final verdict of Relevant or Irrelevant.  If the full source code
                is needed to determine relevance, your final verdict should be "Full Source Needed".
            """)}
        ]

        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
        except BadRequestError as e:
            raise AIException("Error evaluating source code", full_path, e)

        answer = response.choices[0].message.content.lower().strip()
        
        # Log all evaluations in JSONL format
        eval_data = {
            "file": full_path,
            "response": answer,
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open('evaluation.jsonl', 'a') as f:
            f.write(json.dumps(eval_data) + '\n')
            
        return full_path, answer

    def evaluate_relevance(self, full_path: str, evaluation_text: str) -> tuple[str, bool]:
        """
        Convert LLM's evaluation text into a boolean relevance decision
        Raises AIException if a recoverable error occurs.
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
        
                Answer with only Relevant, Irrelevant, or Source.
            """)}
        ]

        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
        except BadRequestError as e:
            raise AIException("Error evaluating relevance", full_path, e)
        verdict = response.choices[0].message.content.lower().strip()
        is_relevant = verdict == "relevant"
        return full_path, is_relevant
