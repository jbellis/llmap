# pip3 install transformers
# python3 __init__.py
import os
import transformers

# Get the directory containing this script
_current_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer = transformers.AutoTokenizer.from_pretrained(
        _current_dir, trust_remote_code=True
        )
