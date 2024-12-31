# pip3 install transformers
# python3 deepseek_v2_tokenizer.py
import os
import transformers

# Get the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer = transformers.AutoTokenizer.from_pretrained(
        current_dir, trust_remote_code=True
        )
