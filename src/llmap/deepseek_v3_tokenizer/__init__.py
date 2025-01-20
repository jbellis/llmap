import os
# suppress "None of PyTorch..." warning before importing transformers
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
import transformers

# Get the directory containing this script
_current_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer = transformers.AutoTokenizer.from_pretrained(
        _current_dir, trust_remote_code=True
        )
