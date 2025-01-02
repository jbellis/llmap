# Java Code Relevance Analyzer

CLI tool that uses AI to perform brute-force RAG against Java source files using Deepseek v3 and Gemini Flash.

(Deepseek is preferred internally, but when analysis requires more tokens than Deepseek can handle, Gemini Flash is used.)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
export GOOGLE_API_KEY=XXX
export DEEPSEEK_API_KEY=YYY

python llmap.py --directory src/ "Where is the database connection configured?"
```

## Debugging

You can see the skeletonized code that llmap sends to the LLM with `parse.py`
```bash
python parse.py [filename]
```

`llmap.py` also takes some debugging parameters, and running it with env variable `LLMAP_VERBOSE=1` will print out each LLM response.

Finally, the evaluations performed by the LLM are logged to `evaluation.jsonl`.

## Output

For each relevant file, prints to stdout:
- File path
- AI analysis explaining relevance and most important code snippets

Errors are logged to stderr.
