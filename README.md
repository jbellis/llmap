# Context extraction at scale

LLMap is a CLI tool that uses AI to perform brute-force RAG against source files using Deepseek v3 and Gemini Flash.

(Deepseek is preferred internally, but when analysis requires more tokens than Deepseek can handle, Gemini Flash is used.)

LLMap performs 2-4 stages for each source file:
 1. Extract context using code skeletons
 2. Determine relevance of skeletonized context
 3. (Optional) Extract context using full source
 4. (Optional) Determine relevance of full source context

Until recently, this would be prohibitively expensive and slow.  But Deepseek-V3 is cheap, fast, and most
importantly allows multiple concurrent requests.  LLMap performs the above analysis 500 files at a time,
so it's reasonably fast even for large codebases.

## Limitations

Currently only Java files are supported by the skeletonization in parse.py.  

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

LLMs APIs are not super reliable, so LLMap caches LLM responses in `.llmap_cache` by question and by processing
stage, so that you don't have to start over from scratch if you get rate limited or run into another hiccup.

Use `--save-cache` to preserve the cache directory after completion, otherwise it is cleaned out on successful
completion.

## Output

For each relevant file, prints to stdout:
- File path
- AI analysis explaining relevance and most important code snippets

Errors are logged to stderr.

## Debugging

You can see the skeletonized code that llmap sends to the LLM with `parse.py`
```bash
python parse.py [filename]
```

`llmap.py` also takes some debugging parameters, and running it with env variable `LLMAP_VERBOSE=1` will print out each LLM response.

Finally, the evaluations performed by the LLM are logged to `evaluation.jsonl`.
