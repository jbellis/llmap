# Context extraction at scale

LLMap is a CLI tool that uses AI to perform brute-force RAG against source files using Deepseek v3 and Gemini Flash.

(Deepseek is preferred internally, but when analysis requires more tokens than Deepseek can handle, Flash is used.)

LLMap performs 3 stages of analysis for each source file:
 1. Coarse analysis using code skeletons
 2. Full source analysis of potentially relevant files from (1)
 3. Refine the output of (2) to only the most relevant snippets

Until recently, this would be prohibitively expensive and slow.  But Deepseek-V3 is cheap, smart, fast,
and most importantly, it allows multiple concurrent requests.  LLMap performs the above analysis
(by default) 500 files at a time,
so it's reasonably fast even for large codebases.

## Limitations

Currently only Java and Python files are supported by the skeletonization in parse.py.

LLMap will process other source files, but it will perform full source analysis on all of them,
which will be slower.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
export GEMINI_API_KEY=XXX
export DEEPSEEK_API_KEY=YYY

find src/ -name "*.java" | python llmap.py "Where is the database connection configured?"
```

LLMs APIs are not super reliable, so LLMap caches LLM responses in `.llmap_cache` by question and by processing
stage, so that you don't have to start over from scratch if you get rate limited or run into another hiccup.

## Output

LLMap prints the most relevant context found to stdout.  You can save this to a file and send it to Aider
or attach it to a conversation with your favorite AI chat tool.

Errors are logged to stderr.

## Debugging

You can see the skeletonized code that llmap sends to the LLM with `parse.py`
```bash
python parse.py [filename]
```

`llmap.py` also takes some debugging parameters, and running it with env variable `LLMAP_VERBOSE=1` will print out each LLM response.

Use `--save-cache` to preserve the cache directory, otherwise it is cleaned out on successful completion.
