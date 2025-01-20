# Context extraction at scale

Tools like Aider and Cursor are great at editing code for you once you give them the right context. But 
[finding that context automatically is largely an unsolved problem](https://spyced.blogspot.com/2024/12/the-missing-piece-in-ai-coding.html),
especially in large codebases.

LLMap is a CLI code search tool designed to solve that problem by asking  
[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) and DeepSeek-R1 to evaluate the relevance of each source file
in your codebase to your problem.

Until recently, this would be prohibitively expensive and slow.  But DeepSeek-V3 is cheap, smart, fast,
and most importantly, it allows multiple concurrent requests.  LLMap performs its analysis
(by default) 500 files at a time, so it's reasonably fast even for large codebases.

LLMap also structures its request to take advantage of DeepSeek's caching.  This means that repeated
searches against the same files will be [faster and less expensive](https://api-docs.deepseek.com/guides/kv_cache).

Finally, LLMap optimizes the problem by using a multi-stage analysis to avoid spending more time
than necessary analyzing obviously irrelevant files.  LLMap performs 3 stages of analysis:
 1. Coarse analysis using code skeletons [DeepSeek-V3]
 2. Full source analysis of potentially relevant files from (1) [DeepSeek-V3]
 3. Refine the output of (2) to only the most relevant snippets [DeepSeek-R1]

## Limitations

Currently only Java and Python files are supported by the skeletonization pass.  
LLMap will process other source files, but it will perform full source analysis on all of them,
which will be slower.

[Extending the parsing to other languages](https://github.com/jbellis/llmap/blob/master/src/llmap/parse.py)
is straightforward; contributions are welcome.

## Installation

```bash
pip install llmap-ai
```

Get a DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com).

## Usage

```bash
export DEEPSEEK_API_KEY=YYY

find src/ -name "*.java" | llmap "Where is the database connection configured?"
```

LLMs APIs are not super reliable, so LLMap caches LLM responses in `.llmap_cache` by question and by processing
stage, so that you don't have to start over from scratch if you get rate limited or run into another hiccup.

The cache is automatically cleaned out when execution completes successfully.

## Output

LLMap prints the most relevant context found to stdout.  You can save this to a file and send it to Aider
or attach it to a conversation with your favorite AI chat tool.

Errors are logged to stderr.

## Options

```
  --sample SAMPLE       Number of random files to sample
  --save-cache          Keep cache directory after completion
  --llm-concurrency LLM_CONCURRENCY
                        Maximum number of concurrent LLM requests
  --no-refine           Skip refinement and combination of analyses
```