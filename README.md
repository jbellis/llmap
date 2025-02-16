# Context extraction at scale

Tools like Aider and Cursor are great at editing code for you once you give them the right context. But 
[finding that context automatically is largely an unsolved problem](https://spyced.blogspot.com/2024/12/the-missing-piece-in-ai-coding.html),
especially in large codebases.

LLMap is a CLI code search tool designed to solve that problem by asking
Gemini Flash (preferred) or DeepSeek V3 to evaluate the relevance of each source file
in your codebase to your problem.

Until recently, this would be prohibitively expensive and slow.  But these models are not only
smart and fast, but also cheap enough to search large codebases exhaustively without worrying about the price.

LLMap also structures its request to take advantage of DeepSeek's automatic caching.  This means that repeated
searches against the same files will be [faster and less expensive](https://api-docs.deepseek.com/guides/kv_cache).
(It is possible to also support this for Gemini but in Gemini caching is not automatic and costs extra.)

Finally, LLMap optimizes the problem by using a multi-stage analysis to avoid spending more time
than necessary analyzing obviously irrelevant files.  LLMap performs 3 stages of analysis:
 1. Coarse analysis using code skeletons [Flash/V3]
 2. Full source analysis of potentially relevant files from (1) [Flash/V3]
 3. Refine the output of (2) to only the most relevant snippets [Pro/R1]

## Limitations

Currently only Java, Python, and C# files are supported by the skeletonization pass.  
LLMap will process other source files, but it will perform full source analysis on all of them,
which will be slower.

[Extending the parsing to other languages](https://github.com/jbellis/llmap/blob/master/src/llmap/parse.py)
is straightforward; contributions are welcome.

## Installation

```bash
pip install llmap-ai
```

Get a Gemini API key from [ai.google.dev](https://ai.google.dev/)
or a DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com).

## Usage

```bash
export GEMINI_API_KEY=YYY # or DEEPSEEK_API_KEY if using DeepSeek

find src/ -name "*.java" | llmap "Where is the database connection configured?"
```

LLMs APIs are not super reliable, so LLMap caches LLM responses in `~/.cache/llmap`
so that you don't have to start over from scratch if you get rate limited or run into another hiccup.
(This also means that if you want to check the raw, unrefined output [see below], you won't have to
reprocess the search.)

## Output

LLMap prints the most relevant context found to stdout.  You can save this to a file and send it to Aider
or attach it to a conversation with your favorite AI chat tool.

Errors are logged to stderr.

## Didn't find what you were looking for?

First, try passing `--no-refine`.  While the refine step is usually helpful in filtering out the noise
(thus taking up less of your context window), sometimes it's too aggressive.

You can also try passing `--no-skeletons` in case DeepSeek was too conservative in its initial filtering. 

Finally, try rephrasing your question with more clues for the LLM to latch onto.  Like any information
retrieval tool, sometimes the way you ask can make a big difference.
- Worse: "How can I add a WITH clause to the CQL SELECT statement?"
- Better: "How can I add a WITH clause to the CQL SELECT statement? It will be used for adding query planning hints like which index to use."

## Options

Commandline parameters:
```
  --sample SAMPLE       Number of random files to sample from the input set
  --llm-concurrency LLM_CONCURRENCY
                        Maximum number of concurrent LLM requests
  --no-refine           Skip refinement and combination of analyses
  --no-skeletons        Skip skeleton analysis phase for all files
```

Environment variables:
```
  LLMAP_CACHE           none|read|write|read/write
  LLMAP_ANALYZE_MODEL   deepseek-chat|deepseek-reasoner
  LLMAP_REFINE_MODEL    deepseek-chat|deepseek-reasoner
```
