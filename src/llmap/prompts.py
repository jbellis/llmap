from textwrap import dedent

from .client import CachingClient, SourceText


def multi_skeleton_relevance(client: CachingClient, skeletons: list[SourceText], question: str) -> str:
    """
    Evaluate multiple skeletons for relevance.
    Skeletons is a list of SourceText objects containing file paths and skeleton text.
    Returns a string containing only the relevant file paths (one per line),
    or no paths if none are relevant.
    """
    combined = []
    for skeleton in skeletons:
        combined.append(f"### FILE: {skeleton.file_path}\n{skeleton.text}\n")
    combined_text = "\n\n".join(combined)

    messages = [
        {"role": "system", "content": "You are a helpful assistant designed to analyze and explain source code."},
        {"role": "user", "content": combined_text},
        {"role": "assistant", "content": "Thank you for providing your source code skeletons for analysis."},
        {"role": "user", "content": dedent(f"""
            I have given you multiple file skeletons, each labeled with "### FILE: path".
            Evaluate each skeleton for relevance to the following question:
            ```
            {question}
            ```

            Think about whether the skeleton provides sufficient information to determine relevance:
            - If the skeleton clearly indicates irrelevance to the question, eliminate it from consideration.
            - If the skeleton clearly shows that the code is relevant to the question,
              OR if implementation details are needed to determine relevance, output its FULL path.
            List ONLY the file paths that appear relevant to answering the question. 
            Output one path per line. If a file is not relevant, do not list it at all.
        """)},
        {"role": "assistant", "content": "Understood."},
    ]
    response = client.ask(messages, client.analyze_model)
    return response.choices[0].message.content


def full_source_relevance(client: CachingClient, source: str, question: str, file_path: str = None) -> SourceText:
    """
    Check source code for relevance
    Args:
        source: The source code to analyze
        question: The question to check relevance against
        file_path: Optional file path for error reporting
    Returns SourceAnalysis containing file path and evaluation text
    Raises AIException if a recoverable error occurs.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant designed to analyze and explain source code."},
        {"role": "user", "content": source},
        {"role": "assistant", "content": "Thank you for providing your source code for analysis."},
        {"role": "user", "content": dedent(f"""
            Evaluate the above source code for relevance to the following question:
            ```
            {question}
            ```

            Give an overall summary, then give the most relevant section(s) of code, if any.
            Prefer to give relevant code in units of functions, classes, or methods, rather
            than isolated lines.
        """)}
    ]

    response = client.ask(messages, client.analyze_model, file_path)
    return SourceText(file_path, response.choices[0].message.content)


def refine_context(client: CachingClient, file_group: list[SourceText], question: str) -> str:
    """
    Process groups of file analyses to extract only the relevant context.

    Args:
        file_groups: List of lists of (file_path, analysis) tuples
        question: The original question being analyzed

    Returns:
        List of processed contexts, one per group
    """
    combined = "\n\n".join(f"File: {analysis.file_path}\n{analysis.text}" for analysis in file_group)

    messages = [
        {"role": "system", "content": "You are a helpful assistant designed to collate source code."},
        {"role": "user", "content": combined},
        {"role": "assistant", "content": "Thank you for providing your source code fragments."},
        {"role": "user", "content": dedent(f"""
            The above text contains analysis of multiple source files related to this question:
            ```
            {question}
            ```

            Extract only the most relevant context and code sections that help answer the question.
            Remove any irrelevant files completely, but preserve file paths for the relevant code fragments.
            Include the relevant code fragments as-is; do not truncate, summarize, or modify them.

            DO NOT include additional commentary or analysis of the provided text.
        """)}
    ]

    response = client.ask(messages, client.refine_model)
    content1 = response.choices[0].message.content
    messages += [
        {"role": "assistant", "content": content1},
        {"role": "user", "content": dedent(f"""
            Take one more look and make sure you didn't miss anything important for answering
            the question:
            ```
            {question}
            ```
        """)}
    ]
    response = client.ask(messages, client.refine_model)
    content2 = response.choices[0].message.content

    return content1 + '\n\n' + content2
