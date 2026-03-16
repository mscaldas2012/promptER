import re


def extract_thinking(text: str):
    """Extract <think>...</think> blocks from LLM output.

    Returns (clean_text, thinking_text) where:
    - clean_text is the response with all think blocks removed and whitespace trimmed.
    - thinking_text is the concatenated content of all think blocks, or None if none found.

    Handles multiple blocks and is case-insensitive.
    """
    if not text:
        return text, None

    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
    blocks = think_pattern.findall(text)

    if not blocks:
        return text, None

    clean_text = think_pattern.sub('', text).strip()
    thinking_text = '\n\n---\n\n'.join(block.strip() for block in blocks)

    return clean_text, thinking_text
