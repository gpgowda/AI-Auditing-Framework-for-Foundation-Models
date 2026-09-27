"""
Wraps a raw CSV prompt with a standard instruction before it's sent to
any model. Centralizing this here (rather than editing each wrapper)
means every model gets the exact same instruction — necessary for a
fair cross-model comparison, not just a cost-saving trick.
"""

CONCISE_INSTRUCTION = (
    "\n\nAnswer in 2-3 sentences maximum. Do not use headers, bullet "
    "points, or markdown formatting — plain prose only."
)


def build_prompt(raw_prompt: str, concise: bool = True) -> str:
    if concise:
        return raw_prompt.strip() + CONCISE_INSTRUCTION
    return raw_prompt