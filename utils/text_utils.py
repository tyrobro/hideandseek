# utils/text_utils.py
# Shared tokenization and text helpers used across agents and the environment.

from typing import List, Tuple
import torch


def encode(tokenizer, text: str, device: str = "cpu") -> torch.Tensor:
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def decode(tokenizer, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(token_ids.squeeze(), skip_special_tokens=True)


def token_span_to_char_span(
    tokenizer, text: str, start_token: int, end_token: int
) -> Tuple[int, int]:
    """Convert token indices to character offsets — useful for logging."""
    encoding = tokenizer(text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    char_start = offsets[start_token][0] if start_token < len(offsets) else 0
    char_end   = offsets[end_token][1]   if end_token   < len(offsets) else len(text)
    return char_start, char_end


def sample_prompts(dataset, n: int = 1) -> List[str]:
    """Sample n prompts from a HuggingFace dataset."""
    indices = torch.randint(0, len(dataset), (n,)).tolist()
    return [dataset[i]["text"][:200] for i in indices]  # first 200 chars as prompt