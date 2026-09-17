# rewards/reward_functions.py

import torch
import torch.nn.functional as F

def compute_perplexity(model, tokenizer, text: str, device="cpu") -> float:
    """
    Compute perplexity of generated text under the base model.
    High perplexity = watermark degraded fluency = penalty for Agent A.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
        loss = outputs.loss   # cross-entropy loss

    return torch.exp(loss).item()

def type_accuracy(pred_type: str, true_type: str) -> float:
    return 1.0 if pred_type == true_type else 0.0