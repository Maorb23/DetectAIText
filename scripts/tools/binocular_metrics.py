# ai_detect/tools/binocular_metrics.py
from __future__ import annotations

import numpy as np
import torch
import transformers

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")


def token_nll(encoding: transformers.BatchEncoding, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Per-token negative log-likelihood for next-token prediction.
    Returns: (B, T-1) tensor with masked positions kept but caller can mask.
    """
    shifted_logits = logits[..., :-1, :].contiguous() / temperature  # (B, T-1, V)
    shifted_labels = encoding.input_ids[..., 1:].contiguous()        # (B, T-1)
    nll = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) # (B, T-1)
    return nll


def perplexity_mean(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Mean token NLL (not exp). Binoculars uses ratios, so staying in NLL space is stable.
    Returns: (B,) numpy
    """
    nll = token_nll(encoding, logits, temperature=temperature)  # (B, T-1)
    mask = encoding.attention_mask[..., 1:].contiguous().bool() # (B, T-1)
    nll = (nll * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return nll.detach().to("cpu").float().numpy()


def cross_entropy_p_q_mean(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    encoding: transformers.BatchEncoding,
    pad_token_id: int,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Cross-entropy H(p, q) per token where:
      p = softmax(p_logits), q = softmax(q_logits)
      H(p,q) = - sum_i p_i log q_i

    Returns mean over non-pad tokens per example: (B,) numpy
    """
    # Align to next-token positions like perplexity
    p = torch.softmax(p_logits[..., :-1, :] / temperature, dim=-1)          # (B, T-1, V)
    log_q = torch.log_softmax(q_logits[..., :-1, :] / temperature, dim=-1)  # (B, T-1, V)

    ce = -(p * log_q).sum(dim=-1)  # (B, T-1)

    # mask: non-pad of the *label* tokens (input_ids shifted)
    shifted_ids = encoding.input_ids[..., 1:].contiguous() # contiguous asserts that memory layout is ok
    mask = (shifted_ids != pad_token_id)

    ce = (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1) # we clamp to avoid div by zero
    return ce.detach().to("cpu").float().numpy()

