# features.py
import torch
import torch.nn.functional as F

def compute_basic_logit_features(critic, texts, max_length=1024):
    """
    critic: CriticModel instance from models.py
    texts: list[str]
    """
    batch = critic.tokenize(texts, max_length=max_length)
    logits = critic.get_logits(batch)           # [B, T, V]
    logprobs = F.log_softmax(logits, dim=-1)    # [B, T, V]

    # gather logprobs of the actual tokens
    input_ids = batch["input_ids"]
    token_logprobs = logprobs.gather(
        -1, input_ids.unsqueeze(-1)
    ).squeeze(-1)                               # [B, T]

    # mask padding
    attn = batch["attention_mask"]
    lengths = attn.sum(dim=1)

    # simple example: mean logprob per sequence
    seq_logprob_mean = (token_logprobs * attn).sum(dim=1) / lengths

    return {
        "mean_logprob": seq_logprob_mean.cpu().numpy(),
        # you can add entropy, margins, etc., here
    }
