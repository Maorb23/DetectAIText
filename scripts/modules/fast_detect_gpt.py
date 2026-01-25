# scripts/modules/fastdetectgpt.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# -------- Core criterion (analytic) --------
@torch.inference_mode()
def sampling_discrepancy_analytic(
    logits_ref: torch.Tensor,    # (B, T, V)  reference/sampling model logits
    logits_score: torch.Tensor,  # (B, T, V)  scoring model logits
    labels: torch.Tensor,        # (B, T)     next-token labels (already shifted)
) -> torch.Tensor:
    """
    Vectorized version of the repo's get_sampling_discrepancy_analytic.
    Returns discrepancy per example: (B,)
    """
    # handle vocab mismatch by truncating to min vocab
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[..., :vocab_size]
        logits_score = logits_score[..., :vocab_size]

    # Overall function:
    # 
    # labels shape: (B, T, 1) for gather
    if labels.ndim == logits_score.ndim - 1:
        print("reshaping labels for gather")
        labels = labels.unsqueeze(-1)

    lprobs_score = torch.log_softmax(logits_score, dim=-1)    # (B, T, V)
    probs_ref = torch.softmax(logits_ref, dim=-1)             # (B, T, V)

    # log-likelihood of observed labels under scoring model
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)  # Matches the probs to the true labels
                                                                            # (B, T)

    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)   # Get the mean of the scoring model under the reference model 
                                                        # (B, T) 
    var_ref = (probs_ref * (lprobs_score ** 2)).sum(dim=-1) - (mean_ref ** 2)  # (B, T)

    # sum over tokens, normalize by sqrt(var sum)
    num = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1))               # (B,)
    den = torch.sqrt(var_ref.sum(dim=-1).clamp_min(1e-12))                  # (B,)
    disc = num / den                                                        # (B,)
    return disc


# -------- Optional probability mapping (no scipy) --------
def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    z = (x - mu) / sigma
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


def prob_from_two_normals(
    x: np.ndarray, mu0: float, sigma0: float, mu1: float, sigma1: float
) -> np.ndarray:
    """
    Balanced prior: p(D0)=p(D1). Then p(D1|x) = p(x|D1) / (p(x|D0)+p(x|D1))
    """
    p0 = _normal_pdf(x, mu0, sigma0)
    p1 = _normal_pdf(x, mu1, sigma1)
    return p1 / (p0 + p1 + 1e-12)


# -------- Config + Tool --------
@dataclass
class FastDetectGPTConfig:
    # In their naming: sampling_model_name = reference model (logits_ref)
    sampling_model_id: str = "tiiuae/falcon-7b"
    scoring_model_id: str = "tiiuae/falcon-7b-instruct"

    device: Optional[str] = None           # "cuda" | "cpu" | None(auto)
    max_tokens: int = 512
    use_bfloat16: bool = False             # T4 supports fp16 better than bf16
    trust_remote_code: bool = True
    hf_token_env: str = "HF_TOKEN"

    # probability params (optional); if None -> output only criterion
    mu0: Optional[float] = None
    sigma0: Optional[float] = None
    mu1: Optional[float] = None
    sigma1: Optional[float] = None


class FastDetectGPTTool:
    """
    Window-level FastDetectGPT features:
      - criterion_raw: the analytic discrepancy (higher often => more AI-like in their setup)
      - prob_ai (optional): mapped probability using (mu0,sigma0,mu1,sigma1)
      - ntokens: number of tokens evaluated
    """

    def __init__(self, cfg: FastDetectGPTConfig):
        self.cfg = cfg

        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        hf_token = os.environ.get(cfg.hf_token_env) or None
        dtype = torch.float16
        if cfg.use_bfloat16 and torch.cuda.is_available():
            dtype = torch.bfloat16

        # IMPORTANT: FastDetectGPT assumes same tokenization between models.
        # Best practice: use the scoring tokenizer and enforce same input_ids for the sampling model.
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(cfg.scoring_model_id, token=hf_token)
        if self.scoring_tokenizer.pad_token_id is None:
            self.scoring_tokenizer.pad_token_id = self.scoring_tokenizer.eos_token_id

        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            cfg.scoring_model_id,
            torch_dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            token=hf_token,
        ).to(device).eval()

        self.same_model = (cfg.sampling_model_id == cfg.scoring_model_id)
        if self.same_model:
            self.sampling_model = self.scoring_model
            self.sampling_tokenizer = self.scoring_tokenizer
        else:
            self.sampling_tokenizer = AutoTokenizer.from_pretrained(cfg.sampling_model_id, token=hf_token)
            if self.sampling_tokenizer.pad_token_id is None:
                self.sampling_tokenizer.pad_token_id = self.sampling_tokenizer.eos_token_id

            self.sampling_model = AutoModelForCausalLM.from_pretrained(
                cfg.sampling_model_id,
                torch_dtype=dtype,
                trust_remote_code=cfg.trust_remote_code,
                token=hf_token,
            ).to(device).eval()

    def _encode_scoring(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.scoring_tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=self.cfg.max_tokens,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        return {k: v.to(self.device) for k, v in enc.items()} # Returns tensors of input_ids, attention_mask on the correct device

    def _encode_sampling(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.sampling_tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=self.cfg.max_tokens,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        return {k: v.to(self.device) for k, v in enc.items()} # Returns tensors of input_ids, attention_mask 

    @torch.inference_mode()
    def featurize_texts_fdg(
        self,
        texts: List[str],
        batch_size: int = 4,
        show_progress: bool = True,
        progress_desc: str = "FastDetectGPT",
    ) -> List[Dict[str, Any]]:
        if not texts:
            return []
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")

        out: List[Dict[str, Any]] = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=progress_desc, unit="batch")

        for i in iterator:
            batch = texts[i : i + batch_size]

            # scoring side
            enc_score = self._encode_scoring(batch)
            input_ids = enc_score["input_ids"]                      # (B, L)
            attn = enc_score.get("attention_mask", None)            # (B, L)
            labels = input_ids[:, 1:].contiguous()                  # (B, L-1)

            logits_score = self.scoring_model(**enc_score).logits[:, :-1, :]  # (B, L-1, V)

            # sampling side (reference)
            if self.same_model:
                logits_ref = logits_score
                ntokens = labels.size(1)
            else:
                enc_ref = self._encode_sampling(batch)
                # enforce same tokens (this is what their repo asserts)
                if not torch.equal(enc_ref["input_ids"][:, 1:], labels):
                    raise ValueError("Tokenizer mismatch between sampling and scoring models.")
                logits_ref = self.sampling_model(**enc_ref).logits[:, :-1, :]  # (B, L-1, V)
                ntokens = labels.size(1)

            # discrepancy per example (B,)
            disc = sampling_discrepancy_analytic(logits_ref, logits_score, labels)

            disc_np = disc.detach().to("cpu").float().numpy()

            # optional probability mapping-- we use it for 
            prob_np: Optional[np.ndarray] = None
            if None not in (self.cfg.mu0, self.cfg.sigma0, self.cfg.mu1, self.cfg.sigma1):
                prob_np = prob_from_two_normals(
                    disc_np,
                    mu0=float(self.cfg.mu0),
                    sigma0=float(self.cfg.sigma0),
                    mu1=float(self.cfg.mu1),
                    sigma1=float(self.cfg.sigma1),
                )

            for j in range(len(batch)):
                feats: Dict[str, Any] = {
                    "criterion_raw": float(disc_np[j]),
                    "ntokens": int(ntokens),
                    "sampling_model_id": self.cfg.sampling_model_id,
                    "scoring_model_id": self.cfg.scoring_model_id,
                    "max_tokens": int(self.cfg.max_tokens),
                }
                if prob_np is not None:
                    feats["prob_ai"] = float(prob_np[j])

                out.append({"fastdetectgpt_features": feats})

        return out
