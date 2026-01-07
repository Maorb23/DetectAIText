# ai_detect/tools/logits.py
# Gets text windows -> logits-based features
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm


@dataclass(frozen=True)
class LogitsFeatures:
    mean_logprob: float
    std_logprob: float
    p10_logprob: float
    p50_logprob: float
    p90_logprob: float
    mean_entropy: float
    mean_top1_prob: float
    frac_top1_gt_0p8: float
    mean_top1_margin: float
    frac_true_in_top10: float
    n_tokens: int


class LogitsFeatureExtractor:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-0.6B",  # <-- changed
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_tokens: int = 512,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.max_tokens = max_tokens

        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def featurize_window(self, text: str) -> LogitsFeatures:
        enc = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            padding=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        if input_ids.shape[1] < 3:
            return LogitsFeatures(
                mean_logprob=0.0, std_logprob=0.0, p10_logprob=0.0, p50_logprob=0.0, p90_logprob=0.0,
                mean_entropy=0.0, mean_top1_prob=0.0, frac_top1_gt_0p8=0.0, mean_top1_margin=0.0,
                frac_true_in_top10=0.0, n_tokens=int(input_ids.shape[1]),
            )

        out = self.model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits  # [1, T, V]
        print("Shape of logits:", logits.shape)

        logits = logits[:, :-1, :]                 # [1, T-1, V], for next-token prediction we dont have labels for the last token
        labels = input_ids[:, 1:]                  # [1, T-1]
        mask = attn[:, 1:].bool()                  # [1, T-1]

        log_probs = torch.log_softmax(logits, dim=-1)  # [1, T-1, V]
        true_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
        true_lp = true_lp[mask].float()  # [N]

        p = log_probs.exp()
        ent = -(p * log_probs).sum(dim=-1)  # [1, T-1]
        ent = ent[mask].float()             # [N]

        top2 = torch.topk(logits, k=2, dim=-1).values  # [1, T-1, 2]
        logZ = torch.logsumexp(logits, dim=-1)         # [1, T-1]
        top1 = top2[..., 0]
        top1_prob = torch.exp((top1 - logZ))[mask].float()  # [N]
        top1_margin = (top2[..., 0] - top2[..., 1])[mask].float()  # [N]

        top10_idx = torch.topk(logits, k=10, dim=-1).indices  # [1, T-1, 10]
        in_top10 = (top10_idx == labels.unsqueeze(-1)).any(dim=-1)[mask].float()  # [N]

        return _summarize(true_lp, ent, top1_prob, top1_margin, in_top10)

    def featurize_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        
        rows: List[Dict[str, Any]] = []

        for ex in tqdm(examples, desc="Logits features", unit="window"):
            text = ex.get("text", "") or ""
            meta = dict(ex.get("meta", {}))
            label = ex.get("label", None)

            t0 = time.perf_counter()
            feats = self.featurize_window(text)
            dt = time.perf_counter() - t0

            rows.append({"label": label, "meta": meta, "features": feats.__dict__, "runtime_s": dt})

        return rows


def _summarize(
    true_lp: torch.Tensor,
    ent: torch.Tensor,
    top1_prob: torch.Tensor,
    top1_margin: torch.Tensor,
    in_top10: torch.Tensor,
) -> LogitsFeatures:
    def q(x: torch.Tensor, p: float) -> float:
        return float(torch.quantile(x, torch.tensor(p, device=x.device)).item())

    mean_lp = float(true_lp.mean().item())
    std_lp = float(true_lp.std(unbiased=False).item()) if true_lp.numel() > 1 else 0.0

    return LogitsFeatures(
        mean_logprob=mean_lp,
        std_logprob=std_lp,
        p10_logprob=q(true_lp, 0.10),
        p50_logprob=q(true_lp, 0.50),
        p90_logprob=q(true_lp, 0.90),
        mean_entropy=float(ent.mean().item()),
        mean_top1_prob=float(top1_prob.mean().item()),
        frac_top1_gt_0p8=float((top1_prob > 0.8).float().mean().item()),
        mean_top1_margin=float(top1_margin.mean().item()),
        frac_true_in_top10=float(in_top10.mean().item()),
        n_tokens=int(true_lp.numel() + 1),
    )
