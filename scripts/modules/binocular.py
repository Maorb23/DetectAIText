# ai_detect/tools/binoculars.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.tools.binoculars_metrics import perplexity_mean, cross_entropy_p_q_mean
from scripts.utils.utils_tokenizer import assert_tokenizer_consistency  # or keep local


BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527


@dataclass
class BinocularsConfig:
    observer_model_id: str = "tiiuae/falcon-7b"
    performer_model_id: str = "tiiuae/falcon-7b-instruct"
    max_tokens: int = 512
    use_bfloat16: bool = True
    mode: str = "low-fpr"  # "low-fpr" | "accuracy"
    device_1: Optional[str] = None
    device_2: Optional[str] = None
    hf_token_env: str = "HF_TOKEN"
    trust_remote_code: bool = True


class BinocularsTool:
    """
    Produces window-level features + a raw binoculars score.
    Calibration (score->prob) should happen later in scorers.py / calibrate.py.
    """

    def __init__(self, cfg: BinocularsConfig):
        self.cfg = cfg
        assert_tokenizer_consistency(cfg.observer_model_id, cfg.performer_model_id)

        self.threshold = BINOCULARS_FPR_THRESHOLD if cfg.mode == "low-fpr" else BINOCULARS_ACCURACY_THRESHOLD

        hf_token = os.environ.get(cfg.hf_token_env) or None
        dtype = torch.bfloat16 if (cfg.use_bfloat16 and torch.cuda.is_available()) else torch.float32

        device_1 = cfg.device_1 or ("cuda:0" if torch.cuda.is_available() else "cpu")
        device_2 = cfg.device_2 or ("cuda:1" if (torch.cuda.device_count() > 1) else device_1)

        self.device_1 = device_1
        self.device_2 = device_2

        self.observer_model = AutoModelForCausalLM.from_pretrained(
            cfg.observer_model_id,
            device_map={"": device_1},
            torch_dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            token=hf_token,
        ).eval()

        self.performer_model = AutoModelForCausalLM.from_pretrained(
            cfg.performer_model_id,
            device_map={"": device_2},
            torch_dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            token=hf_token,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.observer_model_id, token=hf_token)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _tokenize(self, texts: List[str]):
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest" if len(texts) > 1 else False,
            truncation=True,
            max_length=self.cfg.max_tokens,
            return_token_type_ids=False,
        )
        # keep input_ids on CPU; we move per-model call to their device
        return enc

    @torch.inference_mode()
    def _get_logits(self, enc):
        obs = self.observer_model(**{k: v.to(self.device_1) for k, v in enc.items()}).logits
        perf = self.performer_model(**{k: v.to(self.device_2) for k, v in enc.items()}).logits
        if self.device_1.startswith("cuda") or self.device_2.startswith("cuda"):
            torch.cuda.synchronize()
        return obs, perf

    def featurize_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        enc = self._tokenize(texts)
        obs_logits, perf_logits = self._get_logits(enc)

        # compute in numpy for stability
        ppl_nll = perplexity_mean(enc.to(self.device_2), perf_logits)  # performer side
        xent = cross_entropy_p_q_mean(
            p_logits=obs_logits.to(self.device_1),
            q_logits=perf_logits.to(self.device_1),
            encoding=enc.to(self.device_1),
            pad_token_id=self.tokenizer.pad_token_id,
        )

        score = (ppl_nll / (xent + 1e-8)).astype(np.float32)

        out: List[Dict[str, Any]] = []
        for i in range(len(texts)):
            out.append({
                "binoculars_features": {
                    "ppl_nll_mean": float(ppl_nll[i]),
                    "xent_mean": float(xent[i]),
                    "score_raw": float(score[i]),
                    "threshold": float(self.threshold),
                    "mode": self.cfg.mode,
                    "observer_model_id": self.cfg.observer_model_id,
                    "performer_model_id": self.cfg.performer_model_id,
                }
            })
        return out
