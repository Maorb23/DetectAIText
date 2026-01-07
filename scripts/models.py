# models.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class CriticModel:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    def tokenize(self, texts, max_length=1024):
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def get_logits(self, batch):
        with torch.no_grad():
            out = self.model(**batch)
        return out.logits  # [batch, seq, vocab]
