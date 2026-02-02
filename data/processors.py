import torch
import numpy as np
import anndata
from transformers import AutoTokenizer
from typing import List, Dict, Any

from .geneformer_processor import GeneformerProcessor

class TranscriptomeProcessor:


    def __init__(self, processor_type="geneformer", **kwargs):
        if processor_type == "geneformer":
            self.processor = GeneformerProcessor(**kwargs)
        else:
            raise ValueError("Only geneformer supported for now")

    def __call__(self, data, **kwargs):
        return self.processor(data, **kwargs)

    def fit(self, expression_matrix, gene_names):
        return self.processor.fit(expression_matrix, gene_names)

class TextProcessor:


    def __init__(self, model_name="dmis-lab/biobert-v1.1", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

    def __call__(self, texts: List[str], return_tensors="pt", **kwargs) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )

    def decode(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
