import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass  


logger = logging.getLogger(__name__)

@dataclass
class CellWhispererBatch:

    transcriptome_inputs: Dict[str, torch.Tensor]
    text_inputs: Dict[str, torch.Tensor]
    labels: Optional[torch.Tensor] = None
    metadata: Optional[List[Dict]] = None

class PairedDataset(Dataset):


    def __init__(
        self,
        transcriptome_data: Any, 
        text_data: List[str],
        gene_names: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
        transform=None,
    ):
      
        if hasattr(transcriptome_data, "shape"):
            n_samples = transcriptome_data.shape[0]
        else:
            n_samples = len(transcriptome_data)

        assert n_samples == len(text_data), \
            f"Mismatch: {n_samples} transcriptomes vs {len(text_data)} texts"

        self.transcriptome_data = transcriptome_data
        self.text_data = text_data
        self.gene_names = gene_names
        self.metadata = metadata
        self.transform = transform

        logger.info(f"Loaded dataset with {n_samples} samples")

    def __len__(self):
        if hasattr(self.transcriptome_data, "shape"):
            return self.transcriptome_data.shape[0]
        return len(self.transcriptome_data)

    def __getitem__(self, idx):

        x = self.transcriptome_data[idx]
        if "toarray" in dir(x):  
            x = x.toarray().squeeze()

        sample = {
            "transcriptome": x,
            "text": self.text_data[idx],
        }

        if self.gene_names is not None:
            sample["gene_names"] = self.gene_names
        if self.metadata is not None:
            sample["metadata"] = self.metadata[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample

class CellWhispererDataLoader:

    def __init__(
        self,
        dataset: Dataset,
        transcriptome_processor,
        text_tokenizer,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.transcriptome_processor = transcriptome_processor
        self.text_tokenizer = text_tokenizer
        self.batch_size = batch_size

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch: List[Dict]) -> CellWhispererBatch:

        transcriptome_data = np.array([sample["transcriptome"] for sample in batch])
        transcriptome_inputs = self.transcriptome_processor(
            transcriptome_data, return_tensors="pt"
        )

        text_data = [sample["text"] for sample in batch]
        text_inputs = self.text_tokenizer(
            text_data, return_tensors="pt", padding=True, truncation=True, max_length=128
        )


        metadata = [sample.get("metadata") for sample in batch] if "metadata" in batch[0] else None

        return CellWhispererBatch(
            transcriptome_inputs=transcriptome_inputs,
            text_inputs=text_inputs,
            metadata=metadata,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)
