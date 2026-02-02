import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np

@dataclass
class DataCollator:
  
    transcriptome_processor: Any
    text_processor: Any
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        
       
        transcriptome_data = []
        text_data = []
        metadata = []
        
        for feature in features:
            transcriptome_data.append(feature["transcriptome"])
            text_data.append(feature["text"])
            if "metadata" in feature:
                metadata.append(feature["metadata"])
        
        transcriptome_inputs = self.transcriptome_processor(transcriptome_data)
        text_inputs = self.text_processor(text_data)

        batch = {
            "transcriptome_inputs": transcriptome_inputs,
            "text_inputs": text_inputs,
        }
        
        if metadata:
            batch["metadata"] = metadata
        
        return batch

@dataclass
class ContrastiveDataCollator(DataCollator):
    use_hard_negatives: bool = False
    negative_sampling_ratio: float = 0.1
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:

        batch = super().__call__(features)

        batch_size = len(features)
        batch["labels"] = torch.arange(batch_size)

        if self.use_hard_negatives:
            self._add_hard_negatives(batch, features)
        
        return batch
    
    def _add_hard_negatives(self, batch: Dict, features: List[Dict]):

        batch_size = len(features)
        n_negatives = int(batch_size * self.negative_sampling_ratio)
        
        if n_negatives > 0:

            negative_indices = torch.randperm(batch_size)[:n_negatives]
            batch["negative_indices"] = negative_indices

@dataclass 
class MultiModalDataCollator:
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        collated_batch = {}

        all_keys = set()
        for feature in features:
            all_keys.update(feature.keys())

        for key in all_keys:
            values = [feature[key] for feature in features if key in feature]
            
            if isinstance(values[0], torch.Tensor):
                try:
                    collated_batch[key] = torch.stack(values)
                except:
                    collated_batch[key] = self._pad_tensors(values)
            
            elif isinstance(values[0], (int, float, np.number)):
                collated_batch[key] = torch.tensor(values)
            
            elif isinstance(values[0], (list, np.ndarray)):
                try:
                    collated_batch[key] = torch.tensor(np.array(values))
                except:
                    collated_batch[key] = values
            
            else:
                collated_batch[key] = values
        
        return collated_batch
    
    def _pad_tensors(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(tensor.shape[0] for tensor in tensors)
        
        padded_tensors = []
        for tensor in tensors:
            padding = max_len - tensor.shape[0]
            if padding > 0:
                pad_shape = list(tensor.shape)
                pad_shape[0] = padding
 
                padding_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
                
                padded = torch.cat([tensor, padding_tensor], dim=0)
            else:
                padded = tensor
            
            padded_tensors.append(padded)
        
        return torch.stack(padded_tensors)