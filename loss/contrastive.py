# loss/contrastive.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class InfoNCELoss(nn.Module):
    """InfoNCE (NT-Xent) contrastive loss"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(
        self,
        similarity_matrix: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size = similarity_matrix.size(0)
        
        if labels is None:
            # Self-supervised: positive pairs are diagonal
            labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # Compute loss for both directions
        loss_i = self.cross_entropy(similarity_matrix / self.temperature, labels)
        loss_t = self.cross_entropy(similarity_matrix.t() / self.temperature, labels)
        
        return (loss_i + loss_t) / 2

class SymmetricContrastiveLoss(nn.Module):
    """Symmetric contrastive loss as used in CLIP"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(
        self,
        logits_per_transcriptome: torch.Tensor,
        logits_per_text: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = logits_per_transcriptome.size(0)
        labels = torch.arange(batch_size, device=logits_per_transcriptome.device)
        
        # Compute losses
        loss_i = self.cross_entropy(logits_per_transcriptome / self.temperature, labels)
        loss_t = self.cross_entropy(logits_per_text / self.temperature, labels)
        
        return (loss_i + loss_t) / 2

class HardNegativeMiningLoss(nn.Module):
    """Contrastive loss with hard negative mining"""
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss with hard negative mining
        
        Args:
            embeddings_a: First set of embeddings [batch_size, dim]
            embeddings_b: Second set of embeddings [batch_size, dim]
            labels: Optional labels for positive pairs
            
        Returns:
            Loss value
        """
        batch_size = embeddings_a.size(0)
        
        if labels is None:
            labels = torch.arange(batch_size, device=embeddings_a.device)
        
        # Compute similarity matrix
        similarity_matrix = embeddings_a @ embeddings_b.t() / self.temperature
        
        # Positive similarities (diagonal)
        positive_similarities = similarity_matrix.diag()
        
        # Hard negatives: max similarity among negatives
        mask = torch.eye(batch_size, device=embeddings_a.device).bool()
        negative_similarities = similarity_matrix.masked_fill(mask, -float('inf'))
        hard_negatives = negative_similarities.max(dim=1).values
        
        # Compute triplet-like loss
        loss = F.relu(hard_negatives - positive_similarities + self.margin)
        
        return loss.mean()