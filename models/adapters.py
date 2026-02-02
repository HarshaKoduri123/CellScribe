# models/adapters.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdapterLayer(nn.Module):

    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        
        if batch_norm:
            
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.norm = None
        
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
   
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.fc1(x)
        x = self.activation(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class DualAdapter(nn.Module):

    
    def __init__(
        self,
        transcriptome_dim: int,
        text_dim: int,
        projection_dim: int = 2048,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        super().__init__()
        

        self.transcriptome_adapter = AdapterLayer(
            input_dim=transcriptome_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        

        self.text_adapter = AdapterLayer(
            input_dim=text_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        
    def forward(
        self,
        transcriptome_features: torch.Tensor,
        text_features: torch.Tensor,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        transcriptome_embeds = self.transcriptome_adapter(transcriptome_features)
        text_embeds = self.text_adapter(text_features)

        if normalize:
            transcriptome_embeds = F.normalize(transcriptome_embeds, p=2, dim=-1)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        logit_scale = self.logit_scale.exp()
        
        return transcriptome_embeds, text_embeds, logit_scale