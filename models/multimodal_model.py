# models/multimodal_model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Tuple, Dict, Any, Union
import logging


from .geneformer_model import GeneformerModel, GeneformerConfig
from .adapters import DualAdapter

logger = logging.getLogger(__name__)

class CellWhispererConfig:

    
    def __init__(
        self,
        transcriptome_backbone: str = "geneformer",
        text_backbone: str = "dmis-lab/biobert-v1.1",
        projection_dim: int = 2048,
        locking_mode: str = "LU", 
        transcriptome_dim: Optional[int] = None,
        text_dim: Optional[int] = None,
        dropout: float = 0.1,
        batch_norm: bool = True,
        **kwargs,
    ):
        self.transcriptome_backbone = transcriptome_backbone
        self.text_backbone = text_backbone
        self.projection_dim = projection_dim
        self.locking_mode = locking_mode
        self.dropout = dropout
        self.batch_norm = batch_norm
        

        if transcriptome_dim is None:

            transcriptome_dim = 512
        
        if text_dim is None:
            if "biobert" in text_backbone.lower():
                text_dim = 768
            elif "biogpt" in text_backbone.lower():
                text_dim = 1024
            else:
                text_dim = 768 
        
        self.transcriptome_dim = transcriptome_dim
        self.text_dim = text_dim

        self.__dict__.update(kwargs)

class CellWhispererModel(nn.Module):

    
    def __init__(self, config: CellWhispererConfig):
        super().__init__()
        self.config = config
        
        if config.transcriptome_backbone == "geneformer":
            self.transcriptome_encoder = self._init_geneformer(config)
        else:
            raise ValueError(f"Only 'geneformer' backbone supported. Got: {config.transcriptome_backbone}")

        self.text_encoder, self.tokenizer = self._init_text_backbone(config)

        self._apply_locking_mode(config.locking_mode)
  
        self.adapter = DualAdapter(
            transcriptome_dim=config.transcriptome_dim,
            text_dim=config.text_dim,
            projection_dim=config.projection_dim,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
        )
    
    def _init_geneformer(self, config: CellWhispererConfig) -> GeneformerModel:
        """Initialize Geneformer model"""

        geneformer_config = GeneformerConfig(
            hidden_size=config.transcriptome_dim,
            emb_mode="cell", 
            emb_layer=-1, 
            forward_batch_size=32, 
            
        )

        model_path = "ctheodoris/Geneformer"
        if model_path:
            geneformer = GeneformerModel.from_pretrained(
                model_path,
                config=geneformer_config,
                ignore_mismatched_sizes=True  
            )

            
        
        return geneformer
    
    def _init_text_backbone(self, config: CellWhispererConfig) -> Tuple[nn.Module, Any]:

        try:
            model = AutoModel.from_pretrained(config.text_backbone)
            tokenizer = AutoTokenizer.from_pretrained(config.text_backbone)
            

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
            
            return model, tokenizer
        
        except Exception as e:
            logger.error(f"Failed to load text backbone {config.text_backbone}: {e}")
            raise
    
    def _apply_locking_mode(self, locking_mode: str):

        if len(locking_mode) != 2:
            raise ValueError("Locking mode must be 2 characters (e.g., 'LU')")

        if locking_mode[0] == "L":  # Locked
            for param in self.transcriptome_encoder.parameters():
                param.requires_grad = False
            logger.info("Geneformer model locked (frozen)")
        elif locking_mode[0] == "U":  # Unlocked
            for param in self.transcriptome_encoder.parameters():
                param.requires_grad = True
            logger.info("Geneformer model unlocked (trainable)")
        else:
            raise ValueError(f"Unknown locking mode for transcriptome: {locking_mode[0]}")

        if locking_mode[1] == "L":  # Locked
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            logger.info("Text model locked (frozen)")
        elif locking_mode[1] == "U":  # Unlocked
            for param in self.text_encoder.parameters():
                param.requires_grad = True
            logger.info("Text model unlocked (trainable)")
        elif locking_mode[1] == "T":  
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            if hasattr(self.text_encoder, 'encoder'):

                for param in self.text_encoder.encoder.layer[-2:].parameters():
                    param.requires_grad = True
            elif hasattr(self.text_encoder, 'layers'):
                for param in self.text_encoder.layers[-2:].parameters():
                    param.requires_grad = True
            
            logger.info("Text model partially tuned (last 2 layers trainable)")
        else:
            raise ValueError(f"Unknown locking mode for text: {locking_mode[1]}")
    
    def encode_transcriptome(
        self,
        expression_tokens: torch.Tensor,
        expression_token_lengths: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:



        if (expression_tokens >= self.transcriptome_encoder.config.vocab_size).any():
            expression_tokens = torch.clamp(expression_tokens, 0, self.transcriptome_encoder.config.vocab_size - 1)
        transcriptome_features = self.transcriptome_encoder(
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            return_dict=False,
        )

        if isinstance(transcriptome_features, tuple) and len(transcriptome_features) == 2:
            transcriptome_features = transcriptome_features[1]
        
        return transcriptome_features
    
    def encode_text(
        self,
        texts: Union[str, list],
        max_length: int = 128,
        **kwargs,
    ) -> torch.Tensor:

        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            **kwargs,
        )
        
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.text_encoder(**inputs)

        if hasattr(outputs, 'pooler_output'):
            text_features = outputs.pooler_output
        else:
            text_features = outputs.last_hidden_state.mean(dim=1)
        
        return text_features
    
    def forward(
        self,
        transcriptome_inputs: Dict[str, torch.Tensor],
        text_inputs: Dict[str, torch.Tensor],
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:

        transcriptome_features = self.encode_transcriptome(**transcriptome_inputs)

        if "texts" in text_inputs:
            text_features = self.encode_text(
                text_inputs["texts"],
                max_length=text_inputs.get("max_length", 128),
            )
        else:
            device = next(self.parameters()).device
            text_inputs_device = {}
            for key, value in text_inputs.items():
                if torch.is_tensor(value):
                    text_inputs_device[key] = value.to(device)
                else:
                    text_inputs_device[key] = value
            
            outputs = self.text_encoder(**text_inputs_device)
            if hasattr(outputs, 'pooler_output'):
                text_features = outputs.pooler_output
            else:
                text_features = outputs.last_hidden_state.mean(dim=1)

        transcriptome_embeds, text_embeds, logit_scale = self.adapter(
            transcriptome_features,
            text_features,
            normalize=normalize,
        )

        if normalize:
            similarity_matrix = logit_scale * transcriptome_embeds @ text_embeds.t()
        else:
            similarity_matrix = transcriptome_embeds @ text_embeds.t()
        
        return {
            "transcriptome_embeddings": transcriptome_embeds,
            "text_embeddings": text_embeds,
            "similarity_matrix": similarity_matrix,
            "logit_scale": logit_scale,
            "transcriptome_features": transcriptome_features,
            "text_features": text_features,
        }
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[CellWhispererConfig] = None,
        **kwargs,
    ):

        if config is None:

            checkpoint = torch.load(model_path, map_location="cpu")
            if "config" in checkpoint:
                config_dict = checkpoint["config"]
                config = CellWhispererConfig(**config_dict)
            else:
                raise ValueError("No config found in checkpoint")
        
        model = cls(config)

        checkpoint = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        logger.info(f"Loaded pretrained CellWhisperer from {model_path}")
        return model
    
    def save_pretrained(self, save_path: str):
        """Save model and config"""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.__dict__,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Saved model to {save_path}")