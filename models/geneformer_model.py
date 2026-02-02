# models/geneformer_model.py


import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, Union
from transformers import BertForMaskedLM, BertConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.configuration_utils import PretrainedConfig
import os

try:
    from geneformer.in_silico_perturber import quant_layers
    from geneformer.emb_extractor import get_embs
    HAS_GENEFORMER_DEPENDENCIES = True
except ImportError:
    HAS_GENEFORMER_DEPENDENCIES = False
    logger = logging.getLogger(__name__)
    logger.warning("Geneformer dependencies not found. Using simplified implementation.")

# Constants from your code
PAD_TOKEN_ID = 0
MODEL_INPUT_SIZE = 2048

class GeneformerConfig(PretrainedConfig):

    model_type = "geneformer"
    
    def __init__(
        self,
        hidden_size: int = 512,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 4,
        intermediate_size: int = 1024,
        hidden_act: str = "relu",
        hidden_dropout_prob: float = 0.02,
        attention_probs_dropout_prob: float = 0.02,
        max_position_embeddings: int = 2**11,  
        initializer_range: float = 0.2,
        layer_norm_eps: float = 1e-12,
        vocab_size: int = 25426, 
        
        emb_mode: str = "cell",
        emb_layer: int = -1,
        forward_batch_size: int = -1,
        summary_stat: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.vocab_size = vocab_size

        self.emb_mode = emb_mode
        self.emb_layer = emb_layer
        self.forward_batch_size = forward_batch_size
        self.summary_stat = summary_stat

        self.pad_token_id = PAD_TOKEN_ID

class GeneformerModel(PreTrainedModel):

    
    config_class = GeneformerConfig
    base_model_prefix = "geneformer"
    main_input_name = "expression_tokens"
    
    def __init__(self, config: GeneformerConfig):
        super().__init__(config)
        
        # Create BERT configuration
        bert_config = BertConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            vocab_size=config.vocab_size,
            pad_token_id=config.pad_token_id,
            output_hidden_states=True,
            output_attentions=False,
        )

        self.bert = BertForMaskedLM(bert_config)

        self.post_init()

        if not HAS_GENEFORMER_DEPENDENCIES:
            logger.warning("Geneformer dependencies not available. Using simplified embedding extraction.")
    
    def forward(
        self,
        expression_tokens: torch.Tensor,
        expression_token_lengths: torch.Tensor,
        expression_gene: Optional[torch.Tensor] = None,
        expression_expr: Optional[torch.Tensor] = None,
        expression_key_padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, torch.Tensor]:


        if HAS_GENEFORMER_DEPENDENCIES:
            return self._forward_geneformer(
                expression_tokens,
                expression_token_lengths,
                return_dict
            )
        else:

            return self._forward_simple(
                expression_tokens,
                expression_token_lengths,
                return_dict
            )
    
    def _forward_geneformer(
        self,
        expression_tokens: torch.Tensor,
        expression_token_lengths: torch.Tensor,
        return_dict: Optional[bool]
    ) -> torch.Tensor:

        layer_to_quant = quant_layers(self.bert) + self.config.emb_layer
        
        embs = get_embs(
            self.bert,
            expression_tokens,
            expression_token_lengths,
            self.config.emb_mode,
            layer_to_quant,
            self.config.pad_token_id,
            self.config.forward_batch_size,
            self.config.summary_stat,
        )
        

        return (None, embs) if return_dict is False else embs
    
    def _forward_simple(
        self,
        expression_tokens: torch.Tensor,
        expression_token_lengths: torch.Tensor,
        return_dict: Optional[bool]
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        outputs = self.bert(
            input_ids=expression_tokens,
            attention_mask=self._create_attention_mask(expression_tokens),
            output_hidden_states=True,
            return_dict=True,
        )
        
        if self.config.emb_layer == -1:
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs.hidden_states[self.config.emb_layer]

        attention_mask = self._create_attention_mask(expression_tokens)
        if self.config.emb_mode == "cell":
     
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        else:
            embeddings = hidden_states
        
        if return_dict:
            return BaseModelOutputWithPooling(
                last_hidden_state=hidden_states,
                pooler_output=embeddings,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return outputs.logits, embeddings
    
    def _create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input_ids"""
        return (input_ids != self.config.pad_token_id).long()
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Optional[GeneformerConfig] = None,
        **kwargs
    ) -> 'GeneformerModel':

        if config is None:
            try:
                checkpoint = torch.load(pretrained_model_name_or_path, map_location="cpu")
                if "config" in checkpoint:
                    config_dict = checkpoint["config"]
                    config = GeneformerConfig(**config_dict)
                else:
                    config = GeneformerConfig()
                    logger.warning("No config found in checkpoint, using default config")
            except:
                config = GeneformerConfig()
        
        model = cls(config)

        try:
            if pretrained_model_name_or_path.endswith(".pt") or pretrained_model_name_or_path.endswith(".pth"):
                checkpoint = torch.load(pretrained_model_name_or_path, map_location="cpu")
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.bert = BertForMaskedLM.from_pretrained(
                    pretrained_model_name_or_path,
                    config=model.bert.config,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise
        
        return model

    def save_pretrained(self, save_directory: str):

        
        os.makedirs(save_directory, exist_ok=True)
        
        self.config.save_pretrained(save_directory)
        torch.save(
            self.state_dict(),
            os.path.join(save_directory, "pytorch_model.bin")
        )