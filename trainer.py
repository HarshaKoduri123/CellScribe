import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import wandb
import os
from datetime import datetime
import re

from models.multimodal_model import CellWhispererModel, CellWhispererConfig
from loss.contrastive import SymmetricContrastiveLoss
from utils.metrics import RetrievalMetrics

logger = logging.getLogger(__name__)

class CellWhispererTrainer:

    
    def __init__(
        self,
        model: CellWhispererModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        device: str = "cuda",
    ):

        if isinstance(device, str):
            if device == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif device.startswith("cuda:") and torch.cuda.is_available():
                self.device = torch.device(device)
            else:
                self.device = torch.device("cpu")
                logger.warning(f"CUDA not available or specified, using {self.device}")
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.config = self._sanitize_config(config or {})
        
        self.loss_fn = SymmetricContrastiveLoss(
            temperature=self.config.get("temperature", 0.07)
        )
        

        self.optimizer = self._create_optimizer()

        self.scheduler = self._create_scheduler()
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.save_dir = self.config.get("save_dir", "./checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        if self.config.get("use_wandb", False):
            wandb.init(
                project=self.config.get("wandb_project", "cellwhisperer"),
                config=self.config,
            )
    
    def _sanitize_config(self, config: Dict) -> Dict:
        """Convert string values to appropriate types in config"""
        def convert_value(value):
            if isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(v) for v in value]
            elif isinstance(value, str):
                try:
                    if re.match(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$', value):
                        return float(value)
                    elif value.isdigit() or (value[0] == '-' and value[1:].isdigit()):
                        return int(value)
                    elif value.lower() in ('true', 'false'):
                        return value.lower() == 'true'
                    else:
                        return value
                except (ValueError, AttributeError):
                    return value
            else:
                return value
        
        return convert_value(config)
    
    def _create_optimizer(self) -> optim.Optimizer:
        transcriptome_params = []
        text_params = []
        adapter_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "transcriptome_encoder" in name:
                    transcriptome_params.append(param)
                elif "text_encoder" in name:
                    text_params.append(param)
                elif "adapter" in name:
                    adapter_params.append(param)
        learning_rate = float(self.config.get("learning_rate", 1e-5))
        text_lr = float(self.config.get("text_lr", 1e-5))
        transcriptome_lr = float(self.config.get("transcriptome_lr", 1e-6))
        weight_decay = float(self.config.get("weight_decay", 0.01))
        
        logger.debug(f"Learning rates - adapter: {learning_rate}, text: {text_lr}, transcriptome: {transcriptome_lr}")

        optimizer = optim.AdamW([
            {"params": adapter_params, "lr": learning_rate},
            {"params": text_params, "lr": text_lr},
            {"params": transcriptome_params, "lr": transcriptome_lr},
        ], weight_decay=weight_decay)
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        num_epochs = int(self.config.get("num_epochs", 16))
        total_steps = len(self.train_loader) * num_epochs
        
        # Handle both warmup_epochs and warmup_ratio
        warmup_epochs = self.config.get("warmup_epochs")
        warmup_ratio = self.config.get("warmup_ratio")
        
        if warmup_epochs is not None:
            warmup_epochs = float(warmup_epochs)
            warmup_steps = int((warmup_epochs / num_epochs) * total_steps)
        elif warmup_ratio is not None:
            warmup_ratio = float(warmup_ratio)
            warmup_steps = int(warmup_ratio * total_steps)
        else:
            warmup_steps = int(0.03 * total_steps)
        
        max_lr = float(self.config.get("learning_rate", 1e-5))
        
        logger.debug(f"Scheduler - total_steps: {total_steps}, warmup_steps: {warmup_steps}, max_lr: {max_lr}")
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
        )
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = self._move_batch_to_device(batch)
            
            outputs = self.model(
                transcriptome_inputs=batch.transcriptome_inputs,
                text_inputs=batch.text_inputs,
            )
            
            similarity_matrix = outputs["similarity_matrix"]
            logit_scale = outputs["logit_scale"]
            
            loss = self.loss_fn(
                similarity_matrix * logit_scale,
                similarity_matrix.t() * logit_scale,
            )

            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("max_grad_norm", 1.0)
            )
            
            self.optimizer.step()
            self.scheduler.step()

            if "expression_tokens" in batch.transcriptome_inputs:
                batch_size = batch.transcriptome_inputs["expression_tokens"].size(0)
            elif "input_ids" in batch.transcriptome_inputs:
                batch_size = batch.transcriptome_inputs["input_ids"].size(0)
            else:
                for key, value in batch.transcriptome_inputs.items():
                    if torch.is_tensor(value):
                        batch_size = value.size(0)
                        break
                else:
                    batch_size = 0
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            self.global_step += 1

            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            progress_bar.set_postfix({"loss": avg_loss})

            if self.config.get("use_wandb", False) and batch_idx % 10 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "logit_scale": logit_scale.item(),
                    "step": self.global_step,
                })
        
        return {"train_loss": total_loss / total_samples if total_samples > 0 else 0}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        total_samples = 0

        all_transcriptome_embeds = []
        all_text_embeds = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = self._move_batch_to_device(batch)
                

                outputs = self.model(
                    transcriptome_inputs=batch.transcriptome_inputs,
                    text_inputs=batch.text_inputs,
                )
                
                similarity_matrix = outputs["similarity_matrix"]
                logit_scale = outputs["logit_scale"]
                
                loss = self.loss_fn(
                    similarity_matrix * logit_scale,
                    similarity_matrix.t() * logit_scale,
                )

                all_transcriptome_embeds.append(outputs["transcriptome_embeddings"].cpu())
                all_text_embeds.append(outputs["text_embeddings"].cpu())

                if "expression_tokens" in batch.transcriptome_inputs:
                    batch_size = batch.transcriptome_inputs["expression_tokens"].size(0)
                elif "input_ids" in batch.transcriptome_inputs:
                    batch_size = batch.transcriptome_inputs["input_ids"].size(0)
                else:
                    for key, value in batch.transcriptome_inputs.items():
                        if torch.is_tensor(value):
                            batch_size = value.size(0)
                            break
                    else:
                        batch_size = 0
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        transcriptome_embeds = torch.cat(all_transcriptome_embeds, dim=0)
        text_embeds = torch.cat(all_text_embeds, dim=0)

        metrics = RetrievalMetrics.compute_all_metrics(
            transcriptome_embeds.numpy(),
            text_embeds.numpy(),
        )
        
        metrics["val_loss"] = total_loss / total_samples if total_samples > 0 else 0

        if self.config.get("use_wandb", False):
            wandb.log(metrics)
        
        return metrics
    
    def train(self, num_epochs: int):
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch

            train_metrics = self.train_epoch()
            
            val_metrics = self.validate()
            
            # Print epoch summary
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            if val_metrics:
                logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                for metric_name, metric_value in val_metrics.items():
                    if metric_name != "val_loss":
                        logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(val_metrics.get("val_loss", train_metrics["train_loss"]))
    
    def save_checkpoint(self, val_loss: float):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.save_dir,
            f"checkpoint_epoch{self.epoch}_step{self.global_step}.pt"
        )
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "val_loss": val_loss,
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path} with val_loss {val_loss:.4f}")
    
    def _move_batch_to_device(self, batch) -> Any:
        """Move batch tensors to device - handle Geneformer inputs properly"""
        def move_to_device(obj):
            if torch.is_tensor(obj):
                return obj.to(self.device)
            elif isinstance(obj, dict):
                # Recursively move dict values to device
                return {k: move_to_device(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(move_to_device(v) for v in obj)
            elif hasattr(obj, 'to'):
                try:
                    return obj.to(self.device)
                except Exception as e:
                    logger.debug(f"Object .to() method failed: {e}")
                    if hasattr(obj, 'data'):
                        obj.data = move_to_device(obj.data)
                    return obj
            else:
                return obj
        
        if hasattr(batch, '__dict__'):
            moved_dict = {}
            for key, value in batch.__dict__.items():
                moved_dict[key] = move_to_device(value)
            from data.dataset import CellWhispererBatch
            return CellWhispererBatch(**moved_dict)
        else:
            return move_to_device(batch)