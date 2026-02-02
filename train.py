#!/usr/bin/env python

import torch
import numpy as np
import yaml
import logging
from pathlib import Path
import argparse
import sys
import json
from scipy import sparse
sys.path.append(str(Path(__file__).parent))

from models.multimodal_model import CellWhispererModel, CellWhispererConfig
from data.dataset import PairedDataset, CellWhispererDataLoader
from trainer import CellWhispererTrainer
from data.processors import TranscriptomeProcessor, TextProcessor

def setup_logging(log_level: str = "INFO"):

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler(),
        ]
    )

def load_config(config_path: str) -> dict:

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



def load_split(split_dir, prefix):
    X = sparse.load_npz(split_dir / f"{prefix}_X.npz")

    with open(split_dir / f"{prefix}_text.json") as f:
        text = json.load(f)

    with open(split_dir / f"{prefix}_genes.json") as f:
        gene_names = json.load(f)

    return X, text, gene_names



def load_dataset(config: dict):
    root = Path(config["dataset"]["root_dir"])
    splits = config["dataset"]["splits"]

    train_X, train_text, gene_names = load_split(
        root / splits["train"], "train"
    )

    val_X, val_text, _ = load_split(
        root / splits["val"], "val"
    )

    train_dataset = PairedDataset(
        transcriptome_data=train_X,
        text_data=train_text,
        gene_names=gene_names,
    )

    val_dataset = PairedDataset(
        transcriptome_data=val_X,
        text_data=val_text,
        gene_names=gene_names,
    )

    return train_dataset, val_dataset


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train CellWhisperer model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    setup_logging("DEBUG" if args.debug else "INFO")
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    config["training"]["device"] = args.device
    config["logging"]["save_dir"] = args.output_dir

    logger.info("Loading dataset...")
    train_dataset, val_dataset = load_dataset(config)
    
    transcriptome_processor = TranscriptomeProcessor(
        processor_type="geneformer", 
        max_genes=config["data"]["transcriptome"]["max_genes"],
        normalize=config["data"]["transcriptome"]["normalize"],
        log_transform=config["data"]["transcriptome"]["log_transform"],
        target_sum=10000,  
        nproc=config["dataset"].get("nproc", 4),
    )
    
    transcriptome_processor.fit(
        train_dataset.transcriptome_data,
        train_dataset.gene_names if hasattr(train_dataset, 'gene_names') else None
    )
    
    text_processor = TextProcessor(
        model_name=config["model"]["text_backbone"],
        max_length=config["data"]["text"]["max_length"],
    )
    
    train_loader = CellWhispererDataLoader(
        dataset=train_dataset,
        transcriptome_processor=transcriptome_processor,
        text_tokenizer=text_processor,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
    )
    
    val_loader = CellWhispererDataLoader(
        dataset=val_dataset,
        transcriptome_processor=transcriptome_processor,
        text_tokenizer=text_processor,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )

    logger.info("Initializing model...")
    model_config = CellWhispererConfig(
        transcriptome_backbone="geneformer",
        text_backbone=config["model"]["text_backbone"],
        projection_dim=config["model"]["projection_dim"],
        locking_mode=config["model"]["locking_mode"],
        dropout=config["model"]["dropout"],
        transcriptome_dim=512,  
        text_dim=768, 
    )
    
    model = CellWhispererModel(model_config)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    trainer = CellWhispererTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            **config["training"],
            **config["logging"],
            "use_wandb": config["logging"]["wandb_project"] is not None,
        },
        device=args.device,
    )

    logger.info("Starting training...")
    trainer.train(num_epochs=config["training"]["num_epochs"])
    
    logger.info("Training completed!")
    
    final_path = Path(args.output_dir) / "final_model.pt"
    model.save_pretrained(str(final_path))
    logger.info(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()