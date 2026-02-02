#!/usr/bin/env python
"""Testing script for CellWhisperer with UMAP visualization"""

import torch
import numpy as np
import yaml
import logging
from pathlib import Path
import argparse
import sys
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent))

from models.multimodal_model import CellWhispererModel, CellWhispererConfig
from data.dataset import PairedDataset
from data.processors import TranscriptomeProcessor, TextProcessor
from utils.metrics import RetrievalMetrics

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(),
        ]
    )

def load_model_from_checkpoint(checkpoint_path, config, device="cuda"):
    """Load model from checkpoint file"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model config from checkpoint or use provided config
    if "config" in checkpoint:
        model_config_dict = checkpoint["config"]
        # Update with any provided config values
        if config:
            model_config_dict.update(config.get("model", {}))
        model_config = CellWhispererConfig(**model_config_dict)
    elif config:
        model_config = CellWhispererConfig(**config.get("model", {}))
    else:
        raise ValueError("No model config found in checkpoint or provided config")
    
    # Create model
    model = CellWhispererModel(model_config)
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

def create_test_data(config):
    """Create test data if no real data is available"""
    logger = logging.getLogger(__name__)
    
    n_samples = 200  # More samples for better visualization
    n_genes = 20000
    
    # Create realistic test data with clear clusters
    logger.info("Creating synthetic test data with clusters...")
    
    # Define cell type clusters
    cell_types = ["Enterocyte", "T Cell", "B Cell", "Macrophage", "Neuron", 
                  "Hepatocyte", "Fibroblast", "Endothelial", "Epithelial", "Stem Cell"]
    conditions = ["healthy", "inflamed", "treated", "control"]
    tissues = ["intestine", "liver", "brain", "lung", "heart"]
    
    # Create base expression profiles for each cell type
    base_expressions = {}
    for cell_type in cell_types:
        base_expression = np.random.lognormal(mean=-2, sigma=1.5, size=n_genes)
        # Add some cell-type specific signature genes
        signature_genes = np.random.choice(n_genes, size=100, replace=False)
        base_expression[signature_genes] *= np.random.lognormal(mean=1, sigma=0.5, size=100)
        base_expressions[cell_type] = base_expression
    
    transcriptome_data = np.zeros((n_samples, n_genes), dtype=np.float32)
    text_data = []
    labels = []
    
    for i in range(n_samples):
        # Assign to a cell type cluster
        cell_type_idx = i % len(cell_types)
        cell_type = cell_types[cell_type_idx]
        labels.append(cell_type_idx)
        
        condition = np.random.choice(conditions)
        tissue = np.random.choice(tissues)
        
        # Get cell-type specific base expression
        base_expr = base_expressions[cell_type].copy()
        
        # Add individual variation
        individual_factor = np.random.lognormal(mean=0, sigma=0.3, size=n_genes)
        lambda_i = base_expr * individual_factor * 100  # Scale up for counts
        
        # Generate counts
        counts = np.random.poisson(lambda_i)
        transcriptome_data[i] = counts.astype(np.float32)
        
        # Create descriptive text
        text_data.append(f"{cell_type} cell in {tissue}. Condition: {condition}. Sample ID: {i}")
    
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    
    logger.info(f"Created test data: {n_samples} samples, {n_genes} genes")
    logger.info(f"Clusters: {len(cell_types)} cell types")
    logger.info(f"Sample text example: {text_data[0]}")
    
    return transcriptome_data, text_data, gene_names, labels, cell_types

def load_real_test_data(data_path, config):
    """Load real test data from file"""
    logger = logging.getLogger(__name__)
    
    if data_path.endswith('.npy'):
        # Load numpy array
        data = np.load(data_path, allow_pickle=True).item()
        transcriptome_data = data['transcriptome']
        text_data = data['text']
        gene_names = data.get('gene_names', [f"Gene_{i}" for i in range(transcriptome_data.shape[1])])
        labels = data.get('labels', list(range(len(transcriptome_data))))
        cell_types = data.get('cell_types', None)
    elif data_path.endswith('.pkl') or data_path.endswith('.pickle'):
        # Load pickle file
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        transcriptome_data = data['transcriptome']
        text_data = data['text']
        gene_names = data.get('gene_names', [f"Gene_{i}" for i in range(transcriptome_data.shape[1])])
        labels = data.get('labels', list(range(len(transcriptome_data))))
        cell_types = data.get('cell_types', None)
    else:
        raise ValueError(f"Unsupported data format: {data_path}")
    
    logger.info(f"Loaded test data: {len(transcriptome_data)} samples, {len(gene_names)} genes")
    return transcriptome_data, text_data, gene_names, labels, cell_types

def create_umap_visualization(
    embeddings: np.ndarray,
    labels: List,
    title: str,
    label_names: Optional[List[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> plt.Figure:
    """Create UMAP visualization of embeddings"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating UMAP visualization for {len(embeddings)} embeddings...")
    
    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        metric='cosine',
    )
    
    logger.info("Fitting UMAP...")
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Prepare colors and labels
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Use tab20 colormap for many labels, otherwise categorical
    if n_labels <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_labels))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
    
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_names[i] if label_names and i < len(label_names) else f"Cluster {label}"
        
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=label_name,
            alpha=0.7,
            s=50,
            edgecolors='w',
            linewidth=0.5,
        )
    
    # Add labels and title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    
    # Add legend
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    logger.info("UMAP visualization created successfully")
    return fig

def create_joint_umap_visualization(
    transcriptome_embeds: np.ndarray,
    text_embeds: np.ndarray,
    labels: List,
    label_names: Optional[List[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> plt.Figure:
    """Create joint UMAP visualization showing both modalities"""
    
    logger = logging.getLogger(__name__)
    logger.info("Creating joint UMAP visualization...")
    
    # Combine embeddings
    combined_embeds = np.concatenate([transcriptome_embeds, text_embeds], axis=0)
    
    # Create labels for modality
    modality_labels = ['Transcriptome'] * len(transcriptome_embeds) + ['Text'] * len(text_embeds)
    
    # Create combined labels for coloring
    combined_sample_labels = list(labels) + list(labels)  # Same order for both modalities
    
    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        metric='cosine',
    )
    
    logger.info("Fitting UMAP on combined embeddings...")
    embeddings_2d = reducer.fit_transform(combined_embeds)
    
    # Split back to modalities
    transcriptome_2d = embeddings_2d[:len(transcriptome_embeds)]
    text_2d = embeddings_2d[len(transcriptome_embeds):]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Color scheme
    if n_labels <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_labels))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
    
    # Plot 1: Transcriptome embeddings only
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_names[i] if label_names and i < len(label_names) else f"Cluster {label}"
        
        axes[0].scatter(
            transcriptome_2d[mask, 0],
            transcriptome_2d[mask, 1],
            c=[colors[i]],
            label=label_name,
            alpha=0.8,
            s=80,
            edgecolors='w',
            linewidth=1,
            marker='o',
        )
    
    axes[0].set_title("Transcriptome Embeddings", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("UMAP 1", fontsize=12)
    axes[0].set_ylabel("UMAP 2", fontsize=12)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(loc='best', fontsize=9)
    
    # Plot 2: Text embeddings only
    for i, label in enumerate(unique_labels):
        mask = labels == label
        
        axes[1].scatter(
            text_2d[mask, 0],
            text_2d[mask, 1],
            c=[colors[i]],
            label=label_names[i] if label_names and i < len(label_names) else f"Cluster {label}",
            alpha=0.8,
            s=80,
            edgecolors='w',
            linewidth=1,
            marker='s',
        )
    
    axes[1].set_title("Text Embeddings", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("UMAP 1", fontsize=12)
    axes[1].set_ylabel("UMAP 2", fontsize=12)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(loc='best', fontsize=9)
    
    # Plot 3: Combined embeddings with modality differentiation
    for i, label in enumerate(unique_labels):
        mask = labels == label
        
        # Transcriptome points (circles)
        axes[2].scatter(
            transcriptome_2d[mask, 0],
            transcriptome_2d[mask, 1],
            c=[colors[i]],
            alpha=0.6,
            s=60,
            edgecolors='w',
            linewidth=0.5,
            marker='o',
            label=f"{label_names[i] if label_names and i < len(label_names) else f'Cluster {label}'} (Transcriptome)" if i == 0 else None,
        )
        
        # Text points (squares)
        axes[2].scatter(
            text_2d[mask, 0],
            text_2d[mask, 1],
            c=[colors[i]],
            alpha=0.6,
            s=60,
            edgecolors='w',
            linewidth=0.5,
            marker='s',
            label=f"{label_names[i] if label_names and i < len(label_names) else f'Cluster {label}'} (Text)" if i == 0 else None,
        )
        
        # Connect matching pairs with lines
        for j in np.where(mask)[0]:
            axes[2].plot(
                [transcriptome_2d[j, 0], text_2d[j, 0]],
                [transcriptome_2d[j, 1], text_2d[j, 1]],
                color=colors[i],
                alpha=0.2,
                linewidth=0.5,
            )
    
    axes[2].set_title("Combined Embeddings with Pair Connections", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("UMAP 1", fontsize=12)
    axes[2].set_ylabel("UMAP 2", fontsize=12)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(loc='best', fontsize=9)
    
    plt.suptitle("Joint UMAP Visualization of CellWhisperer Embeddings", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    logger.info("Joint UMAP visualization created successfully")
    return fig

def test_model(
    model, 
    test_dataset, 
    transcriptome_processor, 
    text_processor, 
    labels: List,
    cell_types: Optional[List[str]] = None,
    config: Dict = None,
    device="cuda"
):
    """Test model on test dataset with visualization"""
    logger = logging.getLogger(__name__)
    
    model.eval()
    
    # Collect all embeddings
    all_transcriptome_embeds = []
    all_text_embeds = []
    all_labels = []
    all_texts = []
    
    logger.info("Generating embeddings...")
    
    with torch.no_grad():
        # Process in batches for efficiency
        batch_size = config.get("testing", {}).get("batch_size", 32) if config else 32
        n_samples = len(test_dataset)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = list(range(start_idx, end_idx))
            
            # Prepare batch
            batch_transcriptome = []
            batch_text = []
            batch_labels = []
            
            for idx in batch_indices:
                sample = test_dataset[idx]
                batch_transcriptome.append(sample["transcriptome"])
                batch_text.append(sample["text"])
                batch_labels.append(labels[idx])
                all_texts.append(sample["text"])
            
            # Process transcriptome batch
            transcriptome_inputs = transcriptome_processor(np.array(batch_transcriptome))
            transcriptome_inputs = {k: v.to(device) for k, v in transcriptome_inputs.items()}
            
            # Process text batch
            text_inputs = text_processor(batch_text)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            # Get embeddings
            outputs = model(
                transcriptome_inputs=transcriptome_inputs,
                text_inputs=text_inputs,
            )
            
            all_transcriptome_embeds.append(outputs["transcriptome_embeddings"].cpu())
            all_text_embeds.append(outputs["text_embeddings"].cpu())
            all_labels.extend(batch_labels)
            
            if (start_idx // batch_size) % 10 == 0:
                logger.info(f"Processed {end_idx}/{n_samples} samples")
    
    # Concatenate embeddings
    transcriptome_embeds = torch.cat(all_transcriptome_embeds, dim=0).numpy()
    text_embeds = torch.cat(all_text_embeds, dim=0).numpy()
    labels_array = np.array(all_labels)
    
    logger.info(f"Generated embeddings: transcriptome={transcriptome_embeds.shape}, text={text_embeds.shape}")
    
    # Compute retrieval metrics
    logger.info("Computing retrieval metrics...")
    metrics = RetrievalMetrics.compute_all_metrics(transcriptome_embeds, text_embeds)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # Get UMAP parameters from config
    umap_config = config.get("evaluation", {}) if config else {}
    n_neighbors = umap_config.get("umap_n_neighbors", 15)
    min_dist = umap_config.get("umap_min_dist", 0.1)
    
    # Create UMAP visualizations
    fig_transcriptome = create_umap_visualization(
        transcriptome_embeds,
        labels_array,
        title="Transcriptome Embeddings (UMAP)",
        label_names=cell_types,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    
    fig_text = create_umap_visualization(
        text_embeds,
        labels_array,
        title="Text Embeddings (UMAP)",
        label_names=cell_types,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    
    fig_joint = create_joint_umap_visualization(
        transcriptome_embeds,
        text_embeds,
        labels_array,
        label_names=cell_types,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    
    # Find some example retrievals
    logger.info("\n" + "="*50)
    logger.info("RETRIEVAL EXAMPLES")
    logger.info("="*50)
    
    # Compute similarity matrix
    similarity_matrix = transcriptome_embeds @ text_embeds.T
    
    # Show top retrievals for first 5 samples
    example_results = []
    for i in range(min(5, len(test_dataset))):
        similarities = similarity_matrix[i]
        top_indices = np.argsort(-similarities)[:5]
        
        example_result = {
            "query_idx": i,
            "query_text": all_texts[i],
            "retrievals": []
        }
        
        logger.info(f"\nQuery {i} ({cell_types[labels_array[i]] if cell_types else f'Label {labels_array[i]}'}): {all_texts[i][:100]}...")
        logger.info(f"Top retrievals:")
        
        for rank, idx in enumerate(top_indices, 1):
            match_type = "✓" if idx == i else "✗"
            retrieval_text = all_texts[idx]
            similarity = similarities[idx]
            
            example_result["retrievals"].append({
                "rank": rank,
                "retrieval_idx": idx,
                "retrieval_text": retrieval_text,
                "similarity": float(similarity),
                "is_correct": idx == i,
                "label_match": labels_array[idx] == labels_array[i] if idx != i else True
            })
            
            label_info = f" ({cell_types[labels_array[idx]] if cell_types else f'Label {labels_array[idx]}'})" if idx != i else ""
            logger.info(f"  [{rank}] {match_type} {retrieval_text[:80]}... (sim={similarity:.4f}){label_info}")
        
        example_results.append(example_result)
    
    return metrics, transcriptome_embeds, text_embeds, all_texts, labels_array, {
        "fig_transcriptome": fig_transcriptome,
        "fig_text": fig_text,
        "fig_joint": fig_joint,
    }, example_results

def main():
    parser = argparse.ArgumentParser(description="Test CellWhisperer model with UMAP")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--test_data", type=str, help="Path to test data file (.npy or .pkl)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="Output directory")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load or create test data
    if args.test_data:
        logger.info(f"Loading test data from {args.test_data}")
        transcriptome_data, text_data, gene_names, labels, cell_types = load_real_test_data(args.test_data, config)
    else:
        logger.info("No test data provided, creating synthetic data with clusters")
        transcriptome_data, text_data, gene_names, labels, cell_types = create_test_data(config)
    
    # Create dataset
    test_dataset = PairedDataset(
        transcriptome_data=transcriptome_data,
        text_data=text_data,
        gene_names=gene_names,
    )
    
    # Create processors (needed for the model)
    transcriptome_processor = TranscriptomeProcessor(
        model_type=config["model"]["transcriptome_backbone"],
        max_genes=config["data"]["transcriptome"]["max_genes"],
        normalize=config["data"]["transcriptome"]["normalize"],
        log_transform=config["data"]["transcriptome"]["log_transform"],
    )
    
    # Fit processor to data (using the gene names)
    transcriptome_processor.fit(transcriptome_data, gene_names)
    
    text_processor = TextProcessor(
        model_name=config["model"]["text_backbone"],
        max_length=config["data"]["text"]["max_length"],
    )
    
    # Load model from checkpoint
    model = load_model_from_checkpoint(args.checkpoint, config, args.device)
    
    # Test model with visualizations
    logger.info("Starting model testing with UMAP visualization...")
    metrics, transcriptome_embeds, text_embeds, all_texts, labels_array, figures, example_results = test_model(
        model, test_dataset, transcriptome_processor, text_processor, 
        labels, cell_types, config, args.device
    )
    
    # Save UMAP visualizations
    logger.info("Saving UMAP visualizations...")
    
    # Save transcriptome UMAP
    transcriptome_umap_path = output_dir / "umap_transcriptome.png"
    figures["fig_transcriptome"].savefig(transcriptome_umap_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved transcriptome UMAP to: {transcriptome_umap_path}")
    
    # Save text UMAP
    text_umap_path = output_dir / "umap_text.png"
    figures["fig_text"].savefig(text_umap_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved text UMAP to: {text_umap_path}")
    
    # Save joint UMAP
    joint_umap_path = output_dir / "umap_joint.png"
    figures["fig_joint"].savefig(joint_umap_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved joint UMAP to: {joint_umap_path}")
    
    # Close figures to free memory
    plt.close('all')
    
    # Save results
    results = {
        "metrics": metrics,
        "config": config,
        "checkpoint": args.checkpoint,
        "test_data_info": {
            "n_samples": len(test_dataset),
            "data_source": args.test_data or "synthetic",
            "n_clusters": len(np.unique(labels_array)),
            "cell_types": cell_types if cell_types else list(range(len(np.unique(labels_array)))),
        },
        "embeddings_info": {
            "transcriptome_shape": transcriptome_embeds.shape,
            "text_shape": text_embeds.shape,
            "embedding_dim": transcriptome_embeds.shape[1],
        },
        "example_retrievals": example_results,
        "visualization_files": {
            "transcriptome_umap": str(transcriptome_umap_path),
            "text_umap": str(text_umap_path),
            "joint_umap": str(joint_umap_path),
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save results as JSON
    results_file = output_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Save embeddings for later analysis
    embeddings_file = output_dir / "embeddings.npz"
    np.savez_compressed(
        embeddings_file,
        transcriptome_embeddings=transcriptome_embeds,
        text_embeddings=text_embeds,
        texts=all_texts,
        labels=labels_array,
        cell_types=cell_types if cell_types else [],
    )
    
    logger.info(f"Embeddings saved to: {embeddings_file}")
    
    # Save a summary report
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CELLWHISPERER TEST RESULTS WITH UMAP VISUALIZATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test data: {args.test_data or 'Synthetic'}\n")
        f.write(f"Samples: {len(test_dataset)}\n")
        f.write(f"Clusters: {len(np.unique(labels_array))}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Retrieval Metrics:\n")
        f.write("-" * 40 + "\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name:>20}: {metric_value:.4f}\n")
        
        f.write("\nVisualization Files:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Transcriptome UMAP: {transcriptome_umap_path}\n")
        f.write(f"Text UMAP: {text_umap_path}\n")
        f.write(f"Joint UMAP: {joint_umap_path}\n")
        
        f.write("\nExample Retrievals:\n")
        f.write("-" * 40 + "\n")
        for example in example_results[:3]:  # Show first 3 examples
            f.write(f"\nQuery {example['query_idx']}:\n")
            f.write(f"  Text: {example['query_text'][:100]}...\n")
            f.write("  Top retrievals:\n")
            for retrieval in example['retrievals'][:3]:
                match_symbol = "yes" if retrieval['is_correct'] else "no"
                f.write(f"    [{retrieval['rank']}] {match_symbol} {retrieval['retrieval_text'][:80]}... (sim={retrieval['similarity']:.4f})\n")
    
    logger.info(f"Summary saved to: {summary_file}")
    

    
    logger.info("\n" + "="*80)
    logger.info("TESTING COMPLETED SUCCESSFULLY!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Visualizations: {transcriptome_umap_path}")
    logger.info(f"              {text_umap_path}")
    logger.info(f"              {joint_umap_path}")
    logger.info("="*80)

if __name__ == "__main__":
    main()