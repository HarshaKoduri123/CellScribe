import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict, Any
import umap
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

class EmbeddingVisualizer:
    """Visualize embeddings and retrieval results"""
    
    @staticmethod
    def plot_embeddings_umap(
        transcriptome_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "UMAP Visualization",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot UMAP visualization of embeddings"""
        # Combine embeddings
        combined_embeddings = np.vstack([transcriptome_embeddings, text_embeddings])
        
        # Create labels for modality
        n_transcriptome = len(transcriptome_embeddings)
        n_text = len(text_embeddings)
        modality_labels = ["Transcriptome"] * n_transcriptome + ["Text"] * n_text
        
        # Reduce dimensionality with UMAP
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        reduced_embeddings = reducer.fit_transform(combined_embeddings)
        
        # Split back
        transcriptome_umap = reduced_embeddings[:n_transcriptome]
        text_umap = reduced_embeddings[n_transcriptome:]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. Transcriptome embeddings
        scatter1 = axes[0].scatter(
            transcriptome_umap[:, 0],
            transcriptome_umap[:, 1],
            c=labels[:n_transcriptome] if labels is not None else 'blue',
            alpha=0.6,
            s=10,
            cmap='tab20' if labels is not None else None,
        )
        axes[0].set_title('Transcriptome Embeddings')
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        if labels is not None:
            plt.colorbar(scatter1, ax=axes[0])
        
        # 2. Text embeddings
        scatter2 = axes[1].scatter(
            text_umap[:, 0],
            text_umap[:, 1],
            c=labels[-n_text:] if labels is not None else 'red',
            alpha=0.6,
            s=10,
            cmap='tab20' if labels is not None else None,
        )
        axes[1].set_title('Text Embeddings')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        if labels is not None:
            plt.colorbar(scatter2, ax=axes[1])
        
        # 3. Combined with matching lines
        axes[2].scatter(
            transcriptome_umap[:, 0],
            transcriptome_umap[:, 1],
            alpha=0.3,
            s=10,
            label='Transcriptome',
        )
        axes[2].scatter(
            text_umap[:, 0],
            text_umap[:, 1],
            alpha=0.3,
            s=10,
            label='Text',
        )
        
        # Draw lines between matching pairs (sample first 10)
        n_samples = min(10, n_transcriptome)
        for i in range(n_samples):
            axes[2].plot(
                [transcriptome_umap[i, 0], text_umap[i, 0]],
                [transcriptome_umap[i, 1], text_umap[i, 1]],
                'k-',
                alpha=0.2,
                linewidth=0.5,
            )
        
        axes[2].set_title('Aligned Embeddings')
        axes[2].set_xlabel('UMAP 1')
        axes[2].set_ylabel('UMAP 2')
        axes[2].legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_retrieval_results(
        similarity_matrix: np.ndarray,
        query_indices: List[int],
        text_data: List[str],
        transcriptome_labels: Optional[List[str]] = None,
        n_examples: int = 5,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot retrieval examples"""
        fig, axes = plt.subplots(n_examples, 2, figsize=figsize)
        
        if n_examples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, query_idx in enumerate(query_indices[:n_examples]):
            # Get top retrieved texts
            similarities = similarity_matrix[query_idx]
            top_indices = np.argsort(-similarities)[:5]
            
            # Query information
            query_label = transcriptome_labels[query_idx] if transcriptome_labels else f"Query {query_idx}"
            
            # Plot query
            axes[idx, 0].text(0.5, 0.5, f"Query:\n{query_label}", 
                            ha='center', va='center', fontsize=12)
            axes[idx, 0].axis('off')
            axes[idx, 0].set_title(f'Query {idx+1}')
            
            # Plot retrieved texts
            retrieved_texts = "\n".join([
                f"{i+1}. {text_data[j][:50]}..." 
                for i, j in enumerate(top_indices)
            ])
            
            axes[idx, 1].text(0.1, 0.9, f"Top retrieved texts:\n\n{retrieved_texts}", 
                            ha='left', va='top', fontsize=10)
            axes[idx, 1].axis('off')
            axes[idx, 1].set_title(f'Retrieval Results (similarities: {similarities[top_indices[0]]:.3f})')
        
        plt.suptitle("Retrieval Examples", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_similarity_matrix(
        similarity_matrix: np.ndarray,
        transcriptome_labels: Optional[List[str]] = None,
        text_labels: Optional[List[str]] = None,
        title: str = "Similarity Matrix",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot similarity matrix heatmap"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        
        # Add labels if provided
        if transcriptome_labels is not None and len(transcriptome_labels) <= 20:
            ax.set_yticks(range(len(transcriptome_labels)))
            ax.set_yticklabels(transcriptome_labels, fontsize=8)
        
        if text_labels is not None and len(text_labels) <= 20:
            ax.set_xticks(range(len(text_labels)))
            ax.set_xticklabels(text_labels, rotation=45, ha='right', fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        ax.set_xlabel("Text Descriptions")
        ax.set_ylabel("Transcriptome Samples")
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_training_curves(
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        val_metrics: Optional[Dict[str, List[float]]] = None,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot training curves"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Training loss
        axes[0].plot(train_losses, label='Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Validation loss
        if val_losses is not None:
            axes[1].plot(val_losses, label='Validation Loss', color='orange')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Validation Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Validation metrics
        if val_metrics is not None:
            for metric_name, metric_values in val_metrics.items():
                axes[2].plot(metric_values, label=metric_name)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Metric Value')
            axes[2].set_title('Validation Metrics')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    @staticmethod
    def create_interactive_plot(
        embeddings: np.ndarray,
        labels: np.ndarray,
        hover_text: Optional[List[str]] = None,
        title: str = "Interactive Embedding Visualization",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create interactive plot with Plotly"""
        # Reduce to 2D/3D
        if embeddings.shape[1] > 3:
            reducer = umap.UMAP(n_components=3, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings
        
        # Create hover text
        if hover_text is None:
            hover_text = [f"Sample {i}" for i in range(len(embeddings))]
        
        # Create figure
        if embeddings_2d.shape[1] == 3:
            fig = px.scatter_3d(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                z=embeddings_2d[:, 2],
                color=labels.astype(str),
                hover_name=hover_text,
                title=title,
            )
        else:
            fig = px.scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                color=labels.astype(str),
                hover_name=hover_text,
                title=title,
            )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig