import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

logger = logging.getLogger(__name__)

class EmbeddingExtractor:
    """Utility for extracting and processing embeddings"""
    
    @staticmethod
    def extract_batch_embeddings(model, dataloader, device="cuda"):
        """Extract embeddings for entire dataset"""
        model.eval()
        
        all_transcriptome_embeds = []
        all_text_embeds = []
        all_labels = []
        all_metadata = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    transcriptome_inputs=batch["transcriptome_inputs"],
                    text_inputs=batch["text_inputs"],
                )
                
                # Store embeddings
                all_transcriptome_embeds.append(outputs["transcriptome_embeddings"].cpu())
                all_text_embeds.append(outputs["text_embeddings"].cpu())
                
                # Store labels if available
                if "labels" in batch:
                    all_labels.append(batch["labels"].cpu())
                
                # Store metadata if available
                if "metadata" in batch:
                    all_metadata.extend(batch["metadata"])
        
        # Concatenate
        transcriptome_embeds = torch.cat(all_transcriptome_embeds, dim=0).numpy()
        text_embeds = torch.cat(all_text_embeds, dim=0).numpy()
        
        if all_labels:
            labels = torch.cat(all_labels, dim=0).numpy()
        else:
            labels = None
        
        return {
            "transcriptome_embeddings": transcriptome_embeds,
            "text_embeddings": text_embeds,
            "labels": labels,
            "metadata": all_metadata,
        }
    
    @staticmethod
    def reduce_dimensionality(embeddings: np.ndarray, method: str = "umap", 
                            n_components: int = 2, **kwargs) -> np.ndarray:
        """Reduce embedding dimensionality for visualization"""
        if method == "umap":
            reducer = umap.UMAP(n_components=n_components, **kwargs)
        elif method == "pca":
            reducer = PCA(n_components=n_components, **kwargs)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        return reducer.fit_transform(embeddings)
    
    @staticmethod
    def compute_similarity_matrix(embeddings_a: np.ndarray, embeddings_b: np.ndarray,
                                normalize: bool = True) -> np.ndarray:
        """Compute cosine similarity matrix between two sets of embeddings"""
        if normalize:
            embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
            embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)
        
        return embeddings_a @ embeddings_b.T
    
    @staticmethod
    def find_nearest_neighbors(query_embedding: np.ndarray, reference_embeddings: np.ndarray,
                              k: int = 5, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for query embedding"""
        similarities = EmbeddingExtractor.compute_similarity_matrix(
            query_embedding.reshape(1, -1), 
            reference_embeddings,
            normalize=normalize
        ).flatten()
        
        # Get indices of top k similarities
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_similarities = similarities[top_k_indices]
        
        return top_k_indices, top_k_similarities
    
    @staticmethod
    def interpolate_embeddings(embedding_a: np.ndarray, embedding_b: np.ndarray,
                             n_steps: int = 10) -> np.ndarray:
        """Interpolate between two embeddings"""
        alphas = np.linspace(0, 1, n_steps)
        interpolated = []
        
        for alpha in alphas:
            interp = (1 - alpha) * embedding_a + alpha * embedding_b
            interpolated.append(interp)
        
        return np.array(interpolated)

class EmbeddingAnalyzer:
    """Analyze embeddings for insights"""
    
    @staticmethod
    def compute_cluster_quality(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute clustering quality metrics"""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        if len(np.unique(labels)) < 2:
            logger.warning("Need at least 2 clusters for quality metrics")
            return {}
        
        try:
            silhouette = silhouette_score(embeddings, labels)
            calinski = calinski_harabasz_score(embeddings, labels)
            davies = davies_bouldin_score(embeddings, labels)
            
            return {
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski,
                "davies_bouldin_score": davies,
            }
        except Exception as e:
            logger.error(f"Error computing cluster quality: {e}")
            return {}
    
    @staticmethod
    def analyze_embedding_space(embeddings: np.ndarray) -> Dict[str, Any]:
        """Analyze properties of embedding space"""
        # Compute statistics
        mean = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0)
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(embeddings[:100])  # Sample for efficiency
        
        return {
            "embedding_mean": mean,
            "embedding_std": std,
            "mean_distance": np.mean(distances),
            "std_distance": np.std(distances),
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
            "n_embeddings": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1],
        }
    
    @staticmethod
    def compute_embedding_drift(embeddings_before: np.ndarray, embeddings_after: np.ndarray) -> Dict[str, float]:
        """Compute drift between two sets of embeddings"""
        # Center embeddings
        center_before = np.mean(embeddings_before, axis=0)
        center_after = np.mean(embeddings_after, axis=0)
        
        # Compute drift metrics
        center_drift = np.linalg.norm(center_after - center_before)
        
        # Compute distribution statistics
        from scipy.spatial.distance import jensenshannon
        from scipy.stats import wasserstein_distance
        
        # Flatten embeddings for distribution comparison
        flat_before = embeddings_before.flatten()
        flat_after = embeddings_after.flatten()
        
        # Sample for efficiency
        if len(flat_before) > 10000:
            flat_before = np.random.choice(flat_before, 10000, replace=False)
            flat_after = np.random.choice(flat_after, 10000, replace=False)
        
        # Compute distribution distances
        js_distance = jensenshannon(flat_before, flat_after)
        wasserstein_dist = wasserstein_distance(flat_before, flat_after)
        
        return {
            "center_drift": float(center_drift),
            "jensen_shannon_distance": float(js_distance),
            "wasserstein_distance": float(wasserstein_dist),
        }