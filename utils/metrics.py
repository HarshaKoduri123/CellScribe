import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# Add this function at the top level
def compute_retrieval_metrics(
    transcriptome_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Compute all retrieval metrics"""
    return RetrievalMetrics.compute_all_metrics(
        transcriptome_embeddings, text_embeddings, labels, k_values
    )

class RetrievalMetrics:
    """Compute retrieval metrics for multimodal embeddings"""
    
    @staticmethod
    def compute_retrieval_at_k(
        similarity_matrix: np.ndarray,
        query_labels: np.ndarray,
        gallery_labels: np.ndarray,
        k_values: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """
        Compute Recall@K for retrieval
        
        Args:
            similarity_matrix: [n_queries, n_gallery] similarity scores
            query_labels: Labels for query samples
            gallery_labels: Labels for gallery samples
            k_values: List of K values to compute
            
        Returns:
            Dictionary of Recall@K values
        """
        n_queries = similarity_matrix.shape[0]
        
        # Get top-k indices for each query
        top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :max(k_values)]
        
        metrics = {}
        for k in k_values:
            recalls = []
            for i in range(n_queries):
                retrieved_labels = gallery_labels[top_k_indices[i, :k]]
                # Check if correct label is in retrieved labels
                recall = int(query_labels[i] in retrieved_labels)
                recalls.append(recall)
            
            metrics[f"recall@{k}"] = np.mean(recalls)
        
        return metrics
    
    @staticmethod
    def compute_mean_average_precision(
        similarity_matrix: np.ndarray,
        query_labels: np.ndarray,
        gallery_labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute mean Average Precision (mAP)
        
        Args:
            similarity_matrix: [n_queries, n_gallery] similarity scores
            query_labels: Labels for query samples
            gallery_labels: Labels for gallery samples
            
        Returns:
            Dictionary with mAP values
        """
        n_queries = similarity_matrix.shape[0]
        
        # Compute AP for each query
        aps = []
        for i in range(n_queries):
            # Sort by similarity
            sorted_indices = np.argsort(-similarity_matrix[i])
            sorted_relevance = (gallery_labels[sorted_indices] == query_labels[i])
            
            # Compute precision at each position
            n_relevant = np.cumsum(sorted_relevance)
            positions = np.arange(1, len(sorted_relevance) + 1)
            precision_at_k = n_relevant / positions
            
            # Compute AP (area under precision-recall curve)
            if sorted_relevance.sum() > 0:
                ap = np.sum(precision_at_k * sorted_relevance) / sorted_relevance.sum()
                aps.append(ap)
        
        mAP = np.mean(aps) if aps else 0.0
        
        return {
            "mAP": mAP,
            "AP_std": np.std(aps) if aps else 0.0,
        }
    
    @staticmethod
    def compute_precision_recall(
        similarity_matrix: np.ndarray,
        query_labels: np.ndarray,
        gallery_labels: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute precision-recall curve
        
        Returns:
            Dictionary with precision, recall, thresholds
        """
        # Flatten similarity matrix and create binary relevance
        similarities = similarity_matrix.flatten()
        relevance = np.zeros_like(similarities, dtype=bool)
        
        n_queries = similarity_matrix.shape[0]
        n_gallery = similarity_matrix.shape[1]
        
        for i in range(n_queries):
            start_idx = i * n_gallery
            end_idx = (i + 1) * n_gallery
            relevance[start_idx:end_idx] = (gallery_labels == query_labels[i])
        
        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(relevance, similarities)
        
        # Compute average precision
        ap = average_precision_score(relevance, similarities)
        
        return {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "average_precision": ap,
        }
    
    @staticmethod
    def compute_similarity_matrix(
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """Compute cosine similarity matrix"""
        if normalize:
            embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
            embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)
        
        return embeddings_a @ embeddings_b.T
    
    @staticmethod
    def compute_all_metrics(
        transcriptome_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        k_values: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """Compute all retrieval metrics"""
        
        # Compute similarity matrix
        similarity_matrix = RetrievalMetrics.compute_similarity_matrix(
            transcriptome_embeddings, text_embeddings
        )
        
        # If no labels provided, use indices (self-supervised)
        if labels is None:
            n_samples = transcriptome_embeddings.shape[0]
            labels = np.arange(n_samples)
        
        # Split into query and gallery (50/50)
        n_samples = len(labels)
        query_indices = np.random.choice(n_samples, n_samples // 2, replace=False)
        gallery_indices = np.array([i for i in range(n_samples) if i not in query_indices])
        
        query_labels = labels[query_indices]
        gallery_labels = labels[gallery_indices]
        query_similarities = similarity_matrix[query_indices][:, gallery_indices]
        
        # Compute metrics
        metrics = {}
        
        # Recall@K
        recall_metrics = RetrievalMetrics.compute_retrieval_at_k(
            query_similarities, query_labels, gallery_labels, k_values
        )
        metrics.update(recall_metrics)
        
        # mAP
        map_metrics = RetrievalMetrics.compute_mean_average_precision(
            query_similarities, query_labels, gallery_labels
        )
        metrics.update(map_metrics)
        
        # Precision-Recall
        pr_metrics = RetrievalMetrics.compute_precision_recall(
            query_similarities, query_labels, gallery_labels
        )
        metrics["average_precision"] = pr_metrics["average_precision"]
        
        # Print summary
        logger.info("Retrieval Metrics Summary:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        return metrics

class AlignmentMetrics:
    """Metrics for assessing alignment between modalities"""
    
    @staticmethod
    def compute_alignment_loss(
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        pairs: np.ndarray,
    ) -> float:
        """Compute alignment loss for paired embeddings"""
        # Ensure embeddings are normalized
        embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdim=True)
        embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdim=True)
        
        # Compute cosine similarities for positive pairs
        similarities = np.sum(embeddings_a[pairs[:, 0]] * embeddings_b[pairs[:, 1]], axis=1)
        
        # Alignment loss (1 - average similarity)
        return 1.0 - np.mean(similarities)
    
    @staticmethod
    def compute_uniformity_loss(embeddings: np.ndarray) -> float:
        """Compute uniformity loss (spread of embeddings)"""
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdim=True)
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(embeddings[:1000])  # Sample for efficiency
        
        # Uniformity loss (negative log of average pairwise distance)
        avg_distance = np.mean(distances)
        return -np.log(avg_distance)
    
    @staticmethod
    def compute_modality_gap(
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> Dict[str, float]:
        """Compute gap between two modalities"""
        # Compute centers
        center_a = np.mean(embeddings_a, axis=0)
        center_b = np.mean(embeddings_b, axis=0)
        
        # Center shift
        center_shift = np.linalg.norm(center_a - center_b)
        
        # Within-modality variances
        var_a = np.mean(np.var(embeddings_a, axis=0))
        var_b = np.mean(np.var(embeddings_b, axis=0))
        
        # Between-modality distance
        
        avg_cross_distance = np.mean(cdist(embeddings_a[:500], embeddings_b[:500]))
        
        return {
            "center_shift": float(center_shift),
            "variance_a": float(var_a),
            "variance_b": float(var_b),
            "avg_cross_distance": float(avg_cross_distance),
            "variance_ratio": float(var_a / var_b if var_b > 0 else 0),
        }