# data/geneformer_processor.py


import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
import logging
import warnings
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from geneformer.in_silico_perturber import pad_tensor_list
from geneformer.tokenizer import TranscriptomeTokenizer, rank_genes


import anndata
import scanpy as sc

logger = logging.getLogger(__name__)


PAD_TOKEN_ID = 0
MODEL_INPUT_SIZE = 2048

class GeneformerProcessor:

    
    def __init__(
        self,
        nproc: int = 4,
        emb_label: List[str] = None,
        max_genes: int = 2048,
        target_sum: int = 10000,
        normalize: bool = True,
        log_transform: bool = True,
    ):
        self.nproc = nproc
        self.emb_label = emb_label or ["cell_type"]
        self.max_genes = max_genes
        self.target_sum = target_sum
        self.normalize = normalize
        self.log_transform = log_transform
        

        
        self.tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict={k: k for k in self.emb_label},
            nproc=self.nproc,
        )
       

        self.gene_annotation = None
        self.gene_to_index = {}
        self.index_to_gene = {}
        
    def fit(self, expression_matrix: np.ndarray, gene_names: List[str]):

        self.gene_names = gene_names[:self.max_genes]

        self.gene_to_index = {gene: idx for idx, gene in enumerate(self.gene_names)}
        self.index_to_gene = {idx: gene for gene, idx in self.gene_to_index.items()}
        
        logger.info(f"Created vocabulary with {len(self.gene_names)} genes")
        return self
    
    def __call__(
        self,
        expression_data: Union[np.ndarray, List[np.ndarray], anndata.AnnData],
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, Any]:


        if isinstance(expression_data, anndata.AnnData):
            return self._process_anndata(expression_data, return_tensors, **kwargs)
        elif isinstance(expression_data, np.ndarray):
            if expression_data.ndim == 1:
    
                expression_data = [expression_data]
            batch_list = [expression_data[i] for i in range(expression_data.shape[0])]
            return self._process_batch(batch_list, return_tensors, **kwargs)
        elif isinstance(expression_data, list):
            return self._process_batch(expression_data, return_tensors, **kwargs)
        else:
            raise TypeError(f"Unsupported input type: {type(expression_data)}")
    
    def _process_anndata(
        self,
        adata: anndata.AnnData,
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, Any]:

        prepared_adata = self._prepare_anndata(adata)

        tokens, lengths = self._tokenize_anndata(prepared_adata, **kwargs)

        if return_tensors == "pt":
            max_len = max(lengths)
            tokens = [torch.from_numpy(v).to(dtype=torch.long) for v in tokens]
            tokens = pad_tensor_list(
                tokens,
                max_len,
                pad_token_id=PAD_TOKEN_ID,
                model_input_size=MODEL_INPUT_SIZE,
            )
            lengths = torch.tensor(lengths)
        
        return {
            "expression_tokens": tokens,
            "expression_token_lengths": lengths,
        }
    
    def _process_batch(
        self,
        expression_batch: List[np.ndarray],
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, Any]:

        processed_batch = []
        lengths = []
        
        for expr in expression_batch:

            tokens = self._process_single(expr)
            processed_batch.append(tokens)
            lengths.append(len(tokens))

        if return_tensors == "pt":
            max_len = max(lengths)
            
            padded_tokens = []
            for tokens in processed_batch:
                if len(tokens) < max_len:
                    padding = [PAD_TOKEN_ID] * (max_len - len(tokens))
                    padded = tokens + padding
                else:
                    padded = tokens[:max_len]
                padded_tokens.append(torch.tensor(padded, dtype=torch.long))
            
            tokens_tensor = torch.stack(padded_tokens)
            lengths_tensor = torch.tensor(lengths, dtype=torch.long)
            
            return {
                "expression_tokens": tokens_tensor,
                "expression_token_lengths": lengths_tensor,
            }
        else:
            return {
                "expression_tokens": processed_batch,
                "expression_token_lengths": lengths,
            }
    
    def _process_single(self, expression_vector: np.ndarray) -> List[int]:
        expression_vector = np.ascontiguousarray(expression_vector)

        expression_vector = np.maximum(expression_vector, 0)

        if self.normalize:
            total_counts = expression_vector.sum()
            if total_counts > 0:
                expression_vector = expression_vector / total_counts * self.target_sum

        if self.log_transform:
            expression_vector = np.log1p(expression_vector)

        n_genes = min(self.max_genes, len(expression_vector))
        
        if len(expression_vector) > n_genes:

            top_idx = np.argpartition(-expression_vector, n_genes)[:n_genes]

            top_idx_sorted = top_idx[np.argsort(-expression_vector[top_idx])]
        else:
            top_idx_sorted = np.arange(len(expression_vector))

        tokens = top_idx_sorted.astype(np.int64).tolist()
        
        return tokens
    
    def _prepare_anndata(self, adata: anndata.AnnData) -> anndata.AnnData:

        adata = adata.copy()

        if "ensembl_id" not in adata.var.columns:

            if self.gene_annotation is None:
                self._load_gene_annotation()

            ensembl_ids = []
            for gene in adata.var.index:
                if gene in self.gene_annotation:
                    ensembl_ids.append(self.gene_annotation[gene])
                else:
                    ensembl_ids.append("")
            
            adata.var["ensembl_id"] = ensembl_ids

        valid_genes = [x.startswith("ENSG0") for x in adata.var["ensembl_id"]]
        adata = adata[:, valid_genes]
 
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        adata.obs["n_counts"] = adata.obs.total_counts
        
        return adata
    
    def _tokenize_anndata(
        self,
        adata: anndata.AnnData,
        chunk_size: int = 512,
        **kwargs
    ) -> tuple[List[np.ndarray], List[int]]:

        coding_miRNA_mask = np.array([
            self.tokenizer.genelist_dict.get(i, False)
            for i in adata.var["ensembl_id"]
        ])
        coding_miRNA_idx = np.where(coding_miRNA_mask)[0]
  
        norm_factors = np.array([
            self.tokenizer.gene_median_dict[i]
            for i in adata.var["ensembl_id"].iloc[coding_miRNA_idx]
        ])

        gene_tokens = np.array([
            self.tokenizer.gene_token_dict[i]
            for i in adata.var["ensembl_id"].iloc[coding_miRNA_idx]
        ])

        tokens = []
        lengths = []
        
        n_cells = adata.shape[0]
        for i in range(0, n_cells, chunk_size):
            idx = list(range(i, min(i + chunk_size, n_cells)))

            counts = adata[idx, coding_miRNA_idx].X
            n_counts = adata[idx].obs["n_counts"].values[:, None]
            
            if sp.issparse(counts):
                counts = counts.toarray()

            normalized = counts / n_counts * self.target_sum / norm_factors

            for j in range(len(idx)):

                cell_expr = normalized[j]
                non_zero = cell_expr > 0
                
                if non_zero.any():

                    ranked_genes = rank_genes(
                        cell_expr[non_zero],
                        gene_tokens[non_zero]
                    )
                else:

                    ranked_genes = np.array([0], dtype=np.int16)

                ranked_genes = ranked_genes[:MODEL_INPUT_SIZE]
                tokens.append(ranked_genes)
                lengths.append(len(ranked_genes))
        
        return tokens, lengths
    
    def _load_gene_annotation(self):
        self.gene_annotation = {}