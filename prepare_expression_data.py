import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.sparse import issparse

EMBED_DIR  = "./embeddings"
TARGET_DIR = "./targets"

os.makedirs(TARGET_DIR, exist_ok=True)

embed_files = sorted([f for f in os.listdir(EMBED_DIR) if f.endswith('.pkl')])
print(f"Found {len(embed_files)} embedding files")

for fname in embed_files:
    with open(os.path.join(EMBED_DIR, fname), "rb") as f:
        data = pickle.load(f)

    sample_id   = data['sample_id']
    cancer_type = data['cancer_type']
    adata_path  = data['adata_path']
    barcodes    = data['barcodes']
    embeddings  = data['embeddings']   # raw (N, 2048) — keep as is

    print(f"Processing {sample_id} ({cancer_type})")

    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    expr_matrix = adata.X
    if issparse(expr_matrix):
        expr_matrix = expr_matrix.toarray()

    gene_names    = list(adata.var_names)
    adata_barcodes = list(adata.obs_names)
    bc_to_idx      = {bc: i for i, bc in enumerate(adata_barcodes)}

    # Align barcodes
    aligned_X, aligned_y = [], []
    for i, bc in enumerate(barcodes):
        if bc in bc_to_idx:
            aligned_X.append(embeddings[i])
            aligned_y.append(expr_matrix[bc_to_idx[bc]])

    if len(aligned_X) == 0:
        bc_clean_map = {bc.split('-')[0]: i for i, bc in enumerate(adata_barcodes)}
        for i, bc in enumerate(barcodes):
            bc_clean = bc.split('-')[0]
            if bc_clean in bc_clean_map:
                aligned_X.append(embeddings[i])
                aligned_y.append(expr_matrix[bc_clean_map[bc_clean]])

    if len(aligned_X) == 0:
        print(f"No overlap for {sample_id}, skipping.")
        continue

    aligned_X = np.array(aligned_X, dtype=np.float32)   # (N, 2048)
    aligned_y = np.array(aligned_y, dtype=np.float32)   # (N, all_genes)

    print(f"Aligned: {len(aligned_X)} spots | X={aligned_X.shape} | y={aligned_y.shape}")

    out = {
        "sample_id"   : sample_id,
        "cancer_type" : cancer_type,
        "X"           : aligned_X,
        "y"           : aligned_y,
        "gene_names"  : gene_names,
    }
    with open(os.path.join(TARGET_DIR, f"{sample_id}.pkl"), "wb") as f:
        pickle.dump(out, f)