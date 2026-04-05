# HistoMoE

Predicting spatial transcriptomics expression from histology patch embeddings.

This repository currently contains two modeling pipelines:

1. **Initial Baseline (implemented first):**
   - ResNet50 patch embeddings (2048-d) -> PCA (256-d) -> RidgeCV multi-target regression
   - Evaluation with Leave-One-Patient-Out-style CV (implemented as leave-one-sample-id-out)
2. **Mixture-of-Experts (MoE) Extension (implementation underway):**
   - Adds metadata-conditioned routing + 4 neural experts on top of PCA embeddings
   - Trained with warmup supervision and sparse top-2 routing

---

## 1) Repository structure

- [download_data.py](download_data.py): download HEST subset (patches, ST matrices, metadata JSON)
- [extract_embeddings.py](extract_embeddings.py): extract ResNet50 patch embeddings from patch H5 files
- [prepare_expression_data.py](prepare_expression_data.py): align embeddings with ST barcodes and save per-sample target pickles
- [train_and_evaluate.py](train_and_evaluate.py): baseline PCA + RidgeCV + LOPOCV
- [plot_results.py](plot_results.py): baseline results bar plot
- [moe/dataset.py](moe/dataset.py): data utilities + metadata vectorization for MoE
- [moe/gating.py](moe/gating.py): gating network
- [moe/experts.py](moe/experts.py): expert regression head
- [moe/moe_model.py](moe/moe_model.py): MoE assembly and top-2 sparse routing

---

## 2) Data (Not Pushed) and task overview

### Inputs
- Histology patches per sample: H5 files under hest_data/patches
- Spatial transcriptomics matrices per sample: H5AD files under hest_data/st
- Sample-level metadata: CSV and per-sample JSON under hest_data/subset_meta.csv and hest_data/metadata

### Prediction target
For each spatial spot, predict expression of top variable genes (HVGs) using image-derived features.

### Cancer cohorts
The current setup filters to 4 OncoTree cohorts in [download_data.py](download_data.py):
- SKCM
- LUAD
- PAAD
- READ

---

## 3) Initial baseline implementation (first)

This section describes the original working pipeline before MoE.

### Step A: Download selected HEST subset
Script: [download_data.py](download_data.py)

What it does:
- Downloads HEST metadata table from Hugging Face.
- Filters rows to OncoTree codes SKCM/LUAD/PAAD/READ.
- Saves filtered sample table to hest_data/subset_meta.csv.
- Downloads only required files for each selected sample:
  - patches/{sample}.h5
  - st/{sample}.h5ad
  - metadata/{sample}.json

### Step B: Extract image embeddings
Script: [extract_embeddings.py](extract_embeddings.py)

What it does:
- Loads ImageNet-pretrained ResNet50 and removes classification head.
- Encodes each patch into a 2048-d embedding.
- Handles barcode decoding from H5 robustly.
- Saves per-sample pickle in embeddings/ with:
  - sample_id
  - cancer_type
  - embeddings (N_spots x 2048)
  - barcodes
  - adata_path

### Step C: Build aligned supervised learning targets
Script: [prepare_expression_data.py](prepare_expression_data.py)

What it does:
- Loads embeddings pickle + matching H5AD expression matrix.
- Normalizes expression with Scanpy (normalize_total + log1p).
- Aligns image patch barcodes to ST barcodes.
- Saves per-sample training pickle in targets/ with:
  - sample_id
  - cancer_type
  - X (N_spots x 2048)
  - y (N_spots x N_genes)
  - gene_names

### Step D: Train/evaluate baseline model
Script: [train_and_evaluate.py](train_and_evaluate.py)

Core pipeline:
- Loads all per-sample target pickles from targets/.
- Drops low-gene-panel samples (MIN_GENES = 5000).
- Finds genes common across remaining samples.
- Selects top 50 HVGs globally by pooled variance.
- Builds spot-level dataset.
- Runs LeaveOneGroupOut CV with group = sample_id.
- For each fold:
  - PCA fitted on train spots only (up to 256 components).
  - RidgeCV model predicts all 50 genes jointly.
  - Pearson r computed per gene on held-out sample spots, then averaged.
- Writes summary CSV:
  - results/global_baseline_results.csv


---

## 4) MoE implementation (second)

The MoE extension preserves baseline data preparation and fold logic, then replaces RidgeCV with a metadata-conditioned sparse MoE regressor.

### 4.1 Metadata conditioning vector
Implemented in [moe/dataset.py](moe/dataset.py)

For each sample, metadata fields are collected from subset CSV + sample JSON:
- Categorical (one-hot):
  - oncotree_code
  - st_technology
  - preservation_method
- Scalar:
  - spots_under_tissue -> log1p -> z-score
  - magnification -> parsed numeric value -> z-score

This sample-level vector is broadcast to all spots in that sample, then concatenated with PCA embeddings.

Final model input per spot:
- PCA(image embedding): up to 256 dims
- Metadata vector: dynamic (typically ~10-15 dims)
- Concatenated input: around ~268 dims in typical runs

### 4.2 Gating network
Implemented in [moe/gating.py](moe/gating.py)

Architecture:
- Linear(input_dim, 64)
- ReLU
- Linear(64, num_experts)

Output:
- Raw routing logits for 4 experts.

### 4.3 Expert heads
Implemented in [moe/experts.py](moe/experts.py)

Each expert architecture:
- Linear(input_dim, 128)
- ReLU
- Linear(128, output_dim)

Output_dim defaults to number of target genes (50 HVGs).

### 4.4 Sparse top-2 routing
Implemented in [moe/moe_model.py](moe/moe_model.py)

Forward pass:
- Softmax on gating logits.
- Keep only top-2 expert weights per spot.
- Zero out remaining experts.
- Renormalize top-2 weights to sum to 1.
- Predict each expert output.
- Final prediction = weighted sum across experts.

### 4.5 Two-phase training objective (To Be Implemented)
Implemented in [train_moe.py](train_moe.py)

Phase 1 (warmup):
- Loss = MSE(pred, y) + lambda_ce * CrossEntropy(gating_logits, cancer_label)
- Purpose: initialize routing by cancer type.

Phase 2 (main):
- Loss = MSE(pred, y) + lambda_ent * entropy(sparse_routing_weights)
- Purpose: keep routing confident while avoiding degenerate behavior.


---

## 5) How to run

### 5.1 Environment
Use Python 3.10+ recommended.

Install common dependencies used across scripts:

pip install numpy pandas scipy scikit-learn torch torchvision scanpy anndata h5py pillow tqdm plotly kaleido huggingface_hub

Optional:
- Set HF token if dataset access requires authentication:
  - export HF_TOKEN=your_token

### 5.2 Baseline pipeline
Run in this order:

1) python download_data.py
2) python extract_embeddings.py
3) python prepare_expression_data.py
4) python train_and_evaluate.py
5) python plot_results.py

Expected baseline outputs:
- results/global_baseline_results.csv
- results/global_baseline_bar.png

---

## 6) Baseline vs MoE at a glance

### Baseline
- Model: PCA + RidgeCV
- Metadata use: none
- Interpretability: per-cancer mean performance only
- Complexity: low

### MoE
- Model: metadata-conditioned sparse neural mixture of 4 experts
- Metadata use: explicit sample-level conditioning
- Interpretability: per-spot expert assignment logs
- Complexity: moderate

---
