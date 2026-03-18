import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import pearsonr
from collections import defaultdict

TARGET_DIR  = "./targets"
RESULTS_DIR = "./results"
N_HVG       = 50
N_PCA       = 256
MIN_GENES   = 5000    # drop samples with fewer genes than this (STv1 has only 541)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load all samples
all_samples = []
for fname in sorted(os.listdir(TARGET_DIR)):
    if fname.endswith('.pkl'):
        with open(os.path.join(TARGET_DIR, fname), "rb") as f:
            all_samples.append(pickle.load(f))

print(f"Loaded {len(all_samples)} samples")
for s in all_samples:
    print(f"  {s['sample_id']:12s} | {s['cancer_type']:4s} | spots={s['X'].shape[0]} | genes={len(s['gene_names'])}")

# Filter out samples with very small gene panels (old STv1 technology)
filtered = [s for s in all_samples if len(s['gene_names']) >= MIN_GENES]
dropped  = [s['sample_id'] for s in all_samples if len(s['gene_names']) < MIN_GENES]

print(f"Dropped {len(dropped)} low-gene-panel samples: {dropped}")
print(f"Keeping {len(filtered)} samples")

# Check we still have all cancer types
remaining_cancers = set(s['cancer_type'] for s in filtered)
print(f"Remaining cancer types: {remaining_cancers}")

# Step 1: find common genes across filtered samples
print("Finding common genes across filtered samples...")
gene_sets    = [set(s['gene_names']) for s in filtered]
common_genes = sorted(list(gene_sets[0].intersection(*gene_sets[1:])))
print(f"  Common genes: {len(common_genes)}")

if len(common_genes) == 0:
    print("Still no common genes. Printing gene counts per sample:")
    for s in filtered:
        print(f"  {s['sample_id']}: {len(s['gene_names'])} genes")
    exit()

# Step 2: select top-50 HVGs by variance across all spots pooled
print("Selecting top-50 HVGs by pooled variance...")
all_expr = []
for s in filtered:
    gene_idx = [s['gene_names'].index(g) for g in common_genes]
    all_expr.append(s['y'][:, gene_idx])

pooled_expr    = np.vstack(all_expr)
gene_var       = np.var(pooled_expr, axis=0)
top50_idx      = np.argsort(gene_var)[::-1][:N_HVG]
selected_genes = [common_genes[i] for i in top50_idx]
print(f"  Top HVGs: {selected_genes[:10]}...")

# Step 3: build final arrays
X_all, y_all   = [], []
cancer_labels  = []
patient_labels = []

for s in filtered:
    gene_idx = [s['gene_names'].index(g) for g in selected_genes]
    y = s['y'][:, gene_idx]
    X = s['X']
    n = X.shape[0]
    X_all.append(X)
    y_all.append(y)
    cancer_labels.extend([s['cancer_type']] * n)
    patient_labels.extend([s['sample_id']]  * n)

X_all          = np.vstack(X_all)
y_all          = np.vstack(y_all)
cancer_labels  = np.array(cancer_labels)
patient_labels = np.array(patient_labels)

print(f"Final dataset: {X_all.shape[0]} spots | {X_all.shape[1]} features | {y_all.shape[1]} genes")

# Step 4: Patient-stratified LOOCV with PCA fit per fold
logo    = LeaveOneGroupOut()
results = defaultdict(list)

print("--- Running Leave-One-Patient-Out CV ---")

for fold, (train_idx, test_idx) in enumerate(logo.split(X_all, y_all, groups=patient_labels)):
    test_patient = patient_labels[test_idx][0]
    test_cancer  = cancer_labels[test_idx][0]

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    # PCA fit on training fold only
    n_components = min(N_PCA, X_train.shape[0] - 1, X_train.shape[1])
    pca          = PCA(n_components=n_components)
    X_train_pca  = pca.fit_transform(X_train)
    X_test_pca   = pca.transform(X_test)

    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    pearson_scores = []
    for g in range(y_test.shape[1]):
        if np.std(y_test[:, g]) < 1e-6:
            continue
        r, _ = pearsonr(y_test[:, g], y_pred[:, g])
        if not np.isnan(r):
            pearson_scores.append(r)

    if len(pearson_scores) == 0:
        continue

    mean_r = np.mean(pearson_scores)
    results[test_cancer].append(mean_r)
    print(f"  Fold {fold+1:02d} | {test_patient:12s} ({test_cancer:4s}) | R = {mean_r:.4f}")

# Summary
print("===== Global Baseline Results =====")
summary = []
for cancer, scores in results.items():
    mean_r = np.mean(scores)
    std_r  = np.std(scores)
    print(f"  {cancer:6s}  Mean R = {mean_r:.4f} ± {std_r:.4f}  (n={len(scores)} patients)")
    summary.append({"cancer_type": cancer, "mean_pearson_r": mean_r,
                    "std": std_r, "n_patients": len(scores)})

summary_df = pd.DataFrame(summary).sort_values("mean_pearson_r", ascending=False)
summary_df.to_csv(os.path.join(RESULTS_DIR, "global_baseline_results.csv"), index=False)
print(f"{RESULTS_DIR}/global_baseline_results.csv Saved")
print(summary_df.to_string(index=False))