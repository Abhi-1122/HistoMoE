import os
from huggingface_hub import snapshot_download, hf_hub_download
import pandas as pd

SAVE_DIR     = "./hest_data"
CANCER_TYPES = ['SKCM', 'LUAD', 'PAAD', 'READ']
HF_TOKEN     = os.getenv("HF_TOKEN")

os.makedirs(SAVE_DIR, exist_ok=True)

meta_file = hf_hub_download(
    repo_id="MahmoodLab/hest",
    repo_type="dataset",
    filename="HEST_v1_3_0.csv",
    token=HF_TOKEN,
)
meta = pd.read_csv(meta_file)

print(f"Total samples: {len(meta)}")
print(f"Unique species values: {meta['species'].unique()}")
print(f"Top oncotree codes:{meta['oncotree_code'].value_counts().head(30)}")

# Filter — no species filter, just cancer type
subset = meta[meta['oncotree_code'].isin(CANCER_TYPES)]

print(f"Samples found for {CANCER_TYPES}: {len(subset)}")
print(subset[['id', 'oncotree_code', 'organ', 'species']].to_string())

# If still empty, the oncotree codes might be named differently
# Print all unique codes that contain any of your target strings
if len(subset) == 0:
    print("No matches found. Checking for similar oncotree codes...")
    for code in CANCER_TYPES:
        matches = meta[meta['oncotree_code'].str.contains(code, case=False, na=False)]
        print(f"  '{code}' → found {len(matches)} rows: {matches['oncotree_code'].unique()}")
    print("All available oncotree codes:")
    print(meta['oncotree_code'].dropna().unique())
    exit()

subset.to_csv(os.path.join(SAVE_DIR, "subset_meta.csv"), index=False)

sample_ids = subset['id'].tolist()

allow_patterns = []
for sid in sample_ids:
    allow_patterns.append(f"patches/{sid}.h5")
    allow_patterns.append(f"st/{sid}.h5ad")
    allow_patterns.append(f"metadata/{sid}.json")

print(f"Downloading {len(allow_patterns)} files...")

snapshot_download(
    repo_id="MahmoodLab/hest",
    repo_type="dataset",
    local_dir=SAVE_DIR,
    allow_patterns=allow_patterns,
    ignore_patterns=["cellvit_seg/*", "wsis/*", "thumbnails/*", "contours/*"],
    token=HF_TOKEN,
)

print(f"Download complete.")

# Sanity check
print("--- File check ---")
for sid in sample_ids:
    patch_ok = os.path.exists(os.path.join(SAVE_DIR, "patches", f"{sid}.h5"))
    expr_ok  = os.path.exists(os.path.join(SAVE_DIR, "st",      f"{sid}.h5ad"))
    cancer   = subset[subset['id'] == sid]['oncotree_code'].values[0]
    print(f"{sid}  ({cancer})  patches={patch_ok}  expr={expr_ok}")
