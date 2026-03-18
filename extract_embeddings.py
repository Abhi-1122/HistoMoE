# extract_embeddings.py (fixed barcode decoding)
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import h5py
import pickle
from tqdm import tqdm

HEST_DIR   = "./hest_data"
EMBED_DIR  = "./embeddings"
META_PATH  = "./hest_data/subset_meta.csv"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

os.makedirs(EMBED_DIR, exist_ok=True)

def get_encoder():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval().to(DEVICE)
    return model

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def decode_barcodes(raw):
    result = []
    for b in raw:
        if isinstance(b, (np.ndarray, list)):
            b = b[0]                    
        if isinstance(b, bytes):
            result.append(b.decode('utf-8'))
        else:
            result.append(str(b))
    return result

def encode_from_h5(h5_path, encoder):
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        print(f"    .h5 keys: {keys}")

        img_key     = 'img'     if 'img'     in keys else 'imgs'
        barcode_key = 'barcode' if 'barcode' in keys else 'barcodes'

        imgs     = f[img_key][:]
        barcodes = decode_barcodes(f[barcode_key][:])

    print(f"Decoded barcode sample: {barcodes[:3]}")

    all_embeds = []
    for i in range(0, len(imgs), BATCH_SIZE):
        batch = imgs[i:i+BATCH_SIZE]
        tensors = torch.stack([
            transform(Image.fromarray(img.astype(np.uint8)))
            for img in batch
        ]).to(DEVICE)
        with torch.no_grad():
            embeds = encoder(tensors).cpu().numpy()
        all_embeds.append(embeds)

    return np.vstack(all_embeds), barcodes


meta    = pd.read_csv(META_PATH)
encoder = get_encoder()
print(f"Processing {len(meta)} samples on {DEVICE}")

for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Extracting embeddings"):
    sample_id   = row['id']
    cancer_type = row['oncotree_code']
    h5_path     = os.path.join(HEST_DIR, "patches", f"{sample_id}.h5")
    expr_path   = os.path.join(HEST_DIR, "st",      f"{sample_id}.h5ad")

    if not os.path.exists(h5_path):
        print(f"{sample_id} — patch file missing"); continue
    if not os.path.exists(expr_path):
        print(f"{sample_id} — expression file missing"); continue

    print(f"Processing {sample_id} ({cancer_type})")
    embeddings, barcodes = encode_from_h5(h5_path, encoder)

    out = {
        "sample_id"   : sample_id,
        "cancer_type" : cancer_type,
        "embeddings"  : embeddings,
        "barcodes"    : barcodes,
        "adata_path"  : expr_path,
    }
    with open(os.path.join(EMBED_DIR, f"{sample_id}.pkl"), "wb") as f:
        pickle.dump(out, f)
    print(f"{embeddings.shape[0]} patches, dim={embeddings.shape[1]}")

print("Extraction complete.")