import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ScalarNormalizer:
    mean: float
    std: float

    def transform(self, value: float) -> float:
        return (value - self.mean) / self.std


def load_target_samples(target_dir: str) -> List[Dict]:
    all_samples = []
    for fname in sorted(os.listdir(target_dir)):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(target_dir, fname), "rb") as f:
            all_samples.append(pickle.load(f))
    return all_samples


def filter_samples_by_gene_count(samples: List[Dict], min_genes: int) -> Tuple[List[Dict], List[str]]:
    kept = [sample for sample in samples if len(sample["gene_names"]) >= min_genes]
    dropped = [sample["sample_id"] for sample in samples if len(sample["gene_names"]) < min_genes]
    return kept, dropped


def select_common_hvgs(samples: List[Dict], n_hvg: int) -> List[str]:
    gene_sets = [set(sample["gene_names"]) for sample in samples]
    common_genes = sorted(list(gene_sets[0].intersection(*gene_sets[1:])))
    if len(common_genes) == 0:
        raise RuntimeError("No common genes found across filtered samples.")

    pooled_expr = []
    for sample in samples:
        gene_idx = [sample["gene_names"].index(gene) for gene in common_genes]
        pooled_expr.append(sample["y"][:, gene_idx])

    pooled_expr = np.vstack(pooled_expr)
    gene_var = np.var(pooled_expr, axis=0)
    top_idx = np.argsort(gene_var)[::-1][:n_hvg]
    return [common_genes[i] for i in top_idx]


def build_spot_level_arrays(samples: List[Dict], selected_genes: List[str]):
    x_all, y_all = [], []
    cancer_labels = []
    sample_labels = []
    spot_indices = []

    for sample in samples:
        sample_gene_idx = [sample["gene_names"].index(g) for g in selected_genes]
        sample_y = sample["y"][:, sample_gene_idx]
        sample_x = sample["X"]
        n_spots = sample_x.shape[0]

        x_all.append(sample_x)
        y_all.append(sample_y)
        cancer_labels.extend([sample["cancer_type"]] * n_spots)
        sample_labels.extend([sample["sample_id"]] * n_spots)
        spot_indices.extend(np.arange(n_spots).tolist())

    x_all = np.vstack(x_all).astype(np.float32)
    y_all = np.vstack(y_all).astype(np.float32)

    return (
        x_all,
        y_all,
        np.array(cancer_labels),
        np.array(sample_labels),
        np.array(spot_indices),
    )


def load_sample_metadata(subset_meta_csv: str, metadata_dir: str, sample_ids: List[str]) -> Dict[str, Dict]:
    meta_df = pd.read_csv(subset_meta_csv)
    if "id" not in meta_df.columns:
        raise RuntimeError("subset_meta.csv is missing required column: id")
    meta_df = meta_df.set_index("id", drop=False)

    records = {}
    for sample_id in sample_ids:
        record = {}
        if sample_id in meta_df.index:
            row = meta_df.loc[sample_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            record.update(row.to_dict())

        json_path = os.path.join(metadata_dir, f"{sample_id}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                json_record = json.load(f)
            record.update(json_record)

        records[sample_id] = {
            "oncotree_code": str(record.get("oncotree_code", "UNKNOWN")),
            "st_technology": str(record.get("st_technology", "UNKNOWN")),
            "preservation_method": str(record.get("preservation_method", "UNKNOWN")),
            "magnification": _parse_magnification(record.get("magnification", np.nan)),
            "spots_under_tissue": _safe_float(record.get("spots_under_tissue", np.nan)),
        }

    return records


def _parse_magnification(raw_value) -> float:
    if raw_value is None:
        return np.nan
    text = str(raw_value).strip()
    if text == "" or text.lower() == "nan":
        return np.nan
    match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", text)
    if match is None:
        return np.nan
    return float(match.group(1))


def _safe_float(value) -> float:
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(float_value):
        return np.nan
    return float_value


class MetadataVectorizer:
    def __init__(self):
        self.cancer_classes: List[str] = []
        self.tech_classes: List[str] = []
        self.preservation_classes: List[str] = []
        self.spots_norm: ScalarNormalizer | None = None
        self.mag_norm: ScalarNormalizer | None = None

    @property
    def dim(self) -> int:
        return len(self.cancer_classes) + len(self.tech_classes) + len(self.preservation_classes) + 2

    def fit(self, sample_records: Dict[str, Dict], train_sample_ids: np.ndarray) -> None:
        train_ids = sorted(set(train_sample_ids.tolist()))

        cancers = []
        techs = []
        preservations = []
        spots_values = []
        mag_values = []

        for sample_id in train_ids:
            record = sample_records[sample_id]
            cancers.append(record["oncotree_code"])
            techs.append(record["st_technology"])
            preservations.append(record["preservation_method"])

            spots = record["spots_under_tissue"]
            if not np.isnan(spots):
                spots_values.append(np.log1p(max(0.0, spots)))

            mag = record["magnification"]
            if not np.isnan(mag):
                mag_values.append(mag)

        self.cancer_classes = sorted(set(cancers))
        self.tech_classes = sorted(set(techs))
        self.preservation_classes = sorted(set(preservations))

        self.spots_norm = _build_scalar_normalizer(spots_values)
        self.mag_norm = _build_scalar_normalizer(mag_values)

    def transform_sample(self, sample_id: str, sample_records: Dict[str, Dict]) -> np.ndarray:
        record = sample_records[sample_id]

        cancer_vec = _one_hot(record["oncotree_code"], self.cancer_classes)
        tech_vec = _one_hot(record["st_technology"], self.tech_classes)
        preservation_vec = _one_hot(record["preservation_method"], self.preservation_classes)

        spots = record["spots_under_tissue"]
        if np.isnan(spots):
            log_spots = 0.0
        else:
            log_spots = np.log1p(max(0.0, spots))
        norm_spots = self.spots_norm.transform(log_spots) if self.spots_norm is not None else 0.0

        mag = record["magnification"]
        if np.isnan(mag):
            norm_mag = 0.0
        else:
            norm_mag = self.mag_norm.transform(mag) if self.mag_norm is not None else 0.0

        return np.concatenate(
            [
                cancer_vec,
                tech_vec,
                preservation_vec,
                np.array([norm_spots, norm_mag], dtype=np.float32),
            ]
        ).astype(np.float32)

    def transform_spot_labels(self, sample_labels: np.ndarray, sample_records: Dict[str, Dict]) -> np.ndarray:
        out = np.zeros((len(sample_labels), self.dim), dtype=np.float32)
        for sample_id in np.unique(sample_labels):
            row = self.transform_sample(sample_id, sample_records)
            out[sample_labels == sample_id] = row
        return out


def _one_hot(value: str, classes: List[str]) -> np.ndarray:
    vec = np.zeros(len(classes), dtype=np.float32)
    if value in classes:
        vec[classes.index(value)] = 1.0
    return vec


def _build_scalar_normalizer(values: List[float]) -> ScalarNormalizer:
    if len(values) == 0:
        return ScalarNormalizer(mean=0.0, std=1.0)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        std = 1.0
    return ScalarNormalizer(mean=mean, std=std)


def build_cancer_label_mapping(sample_records: Dict[str, Dict], train_sample_ids: np.ndarray):
    train_ids = sorted(set(train_sample_ids.tolist()))
    train_cancers = sorted(set(sample_records[sample_id]["oncotree_code"] for sample_id in train_ids))
    return {cancer: idx for idx, cancer in enumerate(train_cancers)}


def map_spot_cancer_labels(
    sample_labels: np.ndarray,
    sample_records: Dict[str, Dict],
    cancer_to_idx: Dict[str, int],
) -> np.ndarray:
    labels = np.zeros(len(sample_labels), dtype=np.int64)
    for i, sample_id in enumerate(sample_labels):
        cancer = sample_records[sample_id]["oncotree_code"]
        labels[i] = cancer_to_idx[cancer]
    return labels
