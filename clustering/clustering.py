#!/usr/bin/env python3
"""Patient-aware clustering for combined AnnData files.

This script reads h5ad files produced by clustering/combination.py, attaches
sample/patient labels from a CSV mapping, runs the existing PCA -> neighbors ->
Leiden workflow for each feature embedding on a per-patient basis, and
writes a minimal h5ad that keeps expression/count layers, spatial coordinates,
gene names, sample and patient labels, and the resulting cluster labels.

The CSV mapping is expected to have at least two columns:
- sample_id: file stem of the h5ad file
- patient: patient label for that sample

Default cluster parameters match the earlier notebook workflow:
- PCA n_components = 30
- neighbors n_neighbors = 15
- Leiden resolutions = 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA


DEFAULT_PCA_COMPONENTS = 50
DEFAULT_NEIGHBORS = 15
DEFAULT_RESOLUTION = 0.5


def clean_name(name: str) -> str:
	return name.replace("-", "_").replace(" ", "_")


def resolution_label(resolution: float) -> str:
	return str(resolution).replace(".", "p")


def load_patient_mapping(csv_path: Path) -> dict[str, str]:
	mapping_df = pd.read_csv(csv_path)
	required_columns = {"sample_id", "patient"}
	missing = required_columns - set(mapping_df.columns)
	if missing:
		raise ValueError(
			f"{csv_path} is missing required columns: {sorted(missing)}"
		)

	mapping = {}
	for _, row in mapping_df.iterrows():
		sample_id = str(row["sample_id"]).strip()
		patient = str(row["patient"]).strip()
		if sample_id:
			mapping[sample_id] = patient
	return mapping


def collect_h5ad_files(input_dir: Path) -> list[Path]:
	if input_dir.is_file() and input_dir.suffix == ".h5ad":
		return [input_dir]
	if not input_dir.is_dir():
		raise NotADirectoryError(input_dir)

	h5ad_files = sorted(input_dir.rglob("*.h5ad"))
	if not h5ad_files:
		raise FileNotFoundError(f"No .h5ad files found under {input_dir}")
	return h5ad_files


def feature_keys_from_adata(adata: sc.AnnData) -> list[str]:
	feature_sources = adata.uns.get("feature_sources")
	if isinstance(feature_sources, dict) and feature_sources:
		return [f"X_{clean_name(name)}" for name in feature_sources.keys()]

	feature_keys = [key for key in adata.obsm_keys() if key.startswith("X_")]
	feature_keys = [key for key in feature_keys if not key.endswith("_pca")]
	return sorted(feature_keys)


def select_feature_keys(adata: sc.AnnData, feature_names: list[str] | None) -> list[str]:
	feature_keys = feature_keys_from_adata(adata)
	if feature_names is None:
		return feature_keys

	available_by_name = {
		clean_name(key.removeprefix("X_")): key
		for key in feature_keys
	}
	selected_keys = []
	missing_names = []
	for feature_name in feature_names:
		target_name = clean_name(feature_name)
		if target_name in available_by_name:
			selected_keys.append(available_by_name[target_name])
		else:
			missing_names.append(feature_name)

	if missing_names:
		raise ValueError(
			f"Features not found: {missing_names}. "
			f"Available features: {sorted(available_by_name)}"
		)
	return selected_keys


def cluster_patient_subset(
	adata, feature_key: str, cluster_key: str, resolution: float
) -> None:
	pca_key = f"{feature_key}_pca"
	neighbors_key = f"{feature_key}_neighbors"

	features = adata.obsm[feature_key]
	n_components = min(DEFAULT_PCA_COMPONENTS, features.shape[0] - 1, features.shape[1])
	n_neighbors = min(DEFAULT_NEIGHBORS, features.shape[0] - 1)
	if n_components < 1:
		raise ValueError("Patient subset has too few spots for PCA.")
	if n_neighbors < 2:
		raise ValueError("Patient subset has too few spots for neighbors.")

	pca = PCA(n_components=n_components)
	adata.obsm[pca_key] = pca.fit_transform(features)

	sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=pca_key, key_added=neighbors_key)
	sc.tl.leiden(
		adata,
		neighbors_key=neighbors_key,
		key_added=cluster_key,
		resolution=resolution,
	)


def process_files(
	input_dir: Path,
	mapping_csv: Path,
	output_dir: Path,
	feature_names: list[str] | None = None,
	resolutions: list[float] | None = None,
) -> list[Path]:
	if resolutions is None:
		resolutions = [DEFAULT_RESOLUTION]

	sample_to_patient = load_patient_mapping(mapping_csv)
	h5ad_files = collect_h5ad_files(input_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	adata_by_sample: dict[str, sc.AnnData] = {}
	ordered_samples: list[str] = []

	for h5ad_path in h5ad_files:
		sample_id = h5ad_path.stem
		if sample_id not in sample_to_patient:
			print(f"Skipping {sample_id}: not found in {mapping_csv.name}")
			continue

		adata = sc.read_h5ad(h5ad_path)
		adata.obs_names_make_unique()
		patient_id = sample_to_patient[sample_id]
		adata.obs["sample_id"] = sample_id
		adata.obs["patient_id"] = patient_id
		adata_by_sample[sample_id] = adata
		ordered_samples.append(sample_id)

	if not adata_by_sample:
		raise ValueError("No h5ad files matched the sample_id values in the mapping CSV.")

	combined = sc.concat(
		[adata_by_sample[sample_id] for sample_id in ordered_samples],
		join="inner",
		index_unique="-",
		merge="same",
	)

	feature_keys = select_feature_keys(combined, feature_names)
	if not feature_keys:
		raise ValueError("No feature embeddings were found in the combined AnnData.")

	cluster_columns = []
	for feature_key in feature_keys:
		feature_name = clean_name(feature_key.removeprefix("X_"))
		feature_cluster_columns = [
			f"{feature_name}_{resolution_label(resolution)}_cluster"
			for resolution in resolutions
		]
		cluster_columns.extend(feature_cluster_columns)
		for cluster_key in feature_cluster_columns:
			combined.obs[cluster_key] = pd.Series(index=combined.obs.index, dtype="object")

		for patient_id in combined.obs["patient_id"].unique():
			patient_mask = combined.obs["patient_id"] == patient_id
			patient_adata = combined[patient_mask].copy()

			if feature_key not in patient_adata.obsm:
				raise KeyError(f"{feature_key} not found in patient subset for {patient_id}")

			for resolution, cluster_key in zip(resolutions, feature_cluster_columns):
				cluster_patient_subset(patient_adata, feature_key, cluster_key, resolution)
				combined.obs.loc[patient_mask, cluster_key] = (
					patient_adata.obs[cluster_key].astype(str).values
				)

		for cluster_key in feature_cluster_columns:
			combined.obs[cluster_key] = combined.obs[cluster_key].astype("category")

	output_paths: list[Path] = []
	for sample_id in ordered_samples:
		sample_adata = combined[combined.obs["sample_id"] == sample_id].copy()
		patient_id = str(sample_adata.obs["patient_id"].iloc[0])

		spatial = sample_adata.obsm["spatial"].copy() if "spatial" in sample_adata.obsm else None
		sample_adata.obsm.clear()
		if spatial is not None:
			sample_adata.obsm["spatial"] = spatial

		sample_adata.uns.clear()

		for key in list(sample_adata.obsp.keys()):
			del sample_adata.obsp[key]

		keep_columns = cluster_columns
		sample_adata.obs = sample_adata.obs[keep_columns].copy()
		sample_adata.obs["sample_id"] = sample_id
		sample_adata.obs["patient_id"] = patient_id

		sample_adata.layers["exp"] = sample_adata.X.copy()

		save_dir = output_dir / patient_id
		save_dir.mkdir(parents=True, exist_ok=True)
		save_path = save_dir / f"{sample_id}.h5ad"

		sample_adata.write_h5ad(save_path)
		output_paths.append(save_path)

	return output_paths


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Attach patient/sample labels and cluster combined h5ad files by feature."
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		required=True,
		help="Directory containing combined h5ad files or a single h5ad file.",
	)
	parser.add_argument(
		"--mapping-csv",
		type=Path,
		required=True,
		help="CSV file with sample_id and patient columns.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		required=True,
		help="Directory where processed h5ad files will be written.",
	)
	parser.add_argument(
		"--features",
		nargs="+",
		default=None,
		help="Feature names to cluster, for example: uni conch resnet50. If omitted, all features are used.",
	)
	parser.add_argument(
		"--resolution",
		type=float,
		default=None,
		help="Single Leiden resolution. Kept for backward compatibility.",
	)
	parser.add_argument(
		"--resolutions",
		nargs="+",
		type=float,
		default=None,
		help=f"One or more Leiden resolutions. Default: {DEFAULT_RESOLUTION}",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.resolution is not None and args.resolutions is not None:
		raise ValueError("Use either --resolution or --resolutions, not both.")
	resolutions = args.resolutions
	if resolutions is None:
		resolutions = [args.resolution] if args.resolution is not None else None

	process_files(
		args.input_dir,
		args.mapping_csv,
		args.output_dir,
		args.features,
		resolutions,
	)


if __name__ == "__main__":
	main()


'''
python clustering/clustering.py \
     --input-dir datasets/BC1/combined_features \
     --mapping-csv datasets/BC1.csv \
     --output-dir datasets/BC1/clustered/BC1
'''

r'''
Linux example:
python clustering/clustering.py \
	--input-dir datasets/CSCC/combined_features \
	--mapping-csv datasets/CSCC/CSCC.csv \
	--output-dir datasets/CSCC/clustered/CSCC \
	--features conch uni \
	--resolutions 0.5

Specific features and resolution:
python clustering/clustering.py \
	--input-dir datasets/BC1/combined_features \
	--mapping-csv datasets/BC1/BC1.csv \
	--output-dir datasets/BC1/clustered/BC1 \
	--features conch \
	--resolutions 0.5
'''
