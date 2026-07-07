#!/usr/bin/env python3
"""Scan Leiden resolutions for UNI features and summarize cluster counts.

The script follows the patient-aware clustering workflow used in
clustering.py/preprocessing.py, but fixes the input feature to UNI by default
and runs multiple Leiden resolutions. Cluster labels for every resolution are
stored in one combined h5ad file, and cluster counts per patient are saved as a
CSV and plotted as one line per patient.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA


DEFAULT_INPUT_DIR = Path(r"D:/Work/Gene select/datasets/combine/BC1")
DEFAULT_MAPPING_CSV = Path(r"D:/Work/Gene select/datasets/BC1.csv")
DEFAULT_OUTPUT_DIR = Path(r"D:/Work/Gene select/datasets/resolution/BC1")
DEFAULT_FEATURE = "uni"
DEFAULT_RESOLUTIONS = [0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
DEFAULT_PCA_COMPONENTS = 30
DEFAULT_NEIGHBORS = 15


def clean_name(name: str) -> str:
	return name.replace("-", "_").replace(" ", "_")


def resolution_label(resolution: float) -> str:
	return str(resolution).replace(".", "p")


def load_patient_mapping(csv_path: Path) -> dict[str, str]:
	mapping_df = pd.read_csv(csv_path)
	required_columns = {"sample_id", "patient"}
	missing = required_columns - set(mapping_df.columns)
	if missing:
		raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

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


def select_feature_key(adata: sc.AnnData, feature_name: str) -> str:
	target_name = clean_name(feature_name)
	feature_keys = feature_keys_from_adata(adata)
	for key in feature_keys:
		if clean_name(key.removeprefix("X_")) == target_name:
			return key

	available = [clean_name(key.removeprefix("X_")) for key in feature_keys]
	raise ValueError(f"Feature '{feature_name}' was not found. Available features: {available}")


def load_combined_adata(input_dir: Path, mapping_csv: Path) -> sc.AnnData:
	sample_to_patient = load_patient_mapping(mapping_csv)
	h5ad_files = collect_h5ad_files(input_dir)

	adata_by_sample: dict[str, sc.AnnData] = {}
	ordered_samples: list[str] = []

	for h5ad_path in h5ad_files:
		sample_id = h5ad_path.stem
		if sample_id not in sample_to_patient:
			print(f"Skipping {sample_id}: not found in {mapping_csv.name}")
			continue

		adata = sc.read_h5ad(h5ad_path)
		adata.obs_names_make_unique()
		adata.obs["sample_id"] = sample_id
		adata.obs["patient_id"] = sample_to_patient[sample_id]
		adata_by_sample[sample_id] = adata
		ordered_samples.append(sample_id)

	if not adata_by_sample:
		raise ValueError("No h5ad files matched the sample_id values in the mapping CSV.")

	return sc.concat(
		[adata_by_sample[sample_id] for sample_id in ordered_samples],
		join="inner",
		index_unique="-",
		merge="same",
	)


def add_resolution_clusters(
	combined: sc.AnnData,
	feature_key: str,
	resolutions: list[float],
) -> list[str]:
	feature_name = clean_name(feature_key.removeprefix("X_"))
	cluster_columns = [
		f"{feature_name}_resolution_{resolution_label(resolution)}_cluster"
		for resolution in resolutions
	]
	for cluster_key in cluster_columns:
		combined.obs[cluster_key] = pd.Series(index=combined.obs.index, dtype="object")

	for patient_id in combined.obs["patient_id"].unique():
		patient_mask = combined.obs["patient_id"] == patient_id
		patient_adata = combined[patient_mask].copy()

		if feature_key not in patient_adata.obsm:
			raise KeyError(f"{feature_key} not found in patient subset for {patient_id}")

		features = patient_adata.obsm[feature_key]
		n_components = min(DEFAULT_PCA_COMPONENTS, features.shape[0] - 1, features.shape[1])
		n_neighbors = min(DEFAULT_NEIGHBORS, features.shape[0] - 1)
		if n_components < 1:
			raise ValueError(f"Patient {patient_id} has too few spots for PCA.")
		if n_neighbors < 2:
			raise ValueError(f"Patient {patient_id} has too few spots for neighbors.")

		pca_key = f"{feature_key}_pca"
		neighbors_key = f"{feature_key}_neighbors"
		pca = PCA(n_components=n_components)
		patient_adata.obsm[pca_key] = pca.fit_transform(features)

		sc.pp.neighbors(
			patient_adata,
			n_neighbors=n_neighbors,
			use_rep=pca_key,
			key_added=neighbors_key,
		)

		for resolution, cluster_key in zip(resolutions, cluster_columns):
			sc.tl.leiden(
				patient_adata,
				neighbors_key=neighbors_key,
				key_added=cluster_key,
				resolution=resolution,
			)
			combined.obs.loc[patient_mask, cluster_key] = (
				patient_adata.obs[cluster_key].astype(str).values
			)

	for cluster_key in cluster_columns:
		combined.obs[cluster_key] = combined.obs[cluster_key].astype("category")

	return cluster_columns


def make_cluster_count_table(
	combined: sc.AnnData,
	cluster_columns: list[str],
	resolutions: list[float],
) -> pd.DataFrame:
	rows = []
	for resolution, cluster_key in zip(resolutions, cluster_columns):
		for patient_id in combined.obs["patient_id"].dropna().unique():
			patient_mask = combined.obs["patient_id"] == patient_id
			cluster_values = combined.obs.loc[patient_mask, cluster_key].dropna().astype(str)
			rows.append(
				{
					"patient_id": patient_id,
					"resolution": resolution,
					"cluster_column": cluster_key,
					"n_clusters": int(cluster_values.nunique()),
				}
			)
	return pd.DataFrame(rows)


def plot_cluster_counts(summary_df: pd.DataFrame, output_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(9, 6))
	for patient_id, patient_df in summary_df.groupby("patient_id", sort=True):
		patient_df = patient_df.sort_values("resolution")
		ax.plot(
			patient_df["resolution"],
			patient_df["n_clusters"],
			marker="o",
			linewidth=1.8,
			label=str(patient_id),
		)

	ax.set_xlabel("Resolution")
	ax.set_ylabel("Number of clusters")
	ax.set_title("UNI cluster count by Leiden resolution")
	ax.grid(True, linestyle="--", alpha=0.35)
	ax.legend(title="Patient", bbox_to_anchor=(1.02, 1), loc="upper left")
	fig.tight_layout()
	fig.savefig(output_path, dpi=300)
	plt.close(fig)


def process_resolutions(
	input_dir: Path,
	mapping_csv: Path,
	output_dir: Path,
	feature_name: str,
	resolutions: list[float],
) -> tuple[Path, Path, Path]:
	output_dir.mkdir(parents=True, exist_ok=True)

	combined = load_combined_adata(input_dir, mapping_csv)
	feature_key = select_feature_key(combined, feature_name)
	cluster_columns = add_resolution_clusters(combined, feature_key, resolutions)
	summary_df = make_cluster_count_table(combined, cluster_columns, resolutions)

	feature_label = clean_name(feature_name)
	h5ad_path = output_dir / f"{feature_label}_resolution_clusters.h5ad"
	csv_path = output_dir / f"{feature_label}_resolution_cluster_counts.csv"
	plot_path = output_dir / f"{feature_label}_resolution_cluster_counts.png"

	combined.uns["resolution_scan"] = {
		"feature": feature_label,
		"feature_key": feature_key,
		"resolutions": resolutions,
		"cluster_columns": cluster_columns,
		"pca_components": DEFAULT_PCA_COMPONENTS,
		"neighbors": DEFAULT_NEIGHBORS,
	}
	combined.write_h5ad(h5ad_path)
	summary_df.to_csv(csv_path, index=False)
	plot_cluster_counts(summary_df, plot_path)

	return h5ad_path, csv_path, plot_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run patient-aware UNI clustering across multiple Leiden resolutions."
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=DEFAULT_INPUT_DIR,
		help=f"Combined h5ad directory or file. Default: {DEFAULT_INPUT_DIR}",
	)
	parser.add_argument(
		"--mapping-csv",
		type=Path,
		default=DEFAULT_MAPPING_CSV,
		help=f"CSV with sample_id and patient columns. Default: {DEFAULT_MAPPING_CSV}",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
	)
	parser.add_argument(
		"--feature",
		type=str,
		default=DEFAULT_FEATURE,
		help=f"Feature name to use. Default: {DEFAULT_FEATURE}",
	)
	parser.add_argument(
		"--resolutions",
		nargs="+",
		type=float,
		default=DEFAULT_RESOLUTIONS,
		help="Leiden resolutions to scan.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	h5ad_path, csv_path, plot_path = process_resolutions(
		input_dir=args.input_dir,
		mapping_csv=args.mapping_csv,
		output_dir=args.output_dir,
		feature_name=args.feature,
		resolutions=args.resolutions,
	)
	print(f"Saved clustered h5ad: {h5ad_path}")
	print(f"Saved cluster counts: {csv_path}")
	print(f"Saved cluster count plot: {plot_path}")


if __name__ == "__main__":
	main()


r"""
PowerShell example:
python clustering/clustering_resolution.py `
	--input-dir "D:\Work\Gene select\datasets\combine\BC1" `
	--mapping-csv "D:\Work\Gene select\datasets\BC1.csv" `
	--feature "resnet50" `
	--output-dir "D:\Work\Gene select\datasets\resolution\BC1"
"""
