#!/usr/bin/env python3
"""Patient-aware clustering of spots using gene expression.

This is the gene-expression ablation counterpart of
``clustering/clustering.py``. It keeps the same patient grouping and
PCA -> neighbors -> Leiden parameters, but uses ``AnnData.layers["exp"]``
instead of an image-feature embedding. Results are written as one h5ad per sample while
preserving expression/count layers, spatial coordinates, gene names, and the
gene-derived cluster labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA


DEFAULT_PCA_COMPONENTS = 50
DEFAULT_NEIGHBORS = 15
DEFAULT_RESOLUTION = 0.5


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


def gene_expression_matrix(adata: sc.AnnData) -> np.ndarray:
	"""Return the spot-by-gene expression matrix in PCA-compatible form."""
	if "exp" not in adata.layers:
		raise KeyError(
			'The input AnnData has no layers["exp"]. '
			"Use h5ad files produced by clustering/clustering.py."
		)
	expression = adata.layers["exp"]
	if sparse.issparse(expression):
		expression = expression.toarray()
	expression = np.asarray(expression)
	if expression.ndim != 2:
		raise ValueError(f"Expected a 2-D expression matrix, got {expression.shape}")
	if not np.isfinite(expression).all():
		raise ValueError("Gene expression matrix contains NaN or infinite values.")
	return expression


def cluster_patient_subset(
	adata: sc.AnnData, cluster_key: str, resolution: float
) -> None:
	pca_key = "X_gene_pca"
	neighbors_key = "gene_neighbors"

	expression = gene_expression_matrix(adata)
	n_components = min(
		DEFAULT_PCA_COMPONENTS,
		expression.shape[0] - 1,
		expression.shape[1],
	)
	n_neighbors = min(DEFAULT_NEIGHBORS, expression.shape[0] - 1)
	if n_components < 1:
		raise ValueError("Patient subset has too few spots for PCA.")
	if n_neighbors < 2:
		raise ValueError("Patient subset has too few spots for neighbors.")

	pca = PCA(n_components=n_components)
	adata.obsm[pca_key] = pca.fit_transform(expression)

	sc.pp.neighbors(
		adata,
		n_neighbors=n_neighbors,
		use_rep=pca_key,
		key_added=neighbors_key,
	)
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
	resolutions: list[float] | None = None,
) -> list[Path]:
	if resolutions is None:
		resolutions = [DEFAULT_RESOLUTION]

	sample_to_patient = load_patient_mapping(mapping_csv)
	h5ad_files = collect_h5ad_files(input_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	paths_by_patient: dict[str, list[Path]] = {}
	matched_paths: list[Path] = []
	for h5ad_path in h5ad_files:
		sample_id = h5ad_path.stem
		if sample_id not in sample_to_patient:
			print(f"Skipping {sample_id}: not found in {mapping_csv.name}")
			continue
		patient_id = sample_to_patient[sample_id]
		paths_by_patient.setdefault(patient_id, []).append(h5ad_path)
		matched_paths.append(h5ad_path)

	if not matched_paths:
		raise ValueError("No h5ad files matched the sample_id values in the mapping CSV.")

	cluster_columns = [
		f"gene_{resolution_label(resolution)}_cluster"
		for resolution in resolutions
	]

	# Match sc.concat(join="inner") from the image-feature workflow without
	# holding every sample (and all image embeddings) in memory at once.
	first_adata = sc.read_h5ad(matched_paths[0], backed="r")
	first_genes = list(first_adata.var_names)
	first_adata.file.close()
	common_genes = set(first_genes)
	for h5ad_path in matched_paths[1:]:
		adata = sc.read_h5ad(h5ad_path, backed="r")
		common_genes.intersection_update(adata.var_names)
		adata.file.close()
	ordered_common_genes = [gene for gene in first_genes if gene in common_genes]
	if not ordered_common_genes:
		raise ValueError("The matched h5ad files do not share any genes.")

	output_paths: list[Path] = []
	for patient_id, patient_paths in paths_by_patient.items():
		patient_samples: list[sc.AnnData] = []
		for h5ad_path in patient_paths:
			sample_id = h5ad_path.stem
			adata = sc.read_h5ad(h5ad_path)
			adata = adata[:, ordered_common_genes].copy()
			adata.obs_names_make_unique()
			adata.obs["sample_id"] = sample_id
			adata.obs["patient_id"] = patient_id

			spatial = adata.obsm["spatial"].copy() if "spatial" in adata.obsm else None
			adata.obsm.clear()
			if spatial is not None:
				adata.obsm["spatial"] = spatial
			adata.uns.clear()
			patient_samples.append(adata)

		patient_adata = sc.concat(
			patient_samples,
			join="inner",
			index_unique="-",
			merge="same",
		)
		for resolution, cluster_key in zip(resolutions, cluster_columns):
			cluster_patient_subset(patient_adata, cluster_key, resolution)
		for cluster_key in cluster_columns:
			patient_adata.obs[cluster_key] = patient_adata.obs[cluster_key].astype("category")

		for h5ad_path in patient_paths:
			sample_id = h5ad_path.stem
			sample_adata = patient_adata[
				patient_adata.obs["sample_id"] == sample_id
			].copy()

			spatial = (
				sample_adata.obsm["spatial"].copy()
				if "spatial" in sample_adata.obsm
				else None
			)
			sample_adata.obsm.clear()
			if spatial is not None:
				sample_adata.obsm["spatial"] = spatial

			sample_adata.uns.clear()
			for key in list(sample_adata.obsp.keys()):
				del sample_adata.obsp[key]

			sample_adata.obs = sample_adata.obs[cluster_columns].copy()
			sample_adata.obs["sample_id"] = sample_id
			sample_adata.obs["patient_id"] = patient_id
			sample_adata.layers["exp"] = sample_adata.X.copy()

			save_dir = output_dir / patient_id
			save_dir.mkdir(parents=True, exist_ok=True)
			save_path = save_dir / f"{sample_id}.h5ad"
			sample_adata.write_h5ad(save_path)
			output_paths.append(save_path)
			print(f"wrote {save_path}")

	return output_paths


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Cluster combined h5ad files by gene expression per patient."
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=Path("datasets/BC1/clustered/BC1"),
		help="Directory containing combined h5ad files or a single h5ad file.",
	)
	parser.add_argument(
		"--mapping-csv",
		type=Path,
		default=Path("datasets/BC1/BC1.csv"),
		help="CSV file with sample_id and patient columns.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("datasets/BC1/clustered/BC1_gene"),
		help="Directory where gene-clustered h5ad files will be written.",
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

	output_paths = process_files(
		args.input_dir,
		args.mapping_csv,
		args.output_dir,
		resolutions,
	)
	print(f"wrote {len(output_paths)} h5ad files to {args.output_dir}")


if __name__ == "__main__":
	main()


"""
Default BC1 run:

python ablation/clustering.py

Equivalent explicit command:

python ablation/clustering.py \
	--input-dir datasets/BC1/clustered/BC1 \
	--mapping-csv datasets/BC1/BC1.csv \
	--output-dir datasets/BC1/clustered/BC1_gene \
	--resolutions 0.5
"""
