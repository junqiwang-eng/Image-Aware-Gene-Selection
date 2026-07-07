#!/usr/bin/env python3
"""Split combined AnnData files into one feature HDF5 file per feature source.

Combined .h5ad files produced by clustering/combination.py can contain multiple
feature embeddings in ``adata.obsm``. This script writes those embeddings back to
feature-specific .h5 files while keeping expression, counts, coordinates, genes,
sample, and patient metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse


DEFAULT_INPUT_DIR = Path(r"D:/Work/Gene select/datasets/combine/BC1")
DEFAULT_MAPPING_CSV = Path(r"D:/Work/Gene select/datasets/BC1.csv")
DEFAULT_OUTPUT_DIR = Path(r"D:/Work/Gene select/datasets/features/BC1")


def clean_name(name: str) -> str:
	return name.replace("-", "_").replace(" ", "_")


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


def feature_keys_from_adata(adata: sc.AnnData) -> list[tuple[str, str]]:
	"""Return ``(feature_name, obsm_key)`` pairs in a stable source order."""
	feature_sources = adata.uns.get("feature_sources")
	if isinstance(feature_sources, dict) and feature_sources:
		pairs = []
		for source_name in feature_sources.keys():
			feature_name = clean_name(str(source_name))
			obsm_key = f"X_{feature_name}"
			if obsm_key not in adata.obsm:
				raise KeyError(f"{obsm_key} listed in adata.uns['feature_sources'] but missing from adata.obsm")
			pairs.append((feature_name, obsm_key))
		return pairs

	pairs = []
	for obsm_key in sorted(adata.obsm_keys()):
		if obsm_key.startswith("X_") and not obsm_key.endswith("_pca"):
			pairs.append((clean_name(obsm_key.removeprefix("X_")), obsm_key))
	if not pairs:
		raise ValueError("No feature embeddings found in adata.obsm. Expected keys like 'X_conch'.")
	return pairs


def dense_array(values) -> np.ndarray:
	if issparse(values):
		return values.toarray()
	return np.asarray(values)


def string_array(values) -> np.ndarray:
	return np.asarray(list(values), dtype=h5py.string_dtype(encoding="utf-8"))


def write_feature_h5(
	output_path: Path,
	adata: sc.AnnData,
	feature_key: str,
	sample_id: str,
	patient: str,
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	exp = dense_array(adata.X)
	count = dense_array(adata.layers["counts"]) if "counts" in adata.layers else exp
	if "spatial" not in adata.obsm:
		raise KeyError("adata.obsm['spatial'] is required to write coords.")

	with h5py.File(output_path, "w") as handle:
		handle.create_dataset("barcode", data=string_array(adata.obs_names))
		handle.create_dataset("coords", data=np.asarray(adata.obsm["spatial"]))
		handle.create_dataset("count", data=count)
		handle.create_dataset("exp", data=exp)
		handle.create_dataset("features", data=np.asarray(adata.obsm[feature_key]))
		handle.create_dataset("gene_list", data=string_array(adata.var_names))
		handle.create_dataset("sample", data=sample_id, dtype=h5py.string_dtype(encoding="utf-8"))
		handle.create_dataset("patient", data=patient, dtype=h5py.string_dtype(encoding="utf-8"))


def divide_files(input_dir: Path, mapping_csv: Path, output_dir: Path) -> list[Path]:
	sample_to_patient = load_patient_mapping(mapping_csv)
	h5ad_files = collect_h5ad_files(input_dir)
	output_paths: list[Path] = []

	for h5ad_path in h5ad_files:
		sample_id = h5ad_path.stem
		if sample_id not in sample_to_patient:
			print(f"Skipping {sample_id}: not found in {mapping_csv.name}")
			continue

		adata = sc.read_h5ad(h5ad_path)
		adata.obs_names_make_unique()
		patient = sample_to_patient[sample_id]
		adata.obs["sample"] = sample_id
		adata.obs["patient"] = patient

		for feature_name, feature_key in feature_keys_from_adata(adata):
			output_path = output_dir / feature_name / f"{sample_id}.h5"
			write_feature_h5(output_path, adata, feature_key, sample_id, patient)
			output_paths.append(output_path)

	if not output_paths:
		raise ValueError("No feature .h5 files were written.")
	return output_paths


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Split combined h5ad files into feature-specific h5 files."
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=DEFAULT_INPUT_DIR,
		help="Directory containing combined .h5ad files, or one .h5ad file.",
	)
	parser.add_argument(
		"--mapping-csv",
		type=Path,
		default=DEFAULT_MAPPING_CSV,
		help="CSV file with sample_id and patient columns.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Output root. Feature folders are created under this directory.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	divide_files(args.input_dir, args.mapping_csv, args.output_dir)


if __name__ == "__main__":
	main()


"""
PowerShell example:
python clustering/divide.py `
	--input-dir "D:\\Work\\Gene select\\datasets\\combine\\BC1" `
	--mapping-csv "D:\\Work\\Gene select\\datasets\\BC1.csv" `
	--output-dir "D:\\Work\\Gene select\\datasets\\features\\BC1"
"""
