#!/usr/bin/env python3
"""Baseline gene selection without cluster labels.

This script builds three rank-list options:

* HMG: combine all samples, keep common genes expressed in at least 10% of
  spots, and select genes with the highest mean counts.
* HVG: combine all samples, keep common genes expressed in at least 10% of
  spots, normalize counts to CPM 1e4, log1p transform, and use Scanpy HVG
  selection with sample_id as the batch key.
* HMHVG: compute sample-wise top 2000 HVGs, union them, then rank the union by
  mean-count rank plus count-standard-deviation rank on the full count matrix.

The input can be extracted feature .h5 files or AnnData .h5ad files. For
feature .h5 files, this script uses count/counts as raw counts for all
selection steps and keeps exp only as an optional layer. Patient labels are
assigned from the mapping CSV when they are missing from the file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "datasets" / "BC1" / "features" / "conch"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "BC1" / "rank_list"
DEFAULT_MAPPING_CSV = PROJECT_ROOT / "datasets" / "BC1.csv"
TOP_N_GENES = 250
HMHVG_SAMPLE_TOP_N = 2000
MIN_EXPRESSED_SPOT_FRACTION = 0.10
FEATURE_H5_REQUIRED_KEYS = ("barcode", "gene_list")
SELECTION_METHODS = ("hmg", "hvg", "hmhvg")
OUTPUT_NAME_BY_METHOD = {
	"hmg": "tri_hmg",
	"hvg": "tri_hvg",
	"hmhvg": "tri_hmhvg",
}


def collect_input_files(input_dir: Path) -> list[Path]:
	if input_dir.is_file() and input_dir.suffix in {".h5", ".h5ad"}:
		return [input_dir]
	if not input_dir.is_dir():
		raise NotADirectoryError(input_dir)

	input_files = sorted(
		path for path in input_dir.rglob("*") if path.suffix in {".h5", ".h5ad"}
	)
	if not input_files:
		raise FileNotFoundError(f"No .h5 or .h5ad files found under {input_dir}")
	return input_files


def decode_strings(values: np.ndarray) -> list[str]:
	arr = np.asarray(values)
	if arr.ndim > 1 and arr.shape[1] == 1:
		arr = arr[:, 0]

	decoded = []
	for value in arr.tolist():
		if isinstance(value, bytes):
			decoded.append(value.decode("utf-8"))
		elif isinstance(value, np.bytes_):
			decoded.append(value.decode("utf-8"))
		else:
			decoded.append(str(value))
	return decoded


def load_patient_mapping(csv_path: Path | None) -> dict[str, str]:
	if csv_path is None or not csv_path.exists():
		return {}

	patient_by_sample = {}
	with open(csv_path, "r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		required_columns = {"sample_id", "patient"}
		missing_columns = required_columns - set(reader.fieldnames or [])
		if missing_columns:
			raise ValueError(f"Missing columns in {csv_path}: {sorted(missing_columns)}")

		for row in reader:
			sample_id = row["sample_id"].strip()
			patient = row["patient"].strip()
			if sample_id and patient:
				patient_by_sample[sample_id] = patient
	return patient_by_sample


def add_sample_patient_obs(
	adata: sc.AnnData,
	path: Path,
	patient_by_sample: dict[str, str],
) -> sc.AnnData:
	if "sample_id" not in adata.obs.columns:
		adata.obs["sample_id"] = path.stem
	if "patient" not in adata.obs.columns and path.stem in patient_by_sample:
		adata.obs["patient"] = patient_by_sample[path.stem]
	return adata


def load_feature_h5(path: Path, patient_by_sample: dict[str, str]) -> sc.AnnData:
	try:
		import h5py
	except ImportError as exc:
		raise ImportError("Reading feature .h5 files requires h5py.") from exc

	with h5py.File(path, "r") as file:
		missing = [key for key in FEATURE_H5_REQUIRED_KEYS if key not in file]
		if missing:
			raise KeyError(f"{path} is missing required datasets: {missing}")

		barcodes = decode_strings(file["barcode"][:])
		if "count" in file:
			counts = np.asarray(file["count"][:], dtype=np.float32)
		elif "counts" in file:
			counts = np.asarray(file["counts"][:], dtype=np.float32)
		else:
			raise KeyError(f"{path} is missing raw counts dataset: count or counts")

		expression = np.asarray(file["exp"][:], dtype=np.float32) if "exp" in file else None
		genes = decode_strings(file["gene_list"][:])

	if counts.shape != (len(barcodes), len(genes)):
		raise ValueError(
			f"{path} has counts shape {counts.shape}, expected "
			f"({len(barcodes)}, {len(genes)}) from barcode and gene_list."
		)
	if expression is not None and expression.shape != counts.shape:
		raise ValueError(f"{path} has exp shape {expression.shape}, expected {counts.shape}.")

	obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
	var = pd.DataFrame(index=pd.Index(genes, name="gene"))
	adata = sc.AnnData(X=counts.copy(), obs=obs, var=var)
	adata.layers["counts"] = counts
	if expression is not None:
		adata.layers["exp"] = expression
	return add_sample_patient_obs(adata, path, patient_by_sample)


def load_input_files(
	input_files: list[Path],
	patient_by_sample: dict[str, str],
) -> list[sc.AnnData]:
	adata_list: list[sc.AnnData] = []
	for path in input_files:
		if path.suffix == ".h5":
			adata = load_feature_h5(path, patient_by_sample)
			adata_list.append(adata)
			continue

		adata = sc.read_h5ad(path)
		adata.obs_names_make_unique()
		adata = add_sample_patient_obs(adata, path, patient_by_sample)
		adata_list.append(adata)
	return adata_list


def intersect_genes(adata_list: list[sc.AnnData]) -> list[str]:
	if not adata_list:
		raise ValueError("At least one h5ad file is required.")

	common_genes = list(adata_list[0].var_names)
	common_gene_set = set(common_genes)
	for adata in adata_list[1:]:
		common_gene_set &= set(adata.var_names)

	common_genes = [gene for gene in common_genes if gene in common_gene_set]
	if not common_genes:
		raise ValueError("No common genes found across all input h5ad files.")
	return common_genes


def combine_adata(adata_list: list[sc.AnnData], common_genes: list[str]) -> sc.AnnData:
	aligned_adata_list = [adata[:, common_genes].copy() for adata in adata_list]
	return sc.concat(
		aligned_adata_list,
		join="inner",
		index_unique="-",
		merge="same",
	)


def get_sample_column(adata: sc.AnnData) -> str:
	for candidate in ("sample_id", "sample"):
		if candidate in adata.obs.columns:
			return candidate
	raise KeyError("No sample column found. Expected 'sample_id' or 'sample'.")


def get_counts_matrix(adata: sc.AnnData):
	if "counts" in adata.layers:
		return adata.layers["counts"]
	if "count" in adata.layers:
		return adata.layers["count"]
	raise KeyError("Raw counts are required. Expected adata.layers['counts'] or ['count'].")


def dense_array(values) -> np.ndarray:
	if issparse(values):
		return values.toarray()
	return np.asarray(values)


def float_matrix(values):
	if issparse(values):
		return values.astype(np.float32)
	return np.asarray(values, dtype=np.float32)


def filter_genes_by_spot_fraction(
	adata: sc.AnnData,
	min_fraction: float = MIN_EXPRESSED_SPOT_FRACTION,
) -> sc.AnnData:
	counts = get_counts_matrix(adata)
	if issparse(counts):
		expressed_spots = np.asarray((counts > 0).sum(axis=0)).ravel()
	else:
		expressed_spots = np.count_nonzero(np.asarray(counts) > 0, axis=0)

	min_spots = int(np.ceil(adata.n_obs * min_fraction))
	keep_mask = expressed_spots >= min_spots
	filtered = adata[:, keep_mask].copy()
	print(
		f"Filtered genes by expression in >= {min_fraction:.0%} spots "
		f"({min_spots}/{adata.n_obs}): {adata.n_vars} -> {filtered.n_vars}"
	)
	if filtered.n_vars == 0:
		raise ValueError("No genes remain after spot-expression filtering.")
	return filtered


def compute_hvg_gene_list(adata: sc.AnnData, top_n: int = TOP_N_GENES) -> list[str]:
	sample_key = get_sample_column(adata)
	adata_hvg = adata.copy()
	adata_hvg.X = float_matrix(get_counts_matrix(adata_hvg))
	sc.pp.normalize_total(adata_hvg, target_sum=1e4)
	sc.pp.log1p(adata_hvg)
	sc.pp.highly_variable_genes(
		adata_hvg,
		n_top_genes=min(top_n, adata_hvg.n_vars),
		batch_key=sample_key,
		inplace=True,
	)

	sort_columns = [
		column
		for column in ("highly_variable_nbatches", "dispersions_norm", "dispersions", "means")
		if column in adata_hvg.var.columns
	]
	result_df = pd.DataFrame({"Gene": adata_hvg.var_names})
	result_df["Highly_Variable"] = adata_hvg.var["highly_variable"].astype(bool).values
	for column in sort_columns:
		result_df[column] = adata_hvg.var[column].values
	result_df = result_df.replace([np.inf, -np.inf], np.nan).dropna()
	result_df = result_df[result_df["Highly_Variable"]]
	if sort_columns:
		ascending = [False] * len(sort_columns)
		result_df = result_df.sort_values(sort_columns, ascending=ascending)
	result_df = result_df.reset_index(drop=True)

	print("\nTop 20 HVG genes")
	print(result_df.head(20).to_string(index=False))
	return result_df["Gene"].head(top_n).tolist()


def compute_hmg_gene_list(adata: sc.AnnData, top_n: int = TOP_N_GENES) -> list[str]:
	counts = dense_array(get_counts_matrix(adata))
	mean_counts = np.asarray(counts.mean(axis=0)).ravel()
	result_df = pd.DataFrame({"Gene": adata.var_names, "Mean_Counts": mean_counts})
	result_df = result_df.replace([np.inf, -np.inf], np.nan).dropna()
	result_df = result_df.sort_values("Mean_Counts", ascending=False).reset_index(drop=True)

	print("\nTop 20 HMG genes")
	print(result_df.head(20).to_string(index=False))
	return result_df["Gene"].head(top_n).tolist()


def compute_hmhvg_gene_list(
	adata: sc.AnnData,
	top_n: int = TOP_N_GENES,
	sample_top_n: int = HMHVG_SAMPLE_TOP_N,
) -> list[str]:
	sample_key = get_sample_column(adata)
	sample_ids = adata.obs[sample_key].unique()
	union_genes: set[str] = set()

	for sample_id in sample_ids:
		print(f"\n===== Processing sample {sample_id} [HMHVG sample HVG] =====")
		adata_sample = adata[adata.obs[sample_key] == sample_id].copy()
		adata_sample.X = float_matrix(get_counts_matrix(adata_sample))
		sc.pp.normalize_total(adata_sample, inplace=True)
		sc.pp.log1p(adata_sample)
		sc.pp.highly_variable_genes(
			adata_sample,
			n_top_genes=min(sample_top_n, adata_sample.n_vars),
			inplace=True,
		)

		sample_genes = adata_sample.var_names[
			adata_sample.var["highly_variable"].astype(bool).values
		].tolist()
		union_genes.update(sample_genes)
		print(f"Selected {len(sample_genes)} sample HVGs; union size: {len(union_genes)}")

	if not union_genes:
		raise ValueError("No sample-wise HVGs were selected for hmhvg.")

	union_genes_ordered = [gene for gene in adata.var_names if gene in union_genes]
	counts = dense_array(get_counts_matrix(adata[:, union_genes_ordered]))
	all_count_df = pd.DataFrame(counts, columns=union_genes_ordered)
	gene_means = all_count_df.mean(axis=0)
	gene_stds = all_count_df.std(axis=0)
	mean_ranks = gene_means.rank(ascending=False)
	std_ranks = gene_stds.rank(ascending=False)
	combined_score = mean_ranks + std_ranks

	result_df = pd.DataFrame(
		{
			"Gene": combined_score.index,
			"Combined_Score": combined_score.values,
			"Mean_Counts": gene_means.loc[combined_score.index].values,
			"Std_Counts": gene_stds.loc[combined_score.index].values,
		}
	)
	result_df = result_df.replace([np.inf, -np.inf], np.nan).dropna()
	result_df = result_df.sort_values("Combined_Score").reset_index(drop=True)

	print("\nTop 20 HMHVG genes")
	print(result_df.head(20).to_string(index=False))
	return result_df["Gene"].head(top_n).tolist()


def save_gene_list(output_dir: Path, name: str, genes: list[str]) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{name}.json"
	with open(output_path, "w", encoding="utf-8") as handle:
		json.dump(genes, handle, ensure_ascii=False, indent=4)
	print(f"Saved Top{len(genes)} genes to {output_path}")
	return output_path


def process_files(
	input_dir: Path,
	output_dir: Path,
	mapping_csv: Path | None = DEFAULT_MAPPING_CSV,
	selection_methods: list[str] | None = None,
	top_n: int = TOP_N_GENES,
) -> list[Path]:
	if selection_methods is None:
		selection_methods = ["hmg", "hvg", "hmhvg"]

	input_files = collect_input_files(input_dir)
	patient_by_sample = load_patient_mapping(mapping_csv)
	adata_list = load_input_files(input_files, patient_by_sample)
	common_genes = intersect_genes(adata_list)
	print(f"Common genes across all input files: {len(common_genes)}")

	combined = combine_adata(adata_list, common_genes)
	filtered_combined = None

	output_paths = []
	for method in selection_methods:
		if method == "hmg":
			if filtered_combined is None:
				filtered_combined = filter_genes_by_spot_fraction(combined)
			genes = compute_hmg_gene_list(filtered_combined, top_n=top_n)
		elif method == "hvg":
			if filtered_combined is None:
				filtered_combined = filter_genes_by_spot_fraction(combined)
			genes = compute_hvg_gene_list(filtered_combined, top_n=top_n)
		elif method == "hmhvg":
			genes = compute_hmhvg_gene_list(combined, top_n=top_n)
		else:
			raise ValueError(f"Unknown selection method: {method}")
		output_paths.append(save_gene_list(output_dir, OUTPUT_NAME_BY_METHOD[method], genes))

	return output_paths


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compute HVG and HMG baseline gene lists without clusters."
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=DEFAULT_INPUT_DIR,
		help="Directory containing .h5/.h5ad files or a single .h5/.h5ad file.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory where gene list JSON files will be written.",
	)
	parser.add_argument(
		"--mapping-csv",
		type=Path,
		default=DEFAULT_MAPPING_CSV,
		help="CSV with sample_id and patient columns.",
	)
	parser.add_argument(
		"--selection-method",
		type=str,
		nargs="+",
		choices=SELECTION_METHODS,
		default=["hmg", "hvg", "hmhvg"],
		help="Selection method(s) to run. Choices: hmg, hvg, hmhvg.",
	)
	parser.add_argument(
		"--top-n",
		type=int,
		default=TOP_N_GENES,
		help=f"Number of genes to write per method. Default: {TOP_N_GENES}",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	process_files(
		args.input_dir,
		args.output_dir,
		args.mapping_csv,
		args.selection_method,
		args.top_n,
	)


if __name__ == "__main__":
	main()

'''
python clustering/triplex_selection.py \
	--input-dir datasets/CSCC/features/uni \
	--output-dir datasets/CSCC/rank_list \
	--mapping-csv datasets/CSCC.csv \
	--selection-method hmg hvg hmhvg
'''
