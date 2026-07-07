#!/usr/bin/env python3
"""Baseline gene selection without cluster labels.

This script builds two rank lists:

* HVG: compute highly variable genes within each patient, then aggregate
  patient-wise rankings by frequency and average rank.
* HMG: combine all samples and select genes with the highest mean expression.

Both methods first restrict to genes shared by every input file, so the final
gene lists are valid for all patients.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "datasets" / "BC1" / "st"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "rank_list"
DEFAULT_MAPPING_CSV = PROJECT_ROOT / "datasets" / "BC1.csv"
TOP_N_GENES = 250
HVG_FLAVOR = "seurat"


def collect_h5ad_files(input_dir: Path) -> list[Path]:
	if input_dir.is_file() and input_dir.suffix == ".h5ad":
		return [input_dir]
	if not input_dir.is_dir():
		raise NotADirectoryError(input_dir)

	h5ad_files = sorted(input_dir.rglob("*.h5ad"))
	if not h5ad_files:
		raise FileNotFoundError(f"No .h5ad files found under {input_dir}")
	return h5ad_files


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


def load_adata_list(
	h5ad_files: list[Path],
	patient_by_sample: dict[str, str],
) -> list[sc.AnnData]:
	adata_list: list[sc.AnnData] = []
	for path in h5ad_files:
		adata = sc.read_h5ad(path)
		adata.obs_names_make_unique()
		if "sample_id" not in adata.obs.columns:
			adata.obs["sample_id"] = path.stem
		if "patient" not in adata.obs.columns and path.stem in patient_by_sample:
			adata.obs["patient"] = patient_by_sample[path.stem]
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


def get_patient_column(adata: sc.AnnData) -> str:
	for candidate in ("patient_id", "patient"):
		if candidate in adata.obs.columns:
			return candidate
	raise KeyError("No patient column found. Expected 'patient_id' or 'patient'.")


def get_expression_matrix(adata: sc.AnnData):
	if "exp" in adata.layers:
		return adata.layers["exp"]
	return adata.X


def dense_array(values) -> np.ndarray:
	if issparse(values):
		return values.toarray()
	return np.asarray(values)


def float_expression_matrix(values):
	if issparse(values):
		return values.astype(np.float32)
	return np.asarray(values, dtype=np.float32)


def aggregate_ranked_genes(patient_rankings: dict[str, list[str]]) -> pd.DataFrame:
	global_gene_counter = Counter()
	global_gene_ranks: dict[str, list[int]] = {}

	for genes in patient_rankings.values():
		patient_selected = set()
		for rank, gene in enumerate(genes):
			patient_selected.add(gene)
			global_gene_ranks.setdefault(gene, []).append(rank)
		for gene in patient_selected:
			global_gene_counter[gene] += 1

	summary_data = []
	for gene, freq in global_gene_counter.items():
		summary_data.append(
			{
				"Gene": gene,
				"Frequency": freq,
				"Avg_Rank": float(np.mean(global_gene_ranks[gene])),
			}
		)

	summary_df = pd.DataFrame(summary_data)
	if summary_df.empty:
		raise ValueError("No genes were selected.")

	return summary_df.sort_values(
		by=["Frequency", "Avg_Rank"],
		ascending=[False, True],
	).reset_index(drop=True)


def compute_hvg_gene_list(
	adata: sc.AnnData,
	patient_key: str,
	top_n: int = TOP_N_GENES,
) -> list[str]:
	patient_rankings: dict[str, list[str]] = {}
	patient_ids = adata.obs[patient_key].unique()

	print(f"Total Patients: {len(patient_ids)}")
	for pid in patient_ids:
		print(f"\n===== Processing Patient {pid} [HVG] =====")
		adata_sub = adata[adata.obs[patient_key] == pid].copy()
		adata_sub.X = float_expression_matrix(get_expression_matrix(adata_sub))
		sc.pp.highly_variable_genes(
			adata_sub,
			n_top_genes=min(top_n, adata_sub.n_vars),
			flavor=HVG_FLAVOR,
			inplace=True,
		)

		sort_column = "dispersions_norm"
		if sort_column not in adata_sub.var.columns:
			sort_column = "dispersions"
		if sort_column not in adata_sub.var.columns:
			sort_column = "means"

		result_df = pd.DataFrame(
			{
				"Gene": adata_sub.var_names,
				"Highly_Variable": adata_sub.var["highly_variable"].astype(bool).values,
				"Score": adata_sub.var[sort_column].values,
			}
		)
		result_df = result_df.replace([np.inf, -np.inf], np.nan).dropna()
		result_df = result_df[result_df["Highly_Variable"]]
		result_df = result_df.sort_values("Score", ascending=False).reset_index(drop=True)

		top_genes = result_df["Gene"].head(top_n).tolist()
		patient_rankings[str(pid)] = top_genes
		print(f"Selected {len(top_genes)} genes.")

	summary_df = aggregate_ranked_genes(patient_rankings)
	print("\nTop 20 HVG genes")
	print(summary_df.head(20).to_string(index=False))
	return summary_df["Gene"].head(top_n).tolist()


def compute_hmg_gene_list(adata: sc.AnnData, top_n: int = TOP_N_GENES) -> list[str]:
	x_matrix = dense_array(get_expression_matrix(adata))
	mean_exp = np.asarray(x_matrix.mean(axis=0)).ravel()
	result_df = pd.DataFrame({"Gene": adata.var_names, "Mean_Exp": mean_exp})
	result_df = result_df.replace([np.inf, -np.inf], np.nan).dropna()
	result_df = result_df.sort_values("Mean_Exp", ascending=False).reset_index(drop=True)

	print("\nTop 20 HMG genes")
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
) -> list[Path]:
	h5ad_files = collect_h5ad_files(input_dir)
	patient_by_sample = load_patient_mapping(mapping_csv)
	adata_list = load_adata_list(h5ad_files, patient_by_sample)
	common_genes = intersect_genes(adata_list)
	print(f"Common genes across all input files: {len(common_genes)}")

	combined = combine_adata(adata_list, common_genes)
	patient_key = get_patient_column(combined)

	output_paths = []
	hvg_genes = compute_hvg_gene_list(combined, patient_key)
	output_paths.append(save_gene_list(output_dir, "hvg", hvg_genes))

	hmg_genes = compute_hmg_gene_list(combined)
	output_paths.append(save_gene_list(output_dir, "hmg", hmg_genes))
	return output_paths


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compute HVG and HMG baseline gene lists without clusters."
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=DEFAULT_INPUT_DIR,
		help="Directory containing h5ad files or a single h5ad file.",
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
		help="CSV with sample_id and patient columns, used when h5ad lacks patient.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	process_files(args.input_dir, args.output_dir, args.mapping_csv)


if __name__ == "__main__":
	main()

'''
python clustering/baseline_selection.py \
	--input-dir datasets/BC1/st \
	--output-dir datasets/rank_list \
	--mapping-csv datasets/BC1.csv
'''
