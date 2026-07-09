#!/usr/bin/env python3
"""Stem-style HMHVG gene selection.

This script mirrors the triplex HMHVG preprocessing:

1. Read feature .h5 or AnnData .h5ad files.
2. Keep genes common to all samples.
3. For each sample, normalize counts, log1p transform, and select top 2000 HVGs.
4. Union the sample-wise HVGs.
5. On the full raw-count matrix, rank union genes by mean counts and count std.
6. Select the intersection of equal-sized top-k mean and std lists.

If --intersection-top-k is omitted, the script finds the smallest equal top-k
whose intersection contains at least --top-n genes, then writes the first
--top-n genes in mean-rank order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from triplex_selection import (
	PROJECT_ROOT,
	TOP_N_GENES,
	HMHVG_SAMPLE_TOP_N,
	add_sample_patient_obs,
	collect_input_files,
	combine_adata,
	dense_array,
	float_matrix,
	get_counts_matrix,
	get_sample_column,
	intersect_genes,
	load_input_files,
	load_patient_mapping,
)


DEFAULT_DATASET = "BC2"
DEFAULT_FEATURE = "conch"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "datasets" / DEFAULT_DATASET / "features" / DEFAULT_FEATURE
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / DEFAULT_DATASET / "rank_list"
DEFAULT_MAPPING_CSV = PROJECT_ROOT / "datasets" / DEFAULT_DATASET / f"{DEFAULT_DATASET}.csv"
DEFAULT_OUTPUT_NAME = "stem_hmhvg"


def compute_sample_hvg_union(
	adata: sc.AnnData,
	sample_top_n: int = HMHVG_SAMPLE_TOP_N,
) -> list[str]:
	sample_key = get_sample_column(adata)
	sample_ids = adata.obs[sample_key].unique()
	union_genes: set[str] = set()

	for sample_id in sample_ids:
		print(f"\n===== Processing sample {sample_id} [Stem HMHVG sample HVG] =====")
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
		raise ValueError("No sample-wise HVGs were selected.")

	return [gene for gene in adata.var_names if gene in union_genes]


def smallest_equal_top_k_for_intersection(
	mean_order: list[str],
	std_order: list[str],
	target_n: int,
) -> int:
	if target_n <= 0:
		raise ValueError("--top-n must be positive.")
	if len(mean_order) != len(std_order):
		raise ValueError("Mean and std rankings must have the same length.")

	low = min(target_n, len(mean_order))
	high = len(mean_order)
	if len(set(mean_order[:high]) & set(std_order[:high])) < target_n:
		return high

	while low < high:
		mid = (low + high) // 2
		intersection_size = len(set(mean_order[:mid]) & set(std_order[:mid]))
		if intersection_size >= target_n:
			high = mid
		else:
			low = mid + 1
	return low


def select_intersection_genes(
	gene_means: pd.Series,
	gene_stds: pd.Series,
	top_n: int,
	intersection_top_k: int | None = None,
) -> tuple[list[str], int, pd.DataFrame]:
	mean_order = gene_means.sort_values(ascending=False).index.tolist()
	std_order = gene_stds.sort_values(ascending=False).index.tolist()

	if intersection_top_k is None:
		intersection_top_k = smallest_equal_top_k_for_intersection(
			mean_order,
			std_order,
			top_n,
		)
	else:
		intersection_top_k = min(intersection_top_k, len(mean_order))

	intersection = set(mean_order[:intersection_top_k]) & set(std_order[:intersection_top_k])
	selected_genes = [gene for gene in mean_order if gene in intersection][:top_n]

	mean_ranks = gene_means.rank(ascending=False)
	std_ranks = gene_stds.rank(ascending=False)
	result_df = pd.DataFrame(
		{
			"Gene": selected_genes,
			"Mean_Rank": mean_ranks.loc[selected_genes].values,
			"Std_Rank": std_ranks.loc[selected_genes].values,
			"Mean_Counts": gene_means.loc[selected_genes].values,
			"Std_Counts": gene_stds.loc[selected_genes].values,
		}
	)
	return selected_genes, intersection_top_k, result_df


def compute_stem_hmhvg_gene_list(
	adata: sc.AnnData,
	top_n: int = TOP_N_GENES,
	sample_top_n: int = HMHVG_SAMPLE_TOP_N,
	intersection_top_k: int | None = None,
) -> list[str]:
	union_genes = compute_sample_hvg_union(adata, sample_top_n=sample_top_n)
	counts = dense_array(get_counts_matrix(adata[:, union_genes]))
	all_count_df = pd.DataFrame(counts, columns=union_genes)
	gene_means = all_count_df.mean(axis=0)
	gene_stds = all_count_df.std(axis=0)

	selected_genes, used_top_k, result_df = select_intersection_genes(
		gene_means,
		gene_stds,
		top_n=top_n,
		intersection_top_k=intersection_top_k,
	)

	print(f"\nMean/std equal top-k used: {used_top_k}")
	print(f"Intersection output genes: {len(selected_genes)}")
	print("\nTop 20 Stem HMHVG genes")
	print(result_df.head(20).to_string(index=False))
	return selected_genes


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
	output_name: str = DEFAULT_OUTPUT_NAME,
	top_n: int = TOP_N_GENES,
	sample_top_n: int = HMHVG_SAMPLE_TOP_N,
	intersection_top_k: int | None = None,
) -> Path:
	input_files = collect_input_files(input_dir)
	patient_by_sample = load_patient_mapping(mapping_csv)
	adata_list = load_input_files(input_files, patient_by_sample)
	common_genes = intersect_genes(adata_list)
	print(f"Common genes across all input files: {len(common_genes)}")

	combined = combine_adata(adata_list, common_genes)
	genes = compute_stem_hmhvg_gene_list(
		combined,
		top_n=top_n,
		sample_top_n=sample_top_n,
		intersection_top_k=intersection_top_k,
	)
	return save_gene_list(output_dir, output_name, genes)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compute stem-style HMHVG gene list.")
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
		help="Directory where the gene list JSON file will be written.",
	)
	parser.add_argument(
		"--mapping-csv",
		type=Path,
		default=DEFAULT_MAPPING_CSV,
		help="CSV with sample_id and patient columns.",
	)
	parser.add_argument(
		"--output-name",
		type=str,
		default=DEFAULT_OUTPUT_NAME,
		help="Output JSON stem without .json.",
	)
	parser.add_argument(
		"--top-n",
		type=int,
		default=TOP_N_GENES,
		help=f"Number of output genes. Default: {TOP_N_GENES}",
	)
	parser.add_argument(
		"--sample-top-n",
		type=int,
		default=HMHVG_SAMPLE_TOP_N,
		help=f"Sample-wise HVG count before union. Default: {HMHVG_SAMPLE_TOP_N}",
	)
	parser.add_argument(
		"--intersection-top-k",
		type=int,
		default=None,
		help="Fixed equal top-k from mean/std rankings. If omitted, auto-finds the smallest k giving at least --top-n intersected genes.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	process_files(
		args.input_dir,
		args.output_dir,
		args.mapping_csv,
		args.output_name,
		args.top_n,
		args.sample_top_n,
		args.intersection_top_k,
	)


if __name__ == "__main__":
	main()


'''
BC2 example:
python clustering/stem_selection.py \
	--input-dir datasets/BC2/features/conch \
	--output-dir datasets/BC2/rank_list \
	--mapping-csv datasets/BC2/BC2.csv \
	--output-name stem_hmhvg \
	--top-n 250

Use a fixed equal top-k, for example top 241 from each list:
python clustering/stem_selection.py \
	--input-dir datasets/BC2/features/uni \
	--output-dir datasets/BC2/rank_list \
	--mapping-csv datasets/BC2/BC2.csv \
	--intersection-top-k 280
'''
