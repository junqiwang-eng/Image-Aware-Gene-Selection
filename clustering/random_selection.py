#!/usr/bin/env python3
"""Random gene selection from genes shared by all samples.

This script reads feature .h5 or AnnData .h5ad files, keeps only genes that
are common to every sample, combines the samples, and randomly selects a fixed
number of genes from the combined AnnData gene index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scanpy as sc

from triplex_selection import (
	PROJECT_ROOT,
	TOP_N_GENES,
	collect_input_files,
	combine_adata,
	intersect_genes,
	load_input_files,
	load_patient_mapping,
)


DEFAULT_DATASET = "BC1"
DEFAULT_FEATURE = "conch"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "datasets" / DEFAULT_DATASET / "features" / DEFAULT_FEATURE
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / DEFAULT_DATASET / "rank_list"
DEFAULT_MAPPING_CSV = PROJECT_ROOT / "datasets" / f"{DEFAULT_DATASET}.csv"
DEFAULT_OUTPUT_NAME = "random"
DEFAULT_SEED = 42


def compute_random_gene_list(
	adata: sc.AnnData,
	top_n: int = TOP_N_GENES,
	seed: int | None = DEFAULT_SEED,
) -> list[str]:
	if top_n <= 0:
		raise ValueError("--top-n must be positive.")
	if top_n > adata.n_vars:
		raise ValueError(
			f"Requested {top_n} genes, but only {adata.n_vars} common genes are available."
		)

	rng = np.random.default_rng(seed)
	selected_indices = rng.choice(adata.n_vars, size=top_n, replace=False)
	selected_genes = [str(adata.var_names[index]) for index in selected_indices]

	print(f"Random seed: {seed}")
	print(f"Randomly selected genes: {len(selected_genes)}")
	print("\nFirst 20 random genes")
	for gene in selected_genes[:20]:
		print(gene)
	return selected_genes


def save_gene_list(output_dir: Path, name: str, genes: list[str]) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{name}.json"
	with open(output_path, "w", encoding="utf-8") as handle:
		json.dump(genes, handle, ensure_ascii=False, indent=4)
	print(f"Saved {len(genes)} genes to {output_path}")
	return output_path


def process_files(
	input_dir: Path,
	output_dir: Path,
	mapping_csv: Path | None = DEFAULT_MAPPING_CSV,
	output_name: str = DEFAULT_OUTPUT_NAME,
	top_n: int = TOP_N_GENES,
	seed: int | None = DEFAULT_SEED,
) -> Path:
	input_files = collect_input_files(input_dir)
	patient_by_sample = load_patient_mapping(mapping_csv)
	adata_list = load_input_files(input_files, patient_by_sample)
	common_genes = intersect_genes(adata_list)
	print(f"Common genes across all input files: {len(common_genes)}")

	combined = combine_adata(adata_list, common_genes)
	print(f"Combined samples: {combined.n_obs} spots x {combined.n_vars} common genes")
	genes = compute_random_gene_list(combined, top_n=top_n, seed=seed)
	return save_gene_list(output_dir, output_name, genes)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Randomly select genes from the common genes across all samples."
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
		help="Directory where the random gene list JSON file will be written.",
	)
	parser.add_argument(
		"--mapping-csv",
		type=Path,
		default=DEFAULT_MAPPING_CSV,
		help="CSV with sample_id and patient columns. Optional for files already containing obs labels.",
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
		help=f"Number of random genes to write. Default: {TOP_N_GENES}",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=DEFAULT_SEED,
		help=f"Random seed for reproducible selection. Default: {DEFAULT_SEED}",
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
		args.seed,
	)


if __name__ == "__main__":
	main()


'''
Example:
python clustering/random_selection.py \
	--input-dir datasets/CSCC/features/conch \
	--output-dir datasets/CSCC/rank_list \
	--mapping-csv datasets/CSCC.csv \
	--output-name random \
	--top-n 250 \
	--seed 42
'''
