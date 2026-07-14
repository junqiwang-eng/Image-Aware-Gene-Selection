#!/usr/bin/env python3
"""DEG gene selection for clustered AnnData files.

This script mirrors ``clustering/selection.py``. The only substantive change
is the per-patient ranking metric: it uses Scanpy's Wilcoxon differential
expression scores instead of ANOVA F-scores. For every gene, its strongest
one-vs-rest cluster score is used as that patient's DEG score.

For a cluster column such as ``conch_0p5_cluster``, the output is named
``conch_deg_0p5.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


DEFAULT_INPUT_DIR = Path("datasets/BC1/clustered/BC1")
DEFAULT_OUTPUT_DIR = Path("datasets/BC1/rank_list")
MIN_SPOTS = 100
MAX_PVALS_ADJ = 0.05
TOP_N_PER_CLUSTER = 70
TOP_N_PER_PATIENT = 250


def clean_name(name: str) -> str:
	return name.replace("-", "_").replace(" ", "_")


def infer_resolution_label(input_dir: Path) -> str | None:
	name = input_dir.parent.name if input_dir.is_file() else input_dir.name
	match = re.search(r"(?:^|_)(\d+p\d+)$", clean_name(name))
	if match:
		return match.group(1)
	return None


def make_output_name(
	feature_source: str,
	selection_method: str,
	resolution_label: str | None,
) -> str:
	name_parts = [clean_name(feature_source), clean_name(selection_method)]
	if resolution_label:
		name_parts.append(clean_name(resolution_label))
	return "_".join(name_parts) + ".json"


def split_cluster_column(cluster_key: str) -> tuple[str, str | None]:
	base_name = clean_name(cluster_key.removesuffix("_cluster"))
	match = re.match(r"^(?P<feature>.+)_(?P<resolution>\d+p\d+)$", base_name)
	if match:
		return match.group("feature"), match.group("resolution")
	return base_name, None


def collect_h5ad_files(input_dir: Path) -> list[Path]:
	if input_dir.is_file() and input_dir.suffix == ".h5ad":
		return [input_dir]
	if not input_dir.is_dir():
		raise NotADirectoryError(input_dir)

	h5ad_files = sorted(input_dir.rglob("*.h5ad"))
	if not h5ad_files:
		raise FileNotFoundError(f"No .h5ad files found under {input_dir}")
	return h5ad_files


def load_adata_list(h5ad_files: list[Path]) -> list[sc.AnnData]:
	adata_list: list[sc.AnnData] = []
	for path in h5ad_files:
		adata = sc.read_h5ad(path)
		adata.obs_names_make_unique()
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


def combine_adata(
	adata_list: list[sc.AnnData], common_genes: list[str]
) -> sc.AnnData:
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


def get_cluster_columns(adata: sc.AnnData) -> list[str]:
	cluster_columns = [
		column
		for column in adata.obs.columns
		if column.startswith("conch_") and column.endswith("_cluster")
	]
	if not cluster_columns:
		raise ValueError("No conch_*_cluster columns found in adata.obs.")
	return sorted(cluster_columns)


def compute_deg_gene_list(
	adata: sc.AnnData,
	cluster_key: str,
	patient_key: str,
) -> list[str]:
	global_gene_counter = Counter()
	global_gene_ranks: dict[str, list[int]] = {}
	patient_ids = adata.obs[patient_key].unique()

	print(f"Total Patients: {len(patient_ids)}")
	for pid in patient_ids:
		print(f"\n===== Processing Patient {pid} [{cluster_key}] =====")
		adata_sub = adata[adata.obs[patient_key] == pid].copy()
		cluster_counts = adata_sub.obs[cluster_key].value_counts()
		valid_clusters = cluster_counts[cluster_counts >= MIN_SPOTS].index.tolist()
		invalid_clusters = cluster_counts[cluster_counts < MIN_SPOTS].index.tolist()

		if invalid_clusters:
			print(f"⚠️ Remove small clusters: {invalid_clusters}")
		if len(valid_clusters) < 2:
			print(f"❌ Skip patient {pid}: less than 2 valid clusters")
			continue

		adata_filtered = adata_sub[
			adata_sub.obs[cluster_key].isin(valid_clusters)
		].copy()
		adata_filtered.obs[cluster_key] = (
			adata_filtered.obs[cluster_key].cat.remove_unused_categories()
		)
		sc.pp.filter_genes(
			adata_filtered,
			min_cells=max(1, int(len(adata_filtered) * 0.01)),
		)
		if "exp" not in adata_filtered.layers:
			raise KeyError(
				'The input AnnData has no layers["exp"] for DEG selection.'
			)

		sc.tl.rank_genes_groups(
			adata_filtered,
			groupby=cluster_key,
			method="wilcoxon",
			use_raw=False,
			layer="exp",
			n_genes=adata_filtered.n_vars,
			pts=False,
		)

		# Before merging clusters within a patient, retain at most the first 70
		# genes in Scanpy's DEG ranking whose adjusted p-value is below 0.05.
		# If a retained gene marks multiple clusters, keep its strongest positive
		# one-vs-rest Wilcoxon score to form one patient-level ranking.
		deg_result = adata_filtered.uns["rank_genes_groups"]
		gene_scores: dict[str, float] = {}
		for group in deg_result["names"].dtype.names:
			selected_in_cluster = 0
			for gene, score, pval_adj in zip(
				deg_result["names"][group],
				deg_result["scores"][group],
				deg_result["pvals_adj"][group],
			):
				score = float(score)
				pval_adj = float(pval_adj)
				if (
					not np.isfinite(score)
					or not np.isfinite(pval_adj)
					or score <= 0
					or pval_adj >= MAX_PVALS_ADJ
				):
					continue
				gene = str(gene)
				if score > gene_scores.get(gene, -np.inf):
					gene_scores[gene] = score
				selected_in_cluster += 1
				if selected_in_cluster >= TOP_N_PER_CLUSTER:
					break
			print(
				f"Cluster {group}: retained {selected_in_cluster} genes "
				f"with pvals_adj < {MAX_PVALS_ADJ}."
			)

		result_df = pd.DataFrame(
			gene_scores.items(), columns=["Gene", "DEG_score"]
		)
		result_df = result_df.sort_values(
			"DEG_score", ascending=False
		).reset_index(drop=True)
		top_df = result_df.head(TOP_N_PER_PATIENT)

		patient_selected = set()
		for rank, gene in enumerate(top_df["Gene"]):
			patient_selected.add(gene)
			global_gene_ranks.setdefault(gene, []).append(rank)

		for gene in patient_selected:
			global_gene_counter[gene] += 1
		print(f"Selected {len(patient_selected)} genes.")

	summary_data = []
	for gene, freq in global_gene_counter.items():
		avg_rank = float(np.mean(global_gene_ranks[gene]))
		summary_data.append(
			{"Gene": gene, "Frequency": freq, "Avg_Rank": avg_rank}
		)

	summary_df = pd.DataFrame(summary_data)
	if summary_df.empty:
		raise ValueError(f"No genes were selected for cluster column '{cluster_key}'.")

	summary_df = summary_df.sort_values(
		by=["Frequency", "Avg_Rank"], ascending=[False, True]
	).reset_index(drop=True)
	print("\n🏆 Top 20 Genes")
	print(summary_df.head(20).to_string(index=False))

	return summary_df["Gene"].head(250).tolist()


def process_files(
	input_dir: Path,
	output_dir: Path,
	selection_method: str = "deg",
	resolution_label: str | None = None,
) -> list[Path]:
	h5ad_files = collect_h5ad_files(input_dir)
	adata_list = load_adata_list(h5ad_files)
	common_genes = intersect_genes(adata_list)
	combined = combine_adata(adata_list, common_genes)
	patient_key = get_patient_column(combined)
	cluster_columns = get_cluster_columns(combined)
	if resolution_label is None:
		resolution_label = infer_resolution_label(input_dir)

	output_dir.mkdir(parents=True, exist_ok=True)
	output_paths: list[Path] = []

	for cluster_key in cluster_columns:
		feature_source, column_resolution_label = split_cluster_column(cluster_key)
		output_resolution_label = column_resolution_label or resolution_label
		label_parts = [feature_source]
		if output_resolution_label:
			label_parts.append(output_resolution_label)
		print(f"\n===== Running DEG for {' '.join(label_parts)} =====")
		top_genes = compute_deg_gene_list(combined, cluster_key, patient_key)
		output_path = output_dir / make_output_name(
			feature_source,
			selection_method,
			output_resolution_label,
		)
		with open(output_path, "w", encoding="utf-8") as handle:
			json.dump(top_genes, handle, ensure_ascii=False, indent=4)
		output_paths.append(output_path)
		print(f"Saved Top250 genes to {output_path}")

	return output_paths


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compute Wilcoxon DEG gene lists from clustered h5ad files."
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=DEFAULT_INPUT_DIR,
		help="Directory containing clustered h5ad files or a single h5ad file.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory where gene list JSON files will be written.",
	)
	parser.add_argument(
		"--selection-method",
		type=str,
		default="deg",
		help="Selection method label used in output filenames. Default: deg",
	)
	parser.add_argument(
		"--resolution-label",
		type=str,
		default=None,
		help=(
			"Fallback resolution label used when cluster columns do not include "
			"one, for example 0p5."
		),
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	process_files(
		args.input_dir,
		args.output_dir,
		args.selection_method,
		args.resolution_label,
	)


if __name__ == "__main__":
	main()


r"""
Linux example:
python ablation/DEG_selection.py \
	--input-dir datasets/CSCC/clustered/CSCC \
	--output-dir datasets/CSCC/rank_list

"""
