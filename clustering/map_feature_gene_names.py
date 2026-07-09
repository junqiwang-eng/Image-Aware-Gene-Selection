#!/usr/bin/env python3
"""Map raw ST h5ad var_names from Ensembl IDs to gene symbols.

This script is intentionally limited to the raw ``st/*.h5ad`` stage. It updates
``adata.var_names`` before preprocessing/make_paired.py runs, so make_paired.py
can compute spot-level mitochondrial QC and remove mt/ribo/hb genes using
symbol-style gene names.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def collect_h5ad_files(input_dir: Path) -> list[Path]:
	if input_dir.is_file() and input_dir.suffix == ".h5ad":
		return [input_dir]
	if not input_dir.is_dir():
		raise NotADirectoryError(input_dir)

	h5ad_files = sorted(input_dir.rglob("*.h5ad"))
	if not h5ad_files:
		raise FileNotFoundError(f"No .h5ad files found under {input_dir}")
	return h5ad_files


def read_h5ad_genes(path: Path) -> list[str]:
	import anndata as ad

	adata = ad.read_h5ad(path, backed="r")
	try:
		return [str(gene) for gene in adata.var_names]
	finally:
		adata.file.close()


def load_gene_lists(h5ad_files: list[Path]) -> dict[Path, list[str]]:
	return {path: read_h5ad_genes(path) for path in h5ad_files}


def load_mapping_csv(
	path: Path,
	gene_id_column: str,
	gene_name_column: str,
) -> dict[str, str]:
	mapping_df = pd.read_csv(path)
	missing = {gene_id_column, gene_name_column} - set(mapping_df.columns)
	if missing:
		raise ValueError(f"{path} is missing columns: {sorted(missing)}")

	return (
		mapping_df[[gene_id_column, gene_name_column]]
		.dropna()
		.drop_duplicates(subset=[gene_id_column])
		.assign(**{gene_id_column: lambda df: df[gene_id_column].astype(str)})
		.assign(**{gene_name_column: lambda df: df[gene_name_column].astype(str)})
		.set_index(gene_id_column)[gene_name_column]
		.to_dict()
	)


def load_mapping_gprofiler(
	genes: list[str],
	organism: str,
	target_namespace: str,
	chunk_size: int,
) -> dict[str, str]:
	try:
		from gprofiler import GProfiler
	except ImportError as exc:
		raise ImportError(
			"g:Profiler mapping requires gprofiler-official. "
			"Install it with: pip install gprofiler-official"
		) from exc

	gp = GProfiler(return_dataframe=True)
	mapping: dict[str, str] = {}
	unique_genes = sorted(set(genes))
	for start in range(0, len(unique_genes), chunk_size):
		chunk = unique_genes[start : start + chunk_size]
		result = gp.convert(
			organism=organism,
			query=chunk,
			target_namespace=target_namespace,
		)
		if result.empty:
			continue

		converted_column = "converted" if "converted" in result.columns else "name"
		if converted_column not in result.columns:
			raise ValueError(
				f"Cannot find converted gene column in g:Profiler result: {list(result.columns)}"
			)

		for _, row in result.dropna(subset=["incoming", converted_column]).iterrows():
			incoming = str(row["incoming"])
			converted = str(row[converted_column])
			if incoming and converted and incoming not in mapping:
				mapping[incoming] = converted
		print(f"Mapped {min(start + chunk_size, len(unique_genes))}/{len(unique_genes)} genes.")
	return mapping


def map_h5ad_file(input_path: Path, output_path: Path, mapping: dict[str, str]) -> tuple[int, int, int]:
	import anndata as ad

	adata = ad.read_h5ad(input_path)
	original_genes = [str(gene) for gene in adata.var_names]
	mapped_genes = [mapping.get(gene, gene) for gene in original_genes]
	mapped_count = sum(original != mapped for original, mapped in zip(original_genes, mapped_genes))

	adata.var["gene_id_original"] = original_genes
	adata.var_names = mapped_genes
	duplicate_count = int(adata.var_names.duplicated().sum())
	if duplicate_count:
		print(f"{input_path}: {duplicate_count} duplicate gene symbols; applying var_names_make_unique().")
		adata.var_names_make_unique()

	adata.uns["gene_name_mapping_source"] = "map_feature_gene_names.py"
	adata.uns["gene_name_mapping_applied"] = True

	output_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
	adata.write_h5ad(tmp_path)
	tmp_path.replace(output_path)
	return len(original_genes), mapped_count, duplicate_count


def process_files(
	input_dir: Path,
	output_dir: Path | None,
	mapping: dict[str, str],
	in_place: bool,
) -> list[Path]:
	h5ad_files = collect_h5ad_files(input_dir)
	root_dir = input_dir if input_dir.is_dir() else input_dir.parent
	output_paths = []

	for input_path in h5ad_files:
		if in_place:
			output_path = input_path
			backup_path = input_path.with_suffix(input_path.suffix + ".bak")
			if not backup_path.exists():
				shutil.copy2(input_path, backup_path)
		else:
			if output_dir is None:
				raise ValueError("--output-dir is required unless --in-place is used.")
			output_path = output_dir / input_path.relative_to(root_dir)

		n_genes, mapped_count, duplicate_count = map_h5ad_file(input_path, output_path, mapping)
		output_paths.append(output_path)
		print(
			f"{input_path} -> {output_path}: "
			f"{mapped_count}/{n_genes} mapped, "
			f"{duplicate_count} duplicate mapped names"
		)

	return output_paths


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Map raw ST h5ad var_names to gene symbols.")
	parser.add_argument(
		"--input-dir",
		type=Path,
		required=True,
		help="Directory containing raw st .h5ad files, or a single .h5ad file.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Output directory. Required unless --in-place is used.",
	)
	parser.add_argument(
		"--in-place",
		action="store_true",
		help="Overwrite input .h5ad files after creating .bak backups.",
	)
	parser.add_argument(
		"--map-csv",
		type=Path,
		default=None,
		help="Optional local mapping CSV. If omitted, g:Profiler is used.",
	)
	parser.add_argument("--gene-id-column", default="gene_id")
	parser.add_argument("--gene-name-column", default="gene_name")
	parser.add_argument("--organism", default="hsapiens")
	parser.add_argument("--target-namespace", default="HGNC")
	parser.add_argument("--chunk-size", type=int, default=1000)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	h5ad_files = collect_h5ad_files(args.input_dir)
	gene_lists = load_gene_lists(h5ad_files)
	all_genes = [gene for genes in gene_lists.values() for gene in genes]

	if args.map_csv is not None:
		mapping = load_mapping_csv(args.map_csv, args.gene_id_column, args.gene_name_column)
	else:
		mapping = load_mapping_gprofiler(
			all_genes,
			organism=args.organism,
			target_namespace=args.target_namespace,
			chunk_size=args.chunk_size,
		)
	print(f"Loaded mapping for {len(mapping)} input gene IDs.")

	process_files(
		args.input_dir,
		args.output_dir,
		mapping,
		in_place=args.in_place,
	)


if __name__ == "__main__":
	main()


'''
Create mapped copies for raw ST h5ad files, then run make_paired.py on that output dataset:
python clustering/map_feature_gene_names.py \
	--input-dir datasets/BC2/st \
	--output-dir datasets/BC2_symbol/st

Use a local mapping CSV instead of g:Profiler:
python clustering/map_feature_gene_names.py \
	--input-dir datasets/BC2/st \
	--output-dir datasets/BC2_symbol/st \
	--map-csv datasets/BC2/ensembl_to_symbol.csv \
	--gene-id-column gene_id \
	--gene-name-column gene_name

Overwrite raw ST h5ad files after creating .bak backups:
python clustering/map_feature_gene_names.py \
	--input-dir datasets/BC2/st \
	--in-place
'''
