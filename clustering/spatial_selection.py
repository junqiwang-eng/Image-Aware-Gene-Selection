#!/usr/bin/env python3
"""Select spatially autocorrelated genes with sample-wise Moran's I.

The workflow is deliberately sample-aware:

1. In each sample, keep genes expressed in at least 10% of spots and with
   variance at least 1e-8.
2. Align every sample to the intersection of the genes that survived step 1.
3. Build a Squidpy spatial-neighbor graph and calculate Moran's I separately
   for every sample.
4. Average sample Moran's I values within each patient and rank the genes.
5. Merge patient top-k lists using the rule from ``selection.py``: selection
   frequency (descending), then mean within-patient rank (ascending).

Both extracted feature ``.h5`` files and AnnData ``.h5ad`` files are accepted.
For feature H5 input, full-precision spatial coordinates are loaded from the
matching ``<dataset>/st/<sample>.h5ad`` file. Rows are matched by barcode when
possible, with nearest-coordinate matching as a validated fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import issparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "BC1"
MIN_EXPRESSED_SPOT_FRACTION = 0.10
MIN_VARIANCE = 1e-8
TOP_K = 250


@dataclass
class Sample:
	name: str
	patient: str
	genes: list[str]
	expression: object
	spatial: np.ndarray


def decode_strings(values: np.ndarray) -> list[str]:
	values = np.asarray(values)
	if values.ndim > 1 and values.shape[1] == 1:
		values = values[:, 0]
	return [
		value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
		for value in values
	]


def scalar_string(value) -> str:
	if hasattr(value, "shape") and value.shape == ():
		value = value[()]
	if isinstance(value, (bytes, np.bytes_)):
		return value.decode("utf-8")
	return str(value)


def collect_input_files(input_dir: Path) -> list[Path]:
	if input_dir.is_file() and input_dir.suffix in {".h5", ".h5ad"}:
		return [input_dir]
	if not input_dir.is_dir():
		raise NotADirectoryError(input_dir)
	paths = sorted(
		path for path in input_dir.rglob("*") if path.suffix in {".h5", ".h5ad"}
	)
	if not paths:
		raise FileNotFoundError(f"No .h5 or .h5ad files found under {input_dir}")
	return paths


def load_patient_mapping(path: Path | None) -> dict[str, str]:
	if path is None:
		return {}
	if not path.exists():
		raise FileNotFoundError(path)
	with path.open(encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		if not {"sample_id", "patient"}.issubset(reader.fieldnames or []):
			raise ValueError(f"{path} must contain sample_id and patient columns")
		return {
			row["sample_id"].strip(): row["patient"].strip()
			for row in reader
			if row["sample_id"].strip() and row["patient"].strip()
		}


def validate_sample(sample: Sample, source: Path) -> None:
	if not sample.patient:
		raise ValueError(f"No patient label for sample {sample.name} in {source}")
	if len(sample.genes) != len(set(sample.genes)):
		raise ValueError(f"Duplicate gene names found in sample {sample.name}")
	if sample.expression.shape != (len(sample.spatial), len(sample.genes)):
		raise ValueError(
			f"Inconsistent shapes in {source}: expression={sample.expression.shape}, "
			f"spatial={sample.spatial.shape}, genes={len(sample.genes)}"
		)
	if sample.spatial.ndim != 2 or sample.spatial.shape[1] < 2:
		raise ValueError(f"Spatial coordinates in {source} must have shape (spots, >=2)")
	if len(sample.spatial) < 2:
		raise ValueError(f"Sample {sample.name} needs at least 2 spots")


def infer_st_dir(input_dir: Path) -> Path | None:
	"""Infer ``<dataset>/st`` from an input below ``<dataset>/features``."""
	start = input_dir if input_dir.is_dir() else input_dir.parent
	for parent in (start, *start.parents):
		if parent.name == "features":
			return parent.parent / "st"
	return None


def find_st_h5ad(st_dir: Path, sample_name: str) -> Path:
	direct = st_dir / f"{sample_name}.h5ad"
	if direct.is_file():
		return direct
	matches = sorted(st_dir.rglob(f"{sample_name}.h5ad")) if st_dir.is_dir() else []
	if not matches:
		raise FileNotFoundError(
			f"No ST AnnData file found for sample {sample_name!r} under {st_dir}"
		)
	if len(matches) > 1:
		raise ValueError(
			f"Multiple ST AnnData files found for sample {sample_name!r}: {matches}"
		)
	return matches[0]


def _barcode_values(adata) -> list[tuple[str, np.ndarray]]:
	values = [("obs_names", adata.obs_names.astype(str).to_numpy())]
	for column in ("barcode", "barcodes"):
		if column in adata.obs:
			values.append((f"obs[{column!r}]", adata.obs[column].astype(str).to_numpy()))
	return values


def match_spatial_from_h5ad(
	st_path: Path,
	barcodes: list[str] | None,
	coords: np.ndarray | None,
	coord_tolerance: float,
) -> tuple[np.ndarray, str]:
	"""Match feature-H5 rows to full-precision spatial coordinates in ST AnnData."""
	try:
		import anndata as ad
	except ImportError as exc:
		raise ImportError("Reading spatial coordinates from .h5ad requires anndata.") from exc

	adata = ad.read_h5ad(st_path)
	if "spatial" not in adata.obsm:
		raise KeyError(f"{st_path} is missing adata.obsm['spatial']")
	st_spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)
	if st_spatial.ndim != 2 or st_spatial.shape[1] < 2:
		raise ValueError(f"{st_path} obsm['spatial'] must have shape (spots, >=2)")
	st_spatial = st_spatial[:, :2]

	if barcodes is not None:
		for source, st_barcodes in _barcode_values(adata):
			if len(st_barcodes) != len(set(st_barcodes)):
				continue
			index = pd.Index(st_barcodes).get_indexer(barcodes)
			if np.all(index >= 0):
				return st_spatial[index].copy(), f"barcode via {source}"

	if coords is None:
		raise ValueError(
			f"Could not match all barcodes to {st_path}, and feature H5 has no coords fallback"
		)
	coords = np.asarray(coords, dtype=np.float64)
	if coords.ndim != 2 or coords.shape[1] < 2:
		raise ValueError("Feature H5 coords must have shape (spots, >=2)")
	distances, index = cKDTree(st_spatial).query(coords[:, :2], k=1)
	if np.any(distances > coord_tolerance):
		worst = float(np.max(distances))
		raise ValueError(
			f"Coordinate matching to {st_path} exceeded tolerance {coord_tolerance}; "
			f"maximum nearest distance is {worst:.4g}"
		)
	if len(index) != len(set(index.tolist())):
		raise ValueError(f"Coordinate matching to {st_path} is not one-to-one")
	return st_spatial[index].copy(), "nearest coordinate"


def load_h5_sample(
	path: Path,
	patient_by_sample: dict[str, str],
	expression_key: str,
	st_dir: Path,
	coord_tolerance: float,
) -> Sample:
	if expression_key == "X":
		raise ValueError("--expression-key X is only valid for .h5ad input")
	with h5py.File(path, "r") as handle:
		required = {"gene_list", expression_key}
		missing = required - set(handle.keys())
		if missing:
			raise KeyError(f"{path} is missing datasets: {sorted(missing)}")
		if "barcode" not in handle and "coords" not in handle:
			raise KeyError(f"{path} needs barcode or coords to match its ST AnnData file")
		genes = decode_strings(handle["gene_list"][:])
		expression = np.asarray(handle[expression_key][:], dtype=np.float32)
		barcodes = decode_strings(handle["barcode"][:]) if "barcode" in handle else None
		coords = np.asarray(handle["coords"][:, :2]) if "coords" in handle else None
		sample_name = scalar_string(handle["sample"]) if "sample" in handle else path.stem
		patient = scalar_string(handle["patient"]).strip() if "patient" in handle else ""
		if not patient:
			patient = patient_by_sample.get(
				sample_name,
				patient_by_sample.get(path.stem, ""),
			)
	st_path = find_st_h5ad(st_dir, sample_name)
	spatial, _ = match_spatial_from_h5ad(
		st_path,
		barcodes,
		coords,
		coord_tolerance,
	)
	sample = Sample(sample_name, patient, genes, expression, spatial)
	validate_sample(sample, path)
	return sample


def _obs_value(adata, column: str, indices: np.ndarray) -> str:
	values = pd.unique(adata.obs.iloc[indices][column].dropna().astype(str))
	if len(values) != 1:
		raise ValueError(
			f"Expected one {column!r} value per sample, found {values.tolist()}"
		)
	return str(values[0])


def load_h5ad_samples(
	path: Path,
	patient_by_sample: dict[str, str],
	expression_key: str,
) -> list[Sample]:
	try:
		import anndata as ad
	except ImportError as exc:
		raise ImportError("Reading .h5ad input requires anndata.") from exc

	adata = ad.read_h5ad(path)
	if "spatial" not in adata.obsm:
		raise KeyError(f"{path} is missing adata.obsm['spatial']")
	if expression_key == "X":
		matrix = adata.X
	elif expression_key in adata.layers:
		matrix = adata.layers[expression_key]
	else:
		raise KeyError(f"{path} is missing adata.layers[{expression_key!r}]")

	sample_column = next(
		(column for column in ("sample_id", "sample") if column in adata.obs),
		None,
	)
	if sample_column is None:
		sample_groups = [(path.stem, np.arange(adata.n_obs))]
	else:
		sample_groups = [
			(
				str(name),
				np.flatnonzero(
					adata.obs[sample_column].astype(str).to_numpy() == str(name)
				),
			)
			for name in pd.unique(adata.obs[sample_column].astype(str))
		]

	patient_column = next(
		(column for column in ("patient_id", "patient") if column in adata.obs),
		None,
	)
	samples = []
	for sample_name, indices in sample_groups:
		patient = (
			_obs_value(adata, patient_column, indices)
			if patient_column is not None
			else patient_by_sample.get(sample_name, "")
		)
		sample = Sample(
			sample_name,
			patient,
			adata.var_names.astype(str).tolist(),
			matrix[indices].copy(),
			np.asarray(adata.obsm["spatial"])[indices, :2].copy(),
		)
		validate_sample(sample, path)
		samples.append(sample)
	return samples


def load_samples(
	input_dir: Path,
	mapping_csv: Path | None,
	expression_key: str,
	st_dir: Path | None,
	coord_tolerance: float,
) -> list[Sample]:
	patient_by_sample = load_patient_mapping(mapping_csv)
	resolved_st_dir = st_dir or infer_st_dir(input_dir)
	samples = []
	for path in collect_input_files(input_dir):
		if path.suffix == ".h5":
			if resolved_st_dir is None:
				raise ValueError(
					"Cannot infer the ST directory from --input-dir; provide --st-dir"
				)
			samples.append(
				load_h5_sample(
					path,
					patient_by_sample,
					expression_key,
					resolved_st_dir,
					coord_tolerance,
				)
			)
		else:
			samples.extend(load_h5ad_samples(path, patient_by_sample, expression_key))
	names = [sample.name for sample in samples]
	if len(names) != len(set(names)):
		raise ValueError("Sample names must be unique across all input files")
	return samples


def gene_filter_mask(
	expression,
	min_expressed_fraction: float,
	min_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return keep mask, expressed fractions and population variances."""
	if not 0 <= min_expressed_fraction <= 1:
		raise ValueError("min_expressed_fraction must be between 0 and 1")
	if min_variance < 0:
		raise ValueError("min_variance must be non-negative")
	if issparse(expression):
		x = expression.astype(np.float64)
		expressed_fraction = np.asarray((x > 0).mean(axis=0)).ravel()
		means = np.asarray(x.mean(axis=0)).ravel()
		second_moments = np.asarray(x.multiply(x).mean(axis=0)).ravel()
		variances = np.maximum(second_moments - means * means, 0.0)
	else:
		x = np.asarray(expression, dtype=np.float64)
		expressed_fraction = np.mean(x > 0, axis=0)
		variances = np.var(x, axis=0)
	keep = (
		(expressed_fraction >= min_expressed_fraction)
		& (variances >= min_variance)
		& np.isfinite(variances)
	)
	return keep, expressed_fraction, variances


def filter_and_intersect_genes(
	samples: list[Sample],
	min_expressed_fraction: float,
	min_variance: float,
) -> list[str]:
	if not samples:
		raise ValueError("At least one sample is required")
	common_set: set[str] | None = None
	for sample in samples:
		keep, _, _ = gene_filter_mask(
			sample.expression,
			min_expressed_fraction,
			min_variance,
		)
		retained = {gene for gene, keep_gene in zip(sample.genes, keep) if keep_gene}
		common_set = retained if common_set is None else common_set & retained

	common = [gene for gene in samples[0].genes if gene in (common_set or set())]
	if not common:
		raise ValueError("No genes remain in the intersection after sample-wise filtering")
	return common


def align_expression(sample: Sample, genes: list[str]):
	index = {gene: gene_i for gene_i, gene in enumerate(sample.genes)}
	columns = [index[gene] for gene in genes]
	return sample.expression[:, columns]


def moran_scores(
	sample: Sample,
	genes: list[str],
	coord_type: str,
	n_neighs: int,
	delaunay: bool,
	n_perms: int | None,
	n_jobs: int,
	seed: int,
) -> pd.Series:
	try:
		import anndata as ad
		import squidpy as sq
	except ImportError as exc:
		raise ImportError(
			"Moran's I selection requires squidpy and anndata. "
			"Install them in the environment used to run this script."
		) from exc

	adata = ad.AnnData(
		X=align_expression(sample, genes),
		var=pd.DataFrame(index=pd.Index(genes, name="gene")),
	)
	adata.obsm["spatial"] = sample.spatial
	neighbor_kwargs = {
		"spatial_key": "spatial",
		"coord_type": coord_type,
		"delaunay": delaunay,
	}
	if not delaunay:
		neighbor_kwargs["n_neighs"] = min(n_neighs, adata.n_obs - 1)
	sq.gr.spatial_neighbors(adata, **neighbor_kwargs)
	autocorr_kwargs = {
		"mode": "moran",
		"genes": genes,
		"n_jobs": n_jobs,
		"seed": seed,
		"copy": True,
	}
	if n_perms is not None:
		autocorr_kwargs["n_perms"] = n_perms
	result = sq.gr.spatial_autocorr(adata, **autocorr_kwargs)
	if result is None:
		result = adata.uns.get("moranI")
	if result is None or "I" not in result:
		raise RuntimeError("Squidpy did not return a Moran's I result table")
	return result["I"].reindex(genes).astype(float)


def compute_sample_tables(samples: list[Sample], genes: list[str], args) -> list[pd.DataFrame]:
	tables = []
	for sample in samples:
		print(f"[moran] {sample.name} ({sample.patient}): {len(sample.spatial)} spots")
		scores = moran_scores(
			sample,
			genes,
			args.coord_type,
			args.n_neighs,
			args.delaunay,
			args.n_perms,
			args.n_jobs,
			args.seed,
		)
		tables.append(
			pd.DataFrame(
				{
					"sample": sample.name,
					"patient": sample.patient,
					"gene": genes,
					"score": scores.reindex(genes).to_numpy(),
				}
			)
		)
	return tables


def average_samples_by_patient(sample_tables: list[pd.DataFrame]) -> list[pd.DataFrame]:
	all_scores = pd.concat(sample_tables, ignore_index=True)
	patient_tables = []
	for patient, table in all_scores.groupby("patient", sort=True):
		patient_table = (
			table.groupby("gene", as_index=False, sort=False)
			.agg(score=("score", "mean"), samples=("sample", "nunique"))
			.sort_values(["score", "gene"], ascending=[False, True], na_position="last")
			.reset_index(drop=True)
		)
		patient_table.insert(0, "patient", patient)
		patient_table["rank"] = np.arange(1, len(patient_table) + 1)
		print(f"[patient] {patient}: averaged {int(patient_table['samples'].max())} sample(s)")
		patient_tables.append(patient_table)
	return patient_tables


def aggregate_patient_ranks(patient_tables: list[pd.DataFrame], top_k: int) -> pd.DataFrame:
	"""Apply selection.py's frequency then average-rank cross-patient merge."""
	selected_tables = [
		table.dropna(subset=["score"]).head(top_k)[["gene", "rank"]]
		for table in patient_tables
	]
	selected_tables = [table for table in selected_tables if not table.empty]
	if not selected_tables:
		raise ValueError("No finite patient-level Moran's I scores were produced")
	selected = pd.concat(selected_tables, ignore_index=True)
	return (
		selected.groupby("gene", as_index=False)
		.agg(frequency=("rank", "size"), avg_rank=("rank", "mean"))
		.sort_values(
			["frequency", "avg_rank", "gene"],
			ascending=[False, True, True],
		)
		.reset_index(drop=True)
	)


def resolve_dataset_paths(args: argparse.Namespace) -> None:
	"""Fill standard input/output paths from a single dataset name."""
	dataset_dir = PROJECT_ROOT / "datasets" / args.dataset
	if args.input_dir is None:
		args.input_dir = dataset_dir / "features" / "conch"
	if args.st_dir is None:
		args.st_dir = dataset_dir / "st"
	if args.mapping_csv is None:
		args.mapping_csv = dataset_dir / f"{args.dataset}.csv"
	if args.output_dir is None:
		args.output_dir = dataset_dir / "rank_list"


def run(args: argparse.Namespace) -> Path:
	resolve_dataset_paths(args)
	print(
		f"Dataset {args.dataset}: input={args.input_dir}, st={args.st_dir}, "
		f"mapping={args.mapping_csv}, output={args.output_dir}"
	)
	samples = load_samples(
		args.input_dir,
		args.mapping_csv,
		args.expression_key,
		args.st_dir,
		args.coord_match_tolerance,
	)
	patients = {sample.patient for sample in samples}
	print(f"Loaded {len(samples)} samples from {len(patients)} patients")
	genes = filter_and_intersect_genes(
		samples,
		args.min_expressed_fraction,
		args.min_variance,
	)
	sample_tables = compute_sample_tables(samples, genes, args)
	patient_tables = average_samples_by_patient(sample_tables)
	aggregate = aggregate_patient_ranks(patient_tables, args.top_k)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	output_path = args.output_dir / args.output_name
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(
			aggregate["gene"].head(args.top_k).tolist(),
			handle,
			ensure_ascii=False,
			indent=4,
		)
	print(f"Saved {min(args.top_k, len(aggregate))} genes to {output_path}")
	return output_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--dataset",
		default=DEFAULT_DATASET,
		help="Dataset name used to fill all standard paths, for example BC1, BC2, or CSCC.",
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=None,
		help="Optional override. Default: datasets/<dataset>/features/conch.",
	)
	parser.add_argument(
		"--st-dir",
		type=Path,
		default=None,
		help="Optional override. Default: datasets/<dataset>/st.",
	)
	parser.add_argument(
		"--mapping-csv",
		type=Path,
		default=None,
		help="Optional override. Default: datasets/<dataset>/<dataset>.csv.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Optional override. Default: datasets/<dataset>/rank_list.",
	)
	parser.add_argument("--output-name", default="spatial_moran.json")
	parser.add_argument(
		"--expression-key",
		choices=("exp", "count", "counts", "X"),
		default="exp",
		help="HDF5 dataset or AnnData layer used for filtering and Moran's I.",
	)
	parser.add_argument(
		"--min-expressed-fraction",
		type=float,
		default=MIN_EXPRESSED_SPOT_FRACTION,
	)
	parser.add_argument("--min-variance", type=float, default=MIN_VARIANCE)
	parser.add_argument(
		"--coord-match-tolerance",
		type=float,
		default=2.0,
		help="Maximum pixel distance for coords fallback when barcode matching fails.",
	)
	parser.add_argument("--top-k", type=int, default=TOP_K)
	parser.add_argument("--coord-type", choices=("generic", "grid"), default="generic")
	parser.add_argument("--n-neighs", type=int, default=6)
	parser.add_argument(
		"--delaunay",
		action="store_true",
		help="Use Delaunay triangulation instead of a k-nearest-neighbor graph.",
	)
	parser.add_argument(
		"--n-perms",
		type=int,
		default=None,
		help="Optional permutations for Squidpy significance estimates.",
	)
	parser.add_argument("--n-jobs", type=int, default=-1)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	if not 0 <= args.min_expressed_fraction <= 1:
		parser.error("--min-expressed-fraction must be between 0 and 1")
	if args.min_variance < 0:
		parser.error("--min-variance must be non-negative")
	if args.coord_match_tolerance < 0:
		parser.error("--coord-match-tolerance must be non-negative")
	if args.n_neighs < 1:
		parser.error("--n-neighs must be at least 1")
	if args.top_k < 1:
		parser.error("--top-k must be at least 1")
	if args.n_perms is not None and args.n_perms < 1:
		parser.error("--n-perms must be at least 1 when provided")
	return args


if __name__ == "__main__":
	run(parse_args())


r'''
Default BC1 run:
python clustering/spatial_selection.py

Run another dataset; all four paths are filled automatically:
python clustering/spatial_selection.py \
	--dataset BC2 \
	--expression-key exp \
	--top-k 250 \
	--n-jobs -1
'''
