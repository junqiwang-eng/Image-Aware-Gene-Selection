"""Compare Leiden cluster counts across methods and resolution values.

The detailed output contains one row per resolution and gene-list method.  Its
sample columns list the Leiden cluster count for each patient.  The summary
computes, separately for every patient and resolution, the mean, population
standard deviation across gene-list methods.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA

from evaluation_leiden import (
	DEFAULT_DATA_DIR,
	DEFAULT_NEIGHBORS,
	DEFAULT_RANK_LIST_DIR,
	UNDEFINED_LABEL,
	defined_label_mask,
	expression_for_genes,
	load_gene_list,
	resolve_gene_list_path,
	resolution_tag,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS_CSV = PROJECT_ROOT / "annotation" / "evaluation.csv"
DEFAULT_OUTPUT_CSV = (
	PROJECT_ROOT / "annotation" / "leiden_resolution_cluster_counts.csv"
)
DEFAULT_RESOLUTIONS = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9]


def load_default_methods(path: Path) -> list[str]:
	if not path.is_file():
		raise FileNotFoundError(
			f"Default method list not found: {path}; provide methods explicitly"
		)
	frame = pd.read_csv(path)
	if "gene_list" not in frame.columns:
		raise ValueError(f"{path} is missing the 'gene_list' column")
	methods = list(dict.fromkeys(frame["gene_list"].dropna().astype(str)))
	if not methods:
		raise ValueError(f"{path} contains no gene-list methods")
	return methods


def prepare_clustering_input(
	path: Path,
	genes: list[str],
	pca_components: int,
	random_state: int,
) -> tuple[np.ndarray, int, int]:
	"""Prepare one filtered sample once for reuse across all resolutions."""
	adata = ad.read_h5ad(path)
	if "label" not in adata.obs:
		raise KeyError(f"{path} is missing adata.obs['label']")
	if adata.n_obs == 0:
		raise ValueError(f"{path} contains no spots")

	labels = adata.obs["label"].astype(str).to_numpy()
	keep_spots = defined_label_mask(labels)
	excluded_spots = int(np.count_nonzero(~keep_spots))
	adata = adata[keep_spots].copy()
	if adata.n_obs == 0:
		raise ValueError(f"{path} has no spots after excluding '{UNDEFINED_LABEL}'")

	expression, selected_genes = expression_for_genes(adata, genes)
	if pca_components > 0:
		max_components = min(expression.shape)
		if pca_components > max_components:
			raise ValueError(
				f"{path} cannot run PCA with {pca_components} components: selected "
				f"expression has shape {expression.shape} (maximum is {max_components})"
			)
		clustering_input = PCA(
			n_components=pca_components,
			random_state=random_state,
		).fit_transform(expression)
	else:
		clustering_input = expression
	return clustering_input, len(selected_genes), excluded_spots


def build_neighbor_graph(
	clustering_input: np.ndarray,
	n_neighbors: int,
) -> ad.AnnData:
	"""Build the resolution-independent neighbor graph once per sample/method."""
	cluster_adata = ad.AnnData(X=clustering_input)
	cluster_adata.obsm["_clustering_input"] = clustering_input
	neighbors = min(n_neighbors, cluster_adata.n_obs - 1)
	if neighbors < 2:
		raise ValueError("Sample has too few spots for neighbors-based clustering")
	sc.pp.neighbors(
		cluster_adata,
		n_neighbors=neighbors,
		use_rep="_clustering_input",
		key_added="_neighbors",
	)
	return cluster_adata


def count_leiden_clusters(
	cluster_adata: ad.AnnData,
	resolution: float,
	random_state: int,
) -> int:
	"""Run Leiden on a prepared graph and return its number of communities."""
	sc.tl.leiden(
		cluster_adata,
		neighbors_key="_neighbors",
		key_added="_cluster",
		resolution=resolution,
		random_state=random_state,
	)
	return int(cluster_adata.obs["_cluster"].astype(str).nunique())


def evaluate_cluster_counts(
	data_paths: list[Path],
	methods: list[tuple[str, list[str]]],
	resolutions: list[float],
	pca_components: int,
	n_neighbors: int,
	random_state: int,
) -> pd.DataFrame:
	sample_names = [path.stem for path in data_paths]
	rows: list[dict[str, object]] = []

	for method_name, genes in methods:
		prepared_samples: dict[str, ad.AnnData] = {}
		for path in data_paths:
			clustering_input, genes_used, excluded_spots = prepare_clustering_input(
				path,
				genes,
				pca_components=pca_components,
				random_state=random_state,
			)
			prepared_samples[path.stem] = build_neighbor_graph(
				clustering_input,
				n_neighbors=n_neighbors,
			)
			print(
				f"Prepared {method_name}/{path.stem}: spots={len(clustering_input)}, "
				f"excluded={excluded_spots}, genes={genes_used}"
			)

		for resolution in resolutions:
			cluster_counts: dict[str, int] = {}
			for sample_name in sample_names:
				cluster_counts[sample_name] = count_leiden_clusters(
					prepared_samples[sample_name],
					random_state=random_state,
					resolution=resolution,
				)
			rows.append(
				{
					"resolution": resolution_tag(resolution),
					"gene_list": method_name,
					**cluster_counts,
				}
			)
			counts_text = ", ".join(
				f"{sample}={count}" for sample, count in cluster_counts.items()
			)
			print(f"  resolution={resolution:g}: {counts_text}")

	columns = [
		"resolution",
		"gene_list",
		*sample_names,
	]
	return pd.DataFrame(rows, columns=columns)


def resolution_summary(
	detail: pd.DataFrame,
	sample_names: list[str],
	ground_truth: dict[str, object],
) -> pd.DataFrame:
	"""Compute sample-wise statistics across methods at each resolution."""
	rows: list[dict[str, object]] = [
		{
			"resolution": "ground_truth",
			"statistic": "cluster_count",
			**{sample: ground_truth[sample] for sample in sample_names},
		}
	]
	for resolution, group in detail.groupby("resolution", sort=False):
		values = group[sample_names].to_numpy(dtype=float)
		statistics = {
			"mean": values.mean(axis=0),
			"std": values.std(axis=0, ddof=0),
		}
		for statistic, result in statistics.items():
			rows.append(
				{
					"resolution": resolution,
					"statistic": statistic,
					**dict(zip(sample_names, result)),
				}
			)
	return pd.DataFrame(rows, columns=["resolution", "statistic", *sample_names])


def ground_truth_row(data_paths: list[Path]) -> dict[str, object]:
	"""Build the first output row from defined pathologist label counts."""
	cluster_counts: dict[str, int] = {}
	for path in data_paths:
		adata = ad.read_h5ad(path)
		if "label" not in adata.obs:
			raise KeyError(f"{path} is missing adata.obs['label']")
		labels = adata.obs["label"].astype(str).to_numpy()
		defined_labels = labels[defined_label_mask(labels)]
		if defined_labels.size == 0:
			raise ValueError(f"{path} has no labels after excluding '{UNDEFINED_LABEL}'")
		cluster_counts[path.stem] = int(pd.unique(defined_labels).size)
	return {
		"resolution": "ground_truth",
		"gene_list": "ground_truth",
		**cluster_counts,
	}


def suffixed_path(path: Path, suffix: str) -> Path:
	return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"methods",
		nargs="*",
		help=(
			"Gene-list paths, filenames, or stems. By default, methods are read "
			"from annotation/evaluation.csv."
		),
	)
	parser.add_argument(
		"--resolutions",
		type=float,
		nargs="+",
		default=DEFAULT_RESOLUTIONS,
		help="Resolution values to compare (default: 0.1 0.3 0.5 0.6 0.7 0.9).",
	)
	parser.add_argument("--methods-csv", type=Path, default=DEFAULT_METHODS_CSV)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--rank-list-dir", type=Path, default=DEFAULT_RANK_LIST_DIR)
	parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
	parser.add_argument("--random-state", type=int, default=0)
	parser.add_argument("--pca-components", type=int, default=30)
	parser.add_argument("--n-neighbors", type=int, default=DEFAULT_NEIGHBORS)
	parser.add_argument(
		"--no-pca",
		action="store_true",
		help="Run Leiden directly on selected exp without PCA.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not args.resolutions or any(value <= 0 for value in args.resolutions):
		raise ValueError("--resolutions must contain only positive values")
	if args.pca_components < 1 and not args.no_pca:
		raise ValueError("--pca-components must be positive unless --no-pca is used")
	if args.n_neighbors < 2:
		raise ValueError("--n-neighbors must be at least 2")

	method_names = (
		list(dict.fromkeys(args.methods))
		if args.methods
		else load_default_methods(args.methods_csv.resolve())
	)
	methods = []
	for method_name in method_names:
		path = resolve_gene_list_path(method_name, args.rank_list_dir.resolve())
		methods.append((path.stem, load_gene_list(path)))

	data_paths = sorted(args.data_dir.resolve().glob("*.h5ad"))
	if not data_paths:
		raise FileNotFoundError(f"No .h5ad files found in {args.data_dir}")
	pca_components = 0 if args.no_pca else args.pca_components
	detail = evaluate_cluster_counts(
		data_paths,
		methods,
		list(dict.fromkeys(args.resolutions)),
		pca_components=pca_components,
		n_neighbors=args.n_neighbors,
		random_state=args.random_state,
	)

	output_path = args.output_csv.resolve()
	summary_path = suffixed_path(output_path, "summary")
	fscore_path = suffixed_path(output_path, "fscore")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	sample_names = [path.stem for path in data_paths]
	ground_truth = ground_truth_row(data_paths)
	output_detail = pd.concat(
		[
			pd.DataFrame([ground_truth], columns=detail.columns),
			detail,
		],
		ignore_index=True,
	)
	output_detail.to_csv(output_path, index=False, float_format="%.6f")
	resolution_summary(detail, sample_names, ground_truth).to_csv(
		summary_path,
		index=False,
		float_format="%.6f",
	)
	fscore = detail[
		detail["gene_list"].str.contains("fscore", case=False, na=False)
	]
	fscore.to_csv(fscore_path, index=False, float_format="%.6f")

	print(f"\nSaved detailed cluster counts to {output_path}")
	print(f"Saved per-resolution summary to {summary_path}")
	print(f"Saved {len(fscore)} F-score row(s) to {fscore_path}")
	if not fscore.empty:
		print("\nF-score cluster counts:")
		print(fscore.to_string(index=False))


if __name__ == "__main__":
	main()


# Run from the project root with methods from annotation/evaluation.csv:
# python annotation/evaluation_leiden_resolutions.py
# Run selected methods or custom resolutions:
# python annotation/evaluation_leiden_resolutions.py conch_fscore_0p5 spatial_moran
# python annotation/evaluation_leiden_resolutions.py --resolutions 0.2 0.4 0.6 0.8 1.0
