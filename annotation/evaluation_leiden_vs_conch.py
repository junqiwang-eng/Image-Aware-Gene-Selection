"""Evaluate gene-based Leiden clusters against saved CONCH cluster labels.

Unlike ``evaluation_leiden.py``, this program does not use pathologist labels
and does not filter ``undetermined`` spots.  It reads the recursively stored
clustered BC1 samples, runs Leiden on the selected genes for each sample, and
computes ARI, AMI, and NMI against a CONCH cluster column in ``adata.obs``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
	adjusted_mutual_info_score,
	adjusted_rand_score,
	normalized_mutual_info_score,
)

from evaluation_leiden import (
	DEFAULT_NEIGHBORS,
	DEFAULT_RANK_LIST_DIR,
	METRIC_COLUMNS,
	clustered_labels,
	expression_for_genes,
	load_gene_list,
	resolve_gene_list_path,
	resolution_tag,
	update_results_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "BC1" / "clustered" / "BC1"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "annotation" / "evaluation_leiden_vs_conch.csv"
DEFAULT_GENE_RESOLUTION = 0.6
DEFAULT_CONCH_RESOLUTION = 0.5


def conch_cluster_key(resolution: float) -> str:
	return f"conch_{resolution_tag(resolution)}_cluster"


def output_path_with_resolutions(
	path: Path,
	gene_resolution: float,
	conch_resolution: float,
) -> Path:
	suffix = (
		f"_gene_{resolution_tag(gene_resolution)}"
		f"_conch_{resolution_tag(conch_resolution)}"
	)
	if path.stem.endswith(suffix):
		return path
	return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def evaluate_sample(
	path: Path,
	genes: list[str],
	cluster_key: str,
	random_state: int,
	pca_components: int,
	n_neighbors: int,
	resolution: float,
) -> dict[str, object]:
	adata = ad.read_h5ad(path)
	if cluster_key not in adata.obs:
		available = sorted(
			column
			for column in adata.obs.columns
			if column.startswith("conch_") and column.endswith("_cluster")
		)
		raise KeyError(
			f"{path} is missing adata.obs[{cluster_key!r}]; "
			f"available CONCH columns: {available}"
		)
	if adata.n_obs == 0:
		raise ValueError(f"{path} contains no spots")

	conch_labels = adata.obs[cluster_key].astype(str).to_numpy()
	conch_clusters = int(pd.unique(conch_labels).size)
	if conch_clusters < 2:
		raise ValueError(f"{path} has only {conch_clusters} CONCH cluster")

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

	predicted = clustered_labels(
		clustering_input,
		random_state=random_state,
		n_neighbors=n_neighbors,
		resolution=resolution,
	)
	patient = (
		str(adata.obs["patient_id"].iloc[0])
		if "patient_id" in adata.obs and adata.n_obs
		else "unknown"
	)
	return {
		"sample": path.stem,
		"patient": patient,
		"spots": adata.n_obs,
		"conch_clusters": conch_clusters,
		"gene_clusters": int(pd.unique(predicted).size),
		"genes_used": len(selected_genes),
		"genes_missing": len(genes) - len(selected_genes),
		"pca_components": pca_components,
		"ari": adjusted_rand_score(conch_labels, predicted),
		"ami": adjusted_mutual_info_score(conch_labels, predicted),
		"nmi": normalized_mutual_info_score(conch_labels, predicted),
	}


def evaluate_gene_list(
	data_paths: list[Path],
	genes: list[str],
	cluster_key: str,
	random_state: int,
	pca_components: int,
	n_neighbors: int,
	resolution: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
	sample_results = [
		evaluate_sample(
			path,
			genes,
			cluster_key=cluster_key,
			random_state=random_state,
			pca_components=pca_components,
			n_neighbors=n_neighbors,
			resolution=resolution,
		)
		for path in data_paths
	]
	mean_metrics = {
		metric: float(np.mean([float(result[metric]) for result in sample_results]))
		for metric in METRIC_COLUMNS
	}
	return mean_metrics, sample_results


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"gene_lists",
		nargs="+",
		help="One or more gene-list JSON paths, filenames, or stems.",
	)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--rank-list-dir", type=Path, default=DEFAULT_RANK_LIST_DIR)
	parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
	parser.add_argument(
		"--samples",
		nargs="+",
		default=None,
		help="Only evaluate specified sample names, for example SPA121 SPA123.",
	)
	parser.add_argument(
		"--resolution",
		type=float,
		default=DEFAULT_GENE_RESOLUTION,
		help="Resolution for gene-based Leiden clustering (default: 0.6).",
	)
	parser.add_argument(
		"--conch-resolution",
		type=float,
		default=DEFAULT_CONCH_RESOLUTION,
		help="Resolution of the saved CONCH reference clusters (default: 0.5).",
	)
	parser.add_argument(
		"--conch-cluster-key",
		default=None,
		help=(
			"Explicit CONCH obs column. By default this is derived from "
			"--conch-resolution, for example conch_0p5_cluster."
		),
	)
	parser.add_argument("--random-state", type=int, default=0)
	parser.add_argument("--pca-components", type=int, default=30)
	parser.add_argument("--n-neighbors", type=int, default=DEFAULT_NEIGHBORS)
	parser.add_argument(
		"--no-pca",
		action="store_true",
		help="Run gene-based Leiden directly on selected exp without PCA.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.resolution <= 0:
		raise ValueError("--resolution must be positive")
	if args.conch_resolution <= 0:
		raise ValueError("--conch-resolution must be positive")
	if args.pca_components < 1 and not args.no_pca:
		raise ValueError("--pca-components must be positive unless --no-pca is used")
	if args.n_neighbors < 2:
		raise ValueError("--n-neighbors must be at least 2")

	data_paths = sorted(args.data_dir.resolve().rglob("*.h5ad"))
	if not data_paths:
		raise FileNotFoundError(f"No .h5ad files found under {args.data_dir}")
	if args.samples:
		paths_by_sample = {path.stem: path for path in data_paths}
		requested_samples = list(
			dict.fromkeys(Path(sample).stem for sample in args.samples)
		)
		missing = [sample for sample in requested_samples if sample not in paths_by_sample]
		if missing:
			raise FileNotFoundError(
				f"Sample(s) not found under {args.data_dir}: {', '.join(missing)}"
			)
		data_paths = [paths_by_sample[sample] for sample in requested_samples]

	cluster_key = args.conch_cluster_key or conch_cluster_key(args.conch_resolution)
	pca_components = 0 if args.no_pca else args.pca_components
	gene_list_paths = [
		resolve_gene_list_path(value, args.rank_list_dir.resolve())
		for value in args.gene_lists
	]
	rows = []
	for gene_list_path in gene_list_paths:
		genes = load_gene_list(gene_list_path)
		metrics, sample_results = evaluate_gene_list(
			data_paths,
			genes,
			cluster_key=cluster_key,
			random_state=args.random_state,
			pca_components=pca_components,
			n_neighbors=args.n_neighbors,
			resolution=args.resolution,
		)
		print(f"\n{gene_list_path.stem}: {len(genes)} requested genes")
		for result in sample_results:
			print(
				f"  {result['sample']} ({result['patient']}): spots={result['spots']}, "
				f"CONCH clusters={result['conch_clusters']}, "
				f"gene clusters={result['gene_clusters']}, "
				f"genes={result['genes_used']} (missing={result['genes_missing']}), "
				f"ARI={result['ari']:.6f}, AMI={result['ami']:.6f}, "
				f"NMI={result['nmi']:.6f}"
			)
		print(
			f"  sample mean vs {cluster_key}: ARI={metrics['ari']:.6f}, "
			f"AMI={metrics['ami']:.6f}, NMI={metrics['nmi']:.6f}"
		)
		rows.append({"gene_list": gene_list_path.stem, **metrics})

	output_path = output_path_with_resolutions(
		args.output_csv,
		gene_resolution=args.resolution,
		conch_resolution=args.conch_resolution,
	).resolve()
	update_results_csv(output_path, rows)
	print(f"\nSaved {len(rows)} gene-list result(s) to {output_path}")


if __name__ == "__main__":
	main()


# Run from the project root:
# python annotation/evaluation_leiden_vs_conch.py tri_hmg tri_hvg tri_hmhvg stem_hmhvg her2st_DEG_296 conch_cca_pca50 conch_mlp conch_fscore_0p5 spatial_moran --resolution 0.5
# python annotation/evaluation_leiden_vs_conch.py her2st_DEG_296 --resolution 0.6
# python annotation/evaluation_leiden_vs_conch.py tri_hmg --samples SPA121 SPA123
