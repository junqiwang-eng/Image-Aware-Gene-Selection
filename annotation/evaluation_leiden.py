"""Evaluate sample-wise Leiden clusters against BC1 pathologist labels.

For every supplied gene-list JSON and every labeled sample, this script:

1. excludes spots whose ``obs['label']`` is ``undetermined``;
2. selects those genes from ``layers['exp']``;
3. optionally reduces the selected expression matrix with PCA (30 by default);
4. builds a nearest-neighbor graph on the reduced embedding;
5. runs Leiden clustering independently for that sample; and
6. computes ARI, AMI, and NMI on the remaining spots.

The output contains one row per gene list.  Metric values are the unweighted
mean of the sample-wise scores.  Existing rows are retained; a row with the
same gene-list name is updated rather than duplicated.  With ``--visualize``,
no CSV is written.  Instead, each sample is saved as a spatial PNG plot whose
spots are colored by Leiden cluster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import (
	adjusted_mutual_info_score,
	adjusted_rand_score,
	normalized_mutual_info_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "BC1" / "spot_labels"
DEFAULT_RANK_LIST_DIR = PROJECT_ROOT / "datasets" / "BC1" / "rank_list"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "annotation" / "evaluation_leiden.csv"
DEFAULT_ANNOTATED_DIR = PROJECT_ROOT / "datasets" / "BC1" / "annotated"
DEFAULT_THUMBNAIL_DIR = PROJECT_ROOT / "datasets" / "BC1" / "thumbnails"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "datasets" / "BC1" / "metadata"
METRIC_COLUMNS = ["ari", "ami", "nmi"]
UNDEFINED_LABEL = "undetermined"
DEFAULT_NEIGHBORS = 15
DEFAULT_RESOLUTION = 0.5
LABEL_COLORS = {
	"invasive cancer": "red",
	"adipose tissue": "cyan",
	"connective tissue": "blue",
	"breast glands": "green",
	"cancer in situ": "orange",
	"immune infiltrate": "yellow",
}
LABEL_DISPLAY_NAMES = {
	"invasive cancer": "Invasive Cancer",
	"adipose tissue": "Adipose tissue",
	"connective tissue": "Connective tissue",
	"breast glands": "Breast glands",
	"cancer in situ": "In situ cancer",
	"immune infiltrate": "Inflammatory cells",
}


def resolution_tag(resolution: float) -> str:
	"""Format a resolution for filenames, for example 0.5 -> 0p5."""
	text = f"{resolution:.12f}".rstrip("0")
	if text.endswith("."):
		text += "0"
	return text.replace("-", "m").replace(".", "p")


def path_with_resolution(path: Path, resolution: float) -> Path:
	"""Append the resolution tag to a path stem unless it is already present."""
	suffix = f"_{resolution_tag(resolution)}"
	if path.stem.endswith(suffix):
		return path
	return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def resolve_gene_list_path(value: str | Path, rank_list_dir: Path) -> Path:
	"""Resolve either a JSON path or a rank-list filename/stem."""
	value = Path(value)
	candidates = [value]
	if value.suffix.lower() != ".json":
		candidates.append(value.with_suffix(".json"))
	candidates.extend(rank_list_dir / candidate.name for candidate in list(candidates))
	for candidate in candidates:
		if candidate.is_file():
			return candidate.resolve()
	raise FileNotFoundError(
		f"Could not find gene-list JSON {value!s}; also searched {rank_list_dir}"
	)


def load_gene_list(path: Path) -> list[str]:
	with path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)
	if isinstance(payload, dict):
		for key in ("genes", "gene_list"):
			if key in payload:
				payload = payload[key]
				break
	if not isinstance(payload, list) or not all(isinstance(gene, str) for gene in payload):
		raise ValueError(
			f"{path} must contain a JSON string array or an object with 'genes'/'gene_list'"
		)
	genes = list(dict.fromkeys(gene.strip() for gene in payload if gene.strip()))
	if not genes:
		raise ValueError(f"{path} contains no gene names")
	return genes


def dense_float_array(values: object) -> np.ndarray:
	if sparse.issparse(values):
		values = values.toarray()
	result = np.asarray(values, dtype=np.float32)
	if result.ndim != 2:
		raise ValueError(f"Expression matrix must be 2-D, got shape {result.shape}")
	if not np.isfinite(result).all():
		raise ValueError("Expression matrix contains NaN or infinite values")
	return result


def expression_for_genes(
	adata: ad.AnnData, requested_genes: Iterable[str]
) -> tuple[np.ndarray, list[str]]:
	available = set(adata.var_names)
	selected_genes = [gene for gene in requested_genes if gene in available]
	if not selected_genes:
		raise ValueError("None of the requested genes are present in this sample")
	subset = adata[:, selected_genes]
	values = subset.layers["exp"] if "exp" in subset.layers else subset.X
	return dense_float_array(values), selected_genes


def defined_label_mask(labels: np.ndarray) -> np.ndarray:
	"""Return the spots that have a usable pathologist annotation."""
	return np.asarray(
		[label.strip().casefold() != UNDEFINED_LABEL for label in labels],
		dtype=bool,
	)


def match_clusters_to_labels(
	predicted: np.ndarray, labels: np.ndarray
) -> dict[str, str]:
	"""Match as many clusters as possible to labels by maximum spot agreement."""
	cluster_names = sorted(pd.unique(predicted), key=int)
	label_names = sorted(pd.unique(labels))
	overlap = np.zeros((len(cluster_names), len(label_names)), dtype=int)
	for cluster_index, cluster_name in enumerate(cluster_names):
		for label_index, label_name in enumerate(label_names):
			overlap[cluster_index, label_index] = np.count_nonzero(
				(predicted == cluster_name) & (labels == label_name)
			)
	cluster_indices, label_indices = linear_sum_assignment(-overlap)
	return {
		cluster_names[cluster_index]: label_names[label_index]
		for cluster_index, label_index in zip(cluster_indices, label_indices)
	}


def plot_spatial_clusters(
	adata: ad.AnnData,
	predicted: np.ndarray,
	labels: np.ndarray,
	output_path: Path,
	thumbnail_path: Path,
	metadata_path: Path,
	title: str,
	spot_size: float,
	dpi: int,
) -> None:
	"""Overlay discrete Leiden cluster colors on the tissue thumbnail."""
	import matplotlib

	matplotlib.use("Agg")
	from matplotlib import pyplot as plt

	if "spatial" not in adata.obsm:
		raise KeyError("AnnData is missing adata.obsm['spatial']")
	spatial = np.asarray(adata.obsm["spatial"], dtype=float)
	if spatial.ndim != 2 or spatial.shape != (adata.n_obs, 2):
		raise ValueError(
			"adata.obsm['spatial'] must have shape "
			f"({adata.n_obs}, 2), got {spatial.shape}"
		)
	if not thumbnail_path.is_file():
		raise FileNotFoundError(f"Missing tissue thumbnail: {thumbnail_path}")
	if not metadata_path.is_file():
		raise FileNotFoundError(f"Missing sample metadata: {metadata_path}")

	with metadata_path.open("r", encoding="utf-8") as handle:
		metadata = json.load(handle)
	full_width = float(metadata["fullres_width"])
	full_height = float(metadata["fullres_height"])
	if full_width <= 0 or full_height <= 0:
		raise ValueError(f"Invalid full-resolution dimensions in {metadata_path}")

	image = plt.imread(thumbnail_path)
	image_height, image_width = image.shape[:2]
	plot_coordinates = spatial.copy()
	plot_coordinates[:, 0] *= image_width / full_width
	plot_coordinates[:, 1] *= image_height / full_height

	cluster_names = sorted(pd.unique(predicted), key=int)
	cluster_to_label = match_clusters_to_labels(predicted, labels)
	fallback_colors = plt.get_cmap("tab20", len(cluster_names))
	figure, axis = plt.subplots(figsize=(9, 9))
	axis.imshow(image, origin="upper")
	for index, cluster_name in enumerate(cluster_names):
		mask = predicted == cluster_name
		matched_label = cluster_to_label.get(cluster_name)
		if matched_label is None:
			color = fallback_colors(index)
			legend_label = f"Cluster {cluster_name} (unmatched)"
		else:
			label_key = matched_label.strip().casefold()
			color = LABEL_COLORS.get(label_key, fallback_colors(index))
			legend_label = (
				f"Cluster {cluster_name}: "
				f"{LABEL_DISPLAY_NAMES.get(label_key, matched_label)}"
			)
		axis.scatter(
			plot_coordinates[mask, 0],
			plot_coordinates[mask, 1],
			s=spot_size,
			color=color,
			edgecolors="black",
			linewidths=0.35,
			label=legend_label,
		)

	x_min, y_min = plot_coordinates.min(axis=0)
	x_max, y_max = plot_coordinates.max(axis=0)
	x_margin = max((x_max - x_min) * 0.06, 10)
	y_margin = max((y_max - y_min) * 0.06, 10)
	axis.set_xlim(max(0, x_min - x_margin), min(image_width, x_max + x_margin))
	axis.set_ylim(min(image_height, y_max + y_margin), max(0, y_min - y_margin))
	axis.set_aspect("equal")
	axis.set_axis_off()
	axis.set_title(title)
	axis.legend(
		loc="center left",
		bbox_to_anchor=(1.01, 0.5),
		frameon=False,
		markerscale=1.2,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
	plt.close(figure)


def clustered_labels(
	clustering_input: np.ndarray,
	random_state: int,
	n_neighbors: int,
	resolution: float,
) -> np.ndarray:
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
	sc.tl.leiden(
		cluster_adata,
		neighbors_key="_neighbors",
		key_added="_cluster",
		resolution=resolution,
		random_state=random_state,
	)
	return cluster_adata.obs["_cluster"].astype(str).to_numpy()


def evaluate_sample(
	path: Path,
	genes: list[str],
	random_state: int,
	pca_components: int,
	n_neighbors: int,
	resolution: float,
	visualization_path: Path | None = None,
	thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR,
	metadata_dir: Path = DEFAULT_METADATA_DIR,
	gene_list_name: str = "gene list",
	spot_size: float = 300.0,
	plot_dpi: int = 200,
) -> dict[str, object]:
	adata = ad.read_h5ad(path)
	if "label" not in adata.obs:
		raise KeyError(f"{path} is missing adata.obs['label']")
	if adata.n_obs == 0:
		raise ValueError(f"{path} contains no spots")

	all_labels = adata.obs["label"].astype(str).to_numpy()
	keep_spots = defined_label_mask(all_labels)
	excluded_spots = int(np.count_nonzero(~keep_spots))
	adata = adata[keep_spots].copy()
	labels = all_labels[keep_spots]
	if adata.n_obs == 0:
		raise ValueError(f"{path} has no spots after excluding '{UNDEFINED_LABEL}'")
	n_clusters = int(pd.unique(labels).size)
	if n_clusters < 2:
		raise ValueError(
			f"{path} has only {n_clusters} ground-truth label class after excluding "
			f"'{UNDEFINED_LABEL}'"
		)
	if n_clusters > adata.n_obs:
		raise ValueError(
			f"{path} requests {n_clusters} clusters for only {adata.n_obs} spots"
		)

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
	ari = adjusted_rand_score(labels, predicted)
	ami = adjusted_mutual_info_score(labels, predicted)
	nmi = normalized_mutual_info_score(labels, predicted)
	if visualization_path is not None:
		plot_spatial_clusters(
			adata,
			predicted,
			labels,
			visualization_path,
			thumbnail_dir / f"{path.stem}_downscaled_fullres.jpeg",
			metadata_dir / f"{path.stem}.json",
			title=(
				f"{path.stem} - {gene_list_name} "
				f"(resolution={resolution:g}, ARI={ari:.4f})"
			),
			spot_size=spot_size,
			dpi=plot_dpi,
		)
	return {
		"sample": path.stem,
		"spots": adata.n_obs,
		"spots_total": adata.n_obs + excluded_spots,
		"spots_excluded": excluded_spots,
		"classes": n_clusters,
		"clusters": int(pd.unique(predicted).size),
		"genes_used": len(selected_genes),
		"genes_missing": len(genes) - len(selected_genes),
		"pca_components": pca_components,
		"ari": ari,
		"ami": ami,
		"nmi": nmi,
		"visualization_path": visualization_path,
	}


def evaluate_gene_list(
	data_paths: list[Path],
	genes: list[str],
	random_state: int,
	pca_components: int,
	n_neighbors: int,
	resolution: float,
	annotated_dir: Path | None = None,
	thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR,
	metadata_dir: Path = DEFAULT_METADATA_DIR,
	gene_list_name: str = "gene list",
	spot_size: float = 300.0,
	plot_dpi: int = 200,
) -> tuple[dict[str, float], list[dict[str, object]]]:
	sample_results = [
		evaluate_sample(
			path,
			genes,
			random_state=random_state,
			pca_components=pca_components,
			n_neighbors=n_neighbors,
			resolution=resolution,
			visualization_path=(
				annotated_dir
				/ (
					f"{path.stem}_{gene_list_name}_"
					f"{resolution_tag(resolution)}.png"
				)
				if annotated_dir
				else None
			),
			thumbnail_dir=thumbnail_dir,
			metadata_dir=metadata_dir,
			gene_list_name=gene_list_name,
			spot_size=spot_size,
			plot_dpi=plot_dpi,
		)
		for path in data_paths
	]
	mean_metrics = {
		metric: float(np.mean([float(result[metric]) for result in sample_results]))
		for metric in METRIC_COLUMNS
	}
	return mean_metrics, sample_results


def update_results_csv(
	path: Path,
	rows: list[dict[str, object]],
) -> None:
	"""Retain old gene lists and append/update the rows from this invocation."""
	new_frame = pd.DataFrame(rows).set_index("gene_list")[METRIC_COLUMNS]
	new_frame.index.name = "gene_list"
	if path.exists():
		existing = pd.read_csv(path, index_col="gene_list")
		missing_columns = set(METRIC_COLUMNS).difference(existing.columns)
		if missing_columns:
			raise ValueError(f"{path} is missing metric columns: {sorted(missing_columns)}")
		existing = existing[METRIC_COLUMNS]
		# Remove old versions of evaluated rows, then append their fresh values.
		existing = existing.loc[~existing.index.isin(new_frame.index)]
		result = pd.concat([existing, new_frame], axis=0)
	else:
		result = new_frame
	path.parent.mkdir(parents=True, exist_ok=True)
	result.to_csv(path, float_format="%.10f")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"gene_lists",
		nargs="+",
		help="One or more gene-list JSON paths, filenames, or stems.",
	)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--rank-list-dir", type=Path, default=DEFAULT_RANK_LIST_DIR)
	parser.add_argument(
		"--output-csv",
		type=Path,
		default=DEFAULT_OUTPUT_CSV,
		help=(
			"Base output CSV path; the resolution is appended to its filename "
			"(default: annotation/evaluation_leiden_<resolution>.csv)."
		),
	)
	parser.add_argument(
		"--visualize",
		"--Visualize",
		action="store_true",
		help=(
			"Do not write the evaluation CSV; save a spatial Leiden PNG for each "
			"sample instead."
		),
	)
	parser.add_argument(
		"--annotated-dir",
		type=Path,
		default=DEFAULT_ANNOTATED_DIR,
		help="Directory for spatial PNG plots produced by --visualize.",
	)
	parser.add_argument(
		"--thumbnail-dir",
		type=Path,
		default=DEFAULT_THUMBNAIL_DIR,
		help="Directory containing <sample>_downscaled_fullres.jpeg tissue images.",
	)
	parser.add_argument(
		"--metadata-dir",
		type=Path,
		default=DEFAULT_METADATA_DIR,
		help="Directory containing sample JSON files with full-resolution dimensions.",
	)
	parser.add_argument(
		"--spot-size",
		type=float,
		default=300.0,
		help="Marker area for spots in spatial plots (default: 300).",
	)
	parser.add_argument(
		"--plot-dpi",
		type=int,
		default=200,
		help="Resolution of spatial PNG plots (default: 200).",
	)
	parser.add_argument("--random-state", type=int, default=0)
	parser.add_argument("--pca-components", type=int, default=30)
	parser.add_argument("--n-neighbors", type=int, default=DEFAULT_NEIGHBORS)
	parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
	parser.add_argument(
		"--no-pca",
		action="store_true",
		help="Run Leiden directly on selected exp without PCA.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.pca_components < 1 and not args.no_pca:
		raise ValueError("--pca-components must be positive unless --no-pca is used")
	if args.n_neighbors < 2:
		raise ValueError("--n-neighbors must be at least 2")
	if args.resolution <= 0:
		raise ValueError("--resolution must be positive")
	if args.spot_size <= 0:
		raise ValueError("--spot-size must be positive")
	if args.plot_dpi <= 0:
		raise ValueError("--plot-dpi must be positive")
	pca_components = 0 if args.no_pca else args.pca_components
	output_csv = path_with_resolution(args.output_csv, args.resolution).resolve()
	data_paths = sorted(args.data_dir.resolve().glob("*.h5ad"))
	if not data_paths:
		raise FileNotFoundError(f"No .h5ad files found in {args.data_dir}")

	gene_list_paths = [
		resolve_gene_list_path(value, args.rank_list_dir.resolve())
		for value in args.gene_lists
	]
	rows = []
	for gene_list_path in gene_list_paths:
		genes = load_gene_list(gene_list_path)
		annotated_dir = None
		if args.visualize:
			annotated_dir = args.annotated_dir.resolve()
		metrics, sample_results = evaluate_gene_list(
			data_paths,
			genes,
			random_state=args.random_state,
			pca_components=pca_components,
			n_neighbors=args.n_neighbors,
			resolution=args.resolution,
			annotated_dir=annotated_dir,
			thumbnail_dir=args.thumbnail_dir.resolve(),
			metadata_dir=args.metadata_dir.resolve(),
			gene_list_name=gene_list_path.stem,
			spot_size=args.spot_size,
			plot_dpi=args.plot_dpi,
		)
		print(f"\n{gene_list_path.stem}: {len(genes)} requested genes")
		for result in sample_results:
			print(
				f"  {result['sample']}: spots={result['spots']}/{result['spots_total']} "
				f"(excluded={result['spots_excluded']}), "
				f"labels={result['classes']}, clusters={result['clusters']}, "
				f"PCA={result['pca_components'] if result['pca_components'] else 'off'}, "
				f"genes={result['genes_used']} "
				f"(missing={result['genes_missing']}), ARI={result['ari']:.6f}, "
				f"AMI={result['ami']:.6f}, NMI={result['nmi']:.6f}"
			)
			if result["visualization_path"] is not None:
				print(f"    saved plot: {result['visualization_path']}")
		print(
			f"  sample mean: ARI={metrics['ari']:.6f}, "
			f"AMI={metrics['ami']:.6f}, NMI={metrics['nmi']:.6f}"
		)
		rows.append({"gene_list": gene_list_path.stem, **metrics})

	if args.visualize:
		print(f"\nSaved spatial cluster plot(s) under {args.annotated_dir.resolve()}")
	else:
		update_results_csv(output_csv, rows)
		print(f"\nSaved {len(rows)} gene-list result(s) to {output_csv}")


if __name__ == "__main__":
	main()


'''
Run from the project root to evaluate every gene list in evaluation.csv:

python annotation/evaluation_leiden.py stem_hmhvg tri_hvg tri_hmg tri_hmhvg spatial_moran conch_cca_pca50 conch_mlp conch_fscore_0p5 her2st_DEG_296 her2st_DEG_200 conch_new_deg_0p5 conch_unrobust_fscore_0p5 --visualize
python annotation/evaluation_leiden.py her2st_DEG_296 her2st_DEG_200 --resolution 0.6
'''
