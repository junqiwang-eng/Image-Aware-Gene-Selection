"""Visualize ground-truth BC1 annotations on the spatial tissue images.

Each ``.h5ad`` file in ``datasets/BC1/spot_labels`` is rendered using its
``obs['label']`` values.  By default, PNG files are written alongside the
K-means visualizations in ``datasets/BC1/annotated``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "BC1" / "spot_labels"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "BC1" / "annotated"
DEFAULT_THUMBNAIL_DIR = PROJECT_ROOT / "datasets" / "BC1" / "thumbnails"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "datasets" / "BC1" / "metadata"
LABEL_COLORS = {
	"invasive cancer": "red",
	"adipose tissue": "cyan",
	"connective tissue": "blue",
	"breast glands": "green",
	"cancer in situ": "orange",
	"immune infiltrate": "yellow",
	"undetermined": "gray",
}
LABEL_DISPLAY_NAMES = {
	"invasive cancer": "Invasive Cancer",
	"adipose tissue": "Adipose tissue",
	"connective tissue": "Connective tissue",
	"breast glands": "Breast glands",
	"cancer in situ": "In situ cancer",
	"immune infiltrate": "Inflammatory cells",
	"undetermined": "Undetermined",
}


def plot_annotation(
	data_path: Path,
	output_path: Path,
	thumbnail_path: Path,
	metadata_path: Path,
	spot_size: float,
	dpi: int,
) -> tuple[int, int]:
	"""Render one sample and return its spot and label counts."""
	import matplotlib

	matplotlib.use("Agg")
	from matplotlib import pyplot as plt

	adata = ad.read_h5ad(data_path)
	if "label" not in adata.obs:
		raise KeyError(f"{data_path} is missing adata.obs['label']")
	if "spatial" not in adata.obsm:
		raise KeyError(f"{data_path} is missing adata.obsm['spatial']")
	if adata.n_obs == 0:
		raise ValueError(f"{data_path} contains no spots")
	if not thumbnail_path.is_file():
		raise FileNotFoundError(f"Missing tissue thumbnail: {thumbnail_path}")
	if not metadata_path.is_file():
		raise FileNotFoundError(f"Missing sample metadata: {metadata_path}")

	labels = adata.obs["label"].astype(str).to_numpy()
	spatial = np.asarray(adata.obsm["spatial"], dtype=float)
	if spatial.shape != (adata.n_obs, 2):
		raise ValueError(
			f"{data_path}: adata.obsm['spatial'] must have shape "
			f"({adata.n_obs}, 2), got {spatial.shape}"
		)
	if not np.isfinite(spatial).all():
		raise ValueError(f"{data_path}: spatial coordinates contain NaN or infinity")

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

	label_names = sorted(np.unique(labels), key=lambda value: value.casefold())
	fallback_colors = plt.get_cmap("tab20", len(label_names))
	figure, axis = plt.subplots(figsize=(9, 9))
	axis.imshow(image, origin="upper")
	for index, label_name in enumerate(label_names):
		mask = labels == label_name
		label_key = label_name.strip().casefold()
		axis.scatter(
			plot_coordinates[mask, 0],
			plot_coordinates[mask, 1],
			s=spot_size,
			color=LABEL_COLORS.get(label_key, fallback_colors(index)),
			edgecolors="black",
			linewidths=0.35,
			label=LABEL_DISPLAY_NAMES.get(label_key, label_name),
		)

	x_min, y_min = plot_coordinates.min(axis=0)
	x_max, y_max = plot_coordinates.max(axis=0)
	x_margin = max((x_max - x_min) * 0.06, 10)
	y_margin = max((y_max - y_min) * 0.06, 10)
	axis.set_xlim(max(0, x_min - x_margin), min(image_width, x_max + x_margin))
	axis.set_ylim(min(image_height, y_max + y_margin), max(0, y_min - y_margin))
	axis.set_aspect("equal")
	axis.set_axis_off()
	axis.set_title(f"{data_path.stem} - Ground-truth annotation")
	axis.legend(
		loc="center left",
		bbox_to_anchor=(1.01, 0.5),
		frameon=False,
		markerscale=1.2,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
	plt.close(figure)
	return adata.n_obs, len(label_names)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"samples",
		nargs="*",
		help="Optional sample names (for example SPA121); defaults to every sample.",
	)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--thumbnail-dir", type=Path, default=DEFAULT_THUMBNAIL_DIR)
	parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
	parser.add_argument("--spot-size", type=float, default=300.0)
	parser.add_argument("--plot-dpi", type=int, default=200)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.spot_size <= 0:
		raise ValueError("--spot-size must be positive")
	if args.plot_dpi <= 0:
		raise ValueError("--plot-dpi must be positive")

	data_dir = args.data_dir.resolve()
	if args.samples:
		data_paths = [data_dir / f"{Path(sample).stem}.h5ad" for sample in args.samples]
		missing = [path for path in data_paths if not path.is_file()]
		if missing:
			raise FileNotFoundError(f"Sample file not found: {missing[0]}")
	else:
		data_paths = sorted(data_dir.glob("*.h5ad"))
		if not data_paths:
			raise FileNotFoundError(f"No .h5ad files found in {data_dir}")

	output_dir = args.output_dir.resolve()
	thumbnail_dir = args.thumbnail_dir.resolve()
	metadata_dir = args.metadata_dir.resolve()
	for data_path in data_paths:
		output_path = output_dir / f"{data_path.stem}_annotation.png"
		spots, classes = plot_annotation(
			data_path,
			output_path,
			thumbnail_dir / f"{data_path.stem}_downscaled_fullres.jpeg",
			metadata_dir / f"{data_path.stem}.json",
			spot_size=args.spot_size,
			dpi=args.plot_dpi,
		)
		print(
			f"{data_path.stem}: spots={spots}, labels={classes}, "
			f"saved plot: {output_path}"
		)


if __name__ == "__main__":
	main()


# Run from the project root:
# python annotation/visualize_annotation.py
# python3 annotation/visualize_annotation.py SPA121
