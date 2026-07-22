"""Attach BC1 pathologist labels to complete CONCH spot data in h5ad format.

The label files use grid coordinates (for example, x=9.02 and y=20.994),
whereas the feature files use zero-padded spot barcodes (``009x021``).  This
script converts the former to the latter.  The complete spot-level CONCH data
are retained and the pathologist class is added as ``obs["label"]``.

Sample selection is patient-aware.  The numeric suffix in ``A1``/``G2`` is
the slice rank within that patient when SPA identifiers are sorted in
descending numeric order.  Consequently A1 maps to SPA154 and G2 maps to
SPA123 for the current BC1 manifest.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import anndata as ad
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "BC1"
LABEL_FILE_PATTERN = re.compile(
	r"^(?P<patient>[A-Za-z]+)(?P<slice_rank>\d+)_labeled_coordinates\.tsv$"
)
SAMPLE_ID_PATTERN = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)$")


@dataclass(frozen=True)
class Annotation:
	barcode: str
	label: str


def decode_strings(values: Iterable[object]) -> list[str]:
	"""Decode an HDF5 string array without assuming a particular string dtype."""
	result = []
	for value in values:
		if isinstance(value, bytes):
			result.append(value.decode("utf-8"))
		else:
			result.append(str(value))
	return result


def decode_scalar(value: object) -> object:
	"""Convert HDF5/numpy scalar values to ordinary Python values."""
	if isinstance(value, bytes):
		return value.decode("utf-8")
	if isinstance(value, np.generic):
		return value.item()
	return value


def round_grid_coordinate(value: float) -> int:
	"""Round positive grid coordinates using conventional half-up rounding."""
	return math.floor(value + 0.5)


def coordinate_barcode(x: float, y: float) -> str:
	return f"{round_grid_coordinate(x):03d}x{round_grid_coordinate(y):03d}"


def parse_float(value: str | None) -> float:
	if value is None or not value.strip() or value.strip().upper() == "NA":
		return math.nan
	return float(value)


def read_annotations(path: Path) -> tuple[dict[str, Annotation], int]:
	"""Read annotations keyed by their canonical grid-coordinate barcode."""
	annotations: dict[str, Annotation] = {}
	missing_coordinates = 0
	with path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle, delimiter="\t")
		required = {"x", "y", "label"}
		missing_columns = required.difference(reader.fieldnames or [])
		if missing_columns:
			raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

		for row in reader:
			x = parse_float(row["x"])
			y = parse_float(row["y"])
			if not (math.isfinite(x) and math.isfinite(y)):
				missing_coordinates += 1
				continue
			barcode = coordinate_barcode(x, y)
			annotation = Annotation(
				barcode=barcode,
				label=row["label"].strip(),
			)
			previous = annotations.get(barcode)
			if previous is not None and previous.label != annotation.label:
				raise ValueError(
					f"{path} assigns conflicting labels to spot {barcode}: "
					f"{previous.label!r} and {annotation.label!r}"
				)
			annotations[barcode] = annotation
	return annotations, missing_coordinates


def sample_number(sample_id: str) -> int:
	match = SAMPLE_ID_PATTERN.match(sample_id)
	if not match:
		raise ValueError(f"Sample ID has no numeric suffix: {sample_id!r}")
	return int(match.group("number"))


def read_patient_samples(mapping_csv: Path) -> dict[str, list[str]]:
	"""Return sample IDs per patient letter, ordered by descending SPA number."""
	patient_samples: dict[str, list[str]] = {}
	with mapping_csv.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		if not {"sample_id", "patient"}.issubset(reader.fieldnames or []):
			raise ValueError(
				f"{mapping_csv} must contain the columns 'sample_id' and 'patient'"
			)
		for row in reader:
			patient = row["patient"].strip()
			# BC1 uses values such as "Patient A".  Keeping the last token also
			# permits a manifest that stores just "A".
			patient_key = patient.split()[-1].upper()
			patient_samples.setdefault(patient_key, []).append(row["sample_id"].strip())

	for samples in patient_samples.values():
		samples.sort(key=sample_number, reverse=True)
	return patient_samples


def resolve_sample_id(
	label_path: Path, patient_samples: dict[str, list[str]]
) -> tuple[str, str, int] | None:
	match = LABEL_FILE_PATTERN.match(label_path.name)
	if match is None:
		return None
	patient = match.group("patient").upper()
	slice_rank = int(match.group("slice_rank"))
	samples = patient_samples.get(patient)
	if not samples or slice_rank > len(samples):
		return None
	return samples[slice_rank - 1], patient, slice_rank


def find_feature_path(feature_dir: Path, sample_id: str) -> Path:
	for suffix in (".h5", ".h5ad"):
		candidate = feature_dir / f"{sample_id}{suffix}"
		if candidate.exists():
			return candidate
	raise FileNotFoundError(f"No feature file found for {sample_id} in {feature_dir}")


def h5_dataset(handle: h5py.File, candidates: Iterable[str]) -> h5py.Dataset | None:
	for name in candidates:
		if name in handle and isinstance(handle[name], h5py.Dataset):
			return handle[name]
	return None


def read_st_spots(st_path: Path) -> tuple[list[str], np.ndarray]:
	"""Read spot IDs and spatial coordinates from an h5ad using h5py only."""
	with h5py.File(st_path, "r") as handle:
		index = h5_dataset(handle, ("obs/_index", "obs_names", "barcode"))
		spatial = h5_dataset(handle, ("obsm/spatial", "spatial", "coords"))
		if index is None or spatial is None:
			raise KeyError(
				f"{st_path} must contain spot IDs and spatial coordinates "
				"(normally obs/_index and obsm/spatial)"
			)
		return decode_strings(index[:]), np.asarray(spatial[:])


def feature_barcodes(
	feature_handle: h5py.File,
	st_path: Path,
	spatial_tolerance: float,
) -> tuple[list[str], str]:
	"""Get one barcode per feature row, using ST spatial data as a fallback."""
	barcodes = h5_dataset(feature_handle, ("barcode", "barcodes", "obs/_index"))
	if barcodes is not None:
		return decode_strings(barcodes[:]), "feature barcode"

	feature_spatial = h5_dataset(
		feature_handle, ("spatial", "coords", "obsm/spatial")
	)
	if feature_spatial is None:
		raise KeyError(
			"Feature file has neither barcodes nor spatial coordinates; its rows "
			"cannot be aligned safely to individual ST spots"
		)

	st_barcodes, st_spatial = read_st_spots(st_path)
	feature_coords = np.asarray(feature_spatial[:], dtype=np.float64)
	st_coords = np.asarray(st_spatial, dtype=np.float64)
	if feature_coords.ndim != 2 or st_coords.ndim != 2:
		raise ValueError("Spatial coordinate arrays must be two-dimensional")

	# Avoid a scipy dependency: BC1 contains hundreds, rather than millions, of
	# spots, so a chunked squared-distance calculation is inexpensive.
	matched_barcodes: list[str] = []
	for start in range(0, len(feature_coords), 1024):
		chunk = feature_coords[start : start + 1024, :2]
		distances = np.sum((chunk[:, None, :] - st_coords[None, :, :2]) ** 2, axis=2)
		nearest = np.argmin(distances, axis=1)
		nearest_distance = np.sqrt(distances[np.arange(len(chunk)), nearest])
		if np.any(nearest_distance > spatial_tolerance):
			worst = float(nearest_distance.max())
			raise ValueError(
				f"Feature/ST spatial matching exceeded tolerance "
				f"{spatial_tolerance:g}; maximum nearest distance was {worst:g}"
			)
		matched_barcodes.extend(st_barcodes[index] for index in nearest)
	if len(set(matched_barcodes)) != len(matched_barcodes):
		raise ValueError("Feature-to-ST spatial matching was not one-to-one")
	return matched_barcodes, "ST spatial fallback"


def annotate_sample(
	feature_path: Path,
	st_path: Path,
	label_path: Path,
	output_path: Path,
	spatial_tolerance: float,
	force: bool,
) -> dict[str, object]:
	annotations, missing_coordinates = read_annotations(label_path)
	if output_path.exists() and not force:
		raise FileExistsError(f"Output already exists (use --force): {output_path}")
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with h5py.File(feature_path, "r") as source:
		barcodes, match_source = feature_barcodes(source, st_path, spatial_tolerance)
		if len(set(barcodes)) != len(barcodes):
			raise ValueError(f"{feature_path} contains duplicate spot barcodes")
		required_datasets = ("exp", "count", "features", "coords", "gene_list")
		missing_datasets = [name for name in required_datasets if name not in source]
		if missing_datasets:
			raise KeyError(
				f"{feature_path} is missing required datasets: {missing_datasets}"
			)
		for name in ("exp", "count", "features", "coords"):
			if source[name].shape[0] != len(barcodes):
				raise ValueError(
					f"{feature_path}:{name} has {source[name].shape[0]} rows but "
					f"there are {len(barcodes)} spot barcodes"
				)
		selected_indices = np.asarray(
			[index for index, barcode in enumerate(barcodes) if barcode in annotations],
			dtype=np.int64,
		)
		if len(selected_indices) == 0:
			raise ValueError(f"No labeled spots matched {feature_path}")
		selected = [annotations[barcodes[index]] for index in selected_indices]
		selected_barcodes = [barcodes[index] for index in selected_indices]
		gene_names = decode_strings(source["gene_list"][:])
		expression = np.asarray(source["exp"][selected_indices], dtype=np.float32)
		counts = np.asarray(source["count"][selected_indices], dtype=np.float32)
		features = np.asarray(source["features"][selected_indices], dtype=np.float32)
		spatial = np.asarray(source["coords"][selected_indices])
		if expression.shape != counts.shape:
			raise ValueError(
				f"{feature_path}: exp shape {expression.shape} does not match "
				f"count shape {counts.shape}"
			)
		if expression.shape[1] != len(gene_names):
			raise ValueError(
				f"{feature_path}: expression has {expression.shape[1]} genes but "
				f"gene_list contains {len(gene_names)} names"
			)
		patient = decode_scalar(source["patient"][()]) if "patient" in source else None
		sample = decode_scalar(source["sample"][()]) if "sample" in source else None
		feature_attrs = {key: decode_scalar(value) for key, value in source.attrs.items()}

	obs = pd.DataFrame(
		{"label": [item.label for item in selected]},
		index=pd.Index(selected_barcodes, name="barcode"),
	)
	var = pd.DataFrame(index=pd.Index(gene_names, name="gene"))
	result_adata = ad.AnnData(X=expression, obs=obs, var=var)
	# Keep both explicit fields from the source H5.  X is also exp by AnnData
	# convention, while layers['exp'] preserves the original field name.
	result_adata.layers["exp"] = expression.copy()
	result_adata.layers["count"] = counts
	result_adata.obsm["features"] = features
	result_adata.obsm["spatial"] = spatial
	if patient is not None:
		result_adata.uns["patient"] = patient
	if sample is not None:
		result_adata.uns["sample"] = sample
	if feature_attrs:
		result_adata.uns["feature_h5_attrs"] = feature_attrs
	result_adata.uns["feature_source"] = str(feature_path)
	result_adata.uns["label_source"] = str(label_path)
	result_adata.uns["spot_match_source"] = match_source
	result_adata.write_h5ad(output_path, compression="gzip")

	matched_barcodes = {barcodes[index] for index in selected_indices}
	return {
		"matched": len(selected_indices),
		"feature_spots": len(barcodes),
		"annotations": len(annotations),
		"unmatched_annotations": len(set(annotations).difference(matched_barcodes)),
		"missing_coordinates": missing_coordinates,
		"match_source": match_source,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
	parser.add_argument("--feature-name", default="conch")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Default: <dataset-dir>/spot_labels",
	)
	parser.add_argument(
		"--spatial-tolerance",
		type=float,
		default=2.0,
		help="Maximum pixel distance used only for the ST spatial fallback.",
	)
	parser.add_argument("--force", action="store_true", help="Overwrite output files.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	dataset_dir = args.dataset_dir.resolve()
	feature_dir = dataset_dir / "features" / args.feature_name
	label_dir = dataset_dir / "lbl"
	st_dir = dataset_dir / "st"
	mapping_csv = dataset_dir / f"{dataset_dir.name}.csv"
	output_dir = (
		args.output_dir.resolve()
		if args.output_dir is not None
		else dataset_dir / "spot_labels"
	)

	patient_samples = read_patient_samples(mapping_csv)
	processed = 0
	for label_path in sorted(label_dir.glob("*_labeled_coordinates.tsv")):
		resolved = resolve_sample_id(label_path, patient_samples)
		if resolved is None:
			print(f"SKIP {label_path.name}: no matching patient/slice in {mapping_csv.name}")
			continue
		sample_id, patient, slice_rank = resolved
		feature_path = find_feature_path(feature_dir, sample_id)
		st_path = st_dir / f"{sample_id}.h5ad"
		output_path = output_dir / f"{sample_id}.h5ad"
		result = annotate_sample(
			feature_path=feature_path,
			st_path=st_path,
			label_path=label_path,
			output_path=output_path,
			spatial_tolerance=args.spatial_tolerance,
			force=args.force,
		)
		processed += 1
		print(
			f"{patient}{slice_rank} -> {sample_id}: {result['matched']}/"
			f"{result['feature_spots']} feature spots labeled; "
			f"{result['unmatched_annotations']} annotations absent after feature QC; "
			f"{result['missing_coordinates']} annotations lacked x/y; "
			f"match={result['match_source']}; output={output_path}"
		)

	if processed == 0:
		raise RuntimeError(f"No label files were processed from {label_dir}")


if __name__ == "__main__":
	main()


'''
python annotation/paried_data.py --force

'''
