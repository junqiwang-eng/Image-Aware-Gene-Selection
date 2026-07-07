import argparse
import csv
from pathlib import Path

import h5py


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "datasets" / "BC1" / "features"
DEFAULT_MAPPING_CSV = PROJECT_ROOT / "datasets" / "BC1.csv"


def load_patient_mapping(csv_path):
	patient_by_sample = {}
	with open(csv_path, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		required_columns = {"sample_id", "patient"}
		missing_columns = required_columns - set(reader.fieldnames or [])
		if missing_columns:
			raise ValueError(f"Missing columns in {csv_path}: {sorted(missing_columns)}")

		for row in reader:
			sample_id = row["sample_id"].strip()
			patient = row["patient"].strip()
			if not sample_id or not patient:
				continue
			patient_by_sample[sample_id] = patient

	return patient_by_sample


def write_string_dataset(h5_file, key, value):
	if key in h5_file:
		del h5_file[key]
	h5_file.create_dataset(key, data=value, dtype=h5py.string_dtype(encoding="utf-8"))


def add_labels_to_h5(h5_path, patient_by_sample):
	sample_id = h5_path.stem
	if sample_id not in patient_by_sample:
		return False, f"missing patient mapping for {sample_id}"

	with h5py.File(h5_path, "r+") as f:
		write_string_dataset(f, "sample", sample_id)
		write_string_dataset(f, "patient", patient_by_sample[sample_id])

	return True, patient_by_sample[sample_id]


def iter_feature_h5_files(features_dir):
	for feature_dir in sorted(path for path in features_dir.iterdir() if path.is_dir()):
		for h5_path in sorted(feature_dir.glob("*.h5")):
			yield feature_dir, h5_path


def add_patient_labels(features_dir, mapping_csv):
	patient_by_sample = load_patient_mapping(mapping_csv)
	updated = 0
	skipped = 0
	feature_counts = {}

	for feature_dir, h5_path in iter_feature_h5_files(features_dir):
		ok, message = add_labels_to_h5(h5_path, patient_by_sample)
		if ok:
			updated += 1
			feature_counts[feature_dir.name] = feature_counts.get(feature_dir.name, 0) + 1
		else:
			skipped += 1
			print(f"Skip {h5_path}: {message}")

	print(f"Updated {updated} files; skipped {skipped} files.")
	for feature_name, count in sorted(feature_counts.items()):
		print(f"  {feature_name}: {count}")


def parse_args():
	parser = argparse.ArgumentParser(
		description="Add sample and patient string datasets to BC1 feature .h5 files."
	)
	parser.add_argument(
		"--features-dir",
		default=DEFAULT_FEATURES_DIR,
		type=Path,
		help="Directory containing feature subfolders.",
	)
	parser.add_argument(
		"--mapping-csv",
		default=DEFAULT_MAPPING_CSV,
		type=Path,
		help="CSV with sample_id and patient columns.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	add_patient_labels(args.features_dir, args.mapping_csv)

'''
	python clustering/add_patient.py
'''
