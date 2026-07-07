import argparse
import csv
from pathlib import Path


DEFAULT_FILES = ["fscore_pcc.csv", "fscore_scc.csv"]
MODEL_ORDER = ["resnet50", "conch", "uni", "midnight"]
MODEL_ALIASES = {
	"resnet": "resnet50",
	"resnet50": "resnet50",
	"resnet50trunc": "resnet50",
	"conch": "conch",
	"conchv1": "conch",
	"uni": "uni",
	"univ1": "uni",
	"midnight": "midnight",
}


def normalize_name(name):
	return (
		str(name)
		.lower()
		.replace("_", "")
		.replace("-", "")
		.replace(" ", "")
	)


def model_sort_key(name):
	model_name = MODEL_ALIASES.get(normalize_name(name), normalize_name(name))
	if model_name in MODEL_ORDER:
		return (0, MODEL_ORDER.index(model_name))
	return (1, model_name)


def experiment_column_sort_key(column):
	parts = str(column).rsplit("_", 1)
	if len(parts) != 2:
		return ((1, str(column)), (1, ""))

	feature_name, json_name = parts
	return (model_sort_key(feature_name), model_sort_key(json_name))


def sorted_columns(columns):
	id_columns = [column for column in ["Patient", "Top_K"] if column in columns]
	experiment_columns = [column for column in columns if column not in id_columns]
	return id_columns + sorted(experiment_columns, key=experiment_column_sort_key)


def sort_csv_columns(csv_path):
	csv_path = Path(csv_path)
	if not csv_path.exists():
		print(f"Skip missing file: {csv_path}")
		return

	with csv_path.open("r", newline="", encoding="utf-8") as file:
		reader = csv.DictReader(file)
		rows = list(reader)
		if not reader.fieldnames:
			print(f"Skip empty file: {csv_path}")
			return

	fieldnames = sorted_columns(reader.fieldnames)

	with csv_path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	print(f"Sorted columns: {csv_path}")


def main():
	parser = argparse.ArgumentParser(
		description="Sort fscore_pcc/fscore_scc columns by feature and json model order."
	)
	parser.add_argument(
		"files",
		nargs="*",
		default=DEFAULT_FILES,
		help="CSV files to sort. Default: fscore_pcc.csv fscore_scc.csv",
	)
	args = parser.parse_args()

	for csv_file in args.files:
		sort_csv_columns(csv_file)


if __name__ == "__main__":
	main()
