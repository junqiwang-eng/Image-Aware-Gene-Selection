"""Specialized 50-gene evaluation with a smaller MLP and compact CSV output."""

import argparse
from pathlib import Path

import train


GENE_NUM = 50
HIDDEN_DIM = 256
TOP_KS = (10, 50)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run 50-gene leave-one-patient-out ST prediction evaluation."
	)
	parser.add_argument(
		"--data-dir",
		default=train.PROJECT_ROOT / "datasets" / "BC1" / "features",
		help="Directory containing feature .h5 files, or feature subfolders.",
	)
	parser.add_argument("--feature-name", default="conch")
	parser.add_argument("--json-paths", nargs="+", default=["resnet50"])
	parser.add_argument("--rank-list-dir", default=None, type=Path)
	parser.add_argument("--mapping-csv", default=None, type=Path)
	parser.add_argument(
		"--model-key",
		default="ff",
		choices=["ff", "linear"],
	)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--gene-num",
		type=int,
		default=GENE_NUM,
		help="Must be 50 for this specialized entry point.",
	)
	parser.add_argument("--test-as-val", action="store_true")
	args = parser.parse_args()
	if args.gene_num != GENE_NUM:
		parser.error(f"train_50.py requires --gene-num {GENE_NUM}")
	return args


def main():
	args = parse_args()
	train.configure_torch_performance()
	rank_list_dir = args.rank_list_dir
	if rank_list_dir is None and Path(args.data_dir).name == "features":
		rank_list_dir = Path(args.data_dir).parent / "rank_list"
	mapping_csv = args.mapping_csv or train.infer_mapping_csv(args.data_dir)
	train.run_fold_evaluation(
		data_dir=args.data_dir,
		feature_name=args.feature_name,
		json_paths=args.json_paths,
		rank_list_dir=rank_list_dir,
		model_key=args.model_key,
		seed=args.seed,
		mapping_csv=mapping_csv,
		test_as_val=args.test_as_val,
		gene_num=GENE_NUM,
		hidden_dim=HIDDEN_DIM,
		top_ks=TOP_KS,
		average_only=True,
	)


if __name__ == "__main__":
	main()
