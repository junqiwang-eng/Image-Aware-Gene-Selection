"""Predict pathologist annotations from selected-gene expression.

Only ``.h5ad`` samples in ``spot_labels`` are used. Spots with an undefined
annotation are removed, and evaluation uses leave-one-sample-out folds.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.metrics import (
	accuracy_score,
	f1_score,
	recall_score,
)
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "BC1" / "spot_labels"
DEFAULT_RANK_LIST_DIR = PROJECT_ROOT / "datasets" / "BC1" / "rank_list"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "annotation_classification"
DEFAULT_GENE_NUM = 200
DEFAULT_HIDDEN_DIM = 32
DEFAULT_LEARNING_RATE = 1e-4
TEST_EPOCH_SWEEP = (50, 75, 100, 125, 150, 175, 200, 250, 300)
DEFAULT_UNDEFINED_LABELS = (
	"",
	"na",
	"n/a",
	"nan",
	"none",
	"null",
	"undefined",
	"undetermined",
	"unknown",
)


@dataclass
class SampleData:
	sample_id: str
	spot_ids: np.ndarray
	expression: np.ndarray
	labels: np.ndarray


def set_global_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def resolve_gene_list_path(path: str | Path) -> Path:
	path = Path(path)
	candidates = [path]
	if not path.suffix:
		candidates.extend(
			[
				DEFAULT_RANK_LIST_DIR / f"{path.name}.json",
				DEFAULT_RANK_LIST_DIR / f"{path.name}_fscore.json",
			]
		)
	elif not path.is_absolute():
		candidates.append(DEFAULT_RANK_LIST_DIR / path.name)

	for candidate in candidates:
		if candidate.is_file():
			return candidate.resolve()
	raise FileNotFoundError(
		f"Cannot find gene list {path!s}. Tried: {[str(item) for item in candidates]}"
	)


def read_gene_list(path: str | Path, gene_num: int | None = None) -> tuple[Path, list[str]]:
	path = resolve_gene_list_path(path)
	with path.open("r", encoding="utf-8") as handle:
		genes = json.load(handle)
	if not isinstance(genes, list) or not genes:
		raise ValueError(f"{path} must contain a non-empty JSON list of gene names")

	# Preserve ranking order while preventing duplicate feature columns.
	unique_genes = list(dict.fromkeys(str(gene).strip() for gene in genes if str(gene).strip()))
	if gene_num is not None:
		if gene_num < 1:
			raise ValueError("gene_num must be at least 1")
		unique_genes = unique_genes[:gene_num]
	return path, unique_genes


def find_annotated_paths(data_dir: str | Path, sample_ids: list[str] | None) -> list[Path]:
	data_dir = Path(data_dir)
	if sample_ids:
		paths = [data_dir / f"{Path(sample_id).stem}.h5ad" for sample_id in sample_ids]
		missing = [path for path in paths if not path.is_file()]
		if missing:
			raise FileNotFoundError(f"Annotated sample not found: {missing[0]}")
	else:
		paths = sorted(data_dir.glob("*.h5ad"))
	if len(paths) < 2:
		raise ValueError(f"Need at least two annotated samples; found {len(paths)}")
	return paths


def common_selected_genes(paths: list[Path], requested_genes: list[str]) -> list[str]:
	available_per_sample = []
	for path in paths:
		adata = ad.read_h5ad(path, backed="r")
		try:
			available_per_sample.append(set(map(str, adata.var_names)))
		finally:
			adata.file.close()

	genes = [gene for gene in requested_genes if all(gene in names for names in available_per_sample)]
	missing = [gene for gene in requested_genes if gene not in genes]
	if missing:
		print(
			f"Warning: skipped {len(missing)} genes absent from at least one sample "
			f"(examples: {missing[:5]})"
		)
	if not genes:
		raise ValueError("None of the requested genes is shared by every annotated sample")
	return genes


def _dense_float32(matrix) -> np.ndarray:
	if sparse.issparse(matrix):
		matrix = matrix.toarray()
	return np.asarray(matrix, dtype=np.float32)


def load_sample(
	path: Path,
	genes: list[str],
	label_key: str,
	undefined_labels: set[str],
) -> SampleData:
	adata = ad.read_h5ad(path)
	if label_key not in adata.obs:
		raise KeyError(f"{path} does not contain obs[{label_key!r}]")

	raw_labels = adata.obs[label_key]
	valid = raw_labels.notna().to_numpy()
	labels = raw_labels.astype(str).str.strip().to_numpy()
	valid &= np.asarray([label.casefold() not in undefined_labels for label in labels])
	if not valid.any():
		raise ValueError(f"{path} has no defined annotation spots after filtering")

	# Prefer the explicitly preserved exp layer, while accepting ordinary h5ad X.
	expression_source = adata.layers["exp"] if "exp" in adata.layers else adata.X
	gene_indices = adata.var_names.get_indexer(genes)
	if np.any(gene_indices < 0):
		raise RuntimeError(f"Gene alignment changed unexpectedly while reading {path}")
	expression = _dense_float32(expression_source[:, gene_indices])[valid]
	if not np.isfinite(expression).all():
		raise ValueError(f"{path} contains NaN or infinite expression values")

	return SampleData(
		sample_id=path.stem,
		spot_ids=np.asarray(adata.obs_names.astype(str))[valid],
		expression=expression,
		labels=labels[valid],
	)


class AnnotationMLP(nn.Module):
	"""Small custom MLP for annotation classification."""

	def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32):
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(hidden_dim, num_classes),
		)

	def forward(self, expression: torch.Tensor) -> torch.Tensor:
		return self.network(expression)


def fit_standardizer(expression: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Compute per-gene statistics from training data only."""
	mean = expression.mean(axis=0, dtype=np.float64).astype(np.float32)
	std = expression.std(axis=0, dtype=np.float64).astype(np.float32)
	std[std < 1e-6] = 1.0
	return mean, std


def standardize(expression: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
	return ((expression - mean) / std).astype(np.float32, copy=False)


def encode_labels(labels: np.ndarray, class_to_index: dict[str, int]) -> np.ndarray:
	return np.asarray([class_to_index[label] for label in labels], dtype=np.int64)


def class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
	counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
	if np.any(counts == 0):
		raise ValueError("Every model class must occur in the current training data")
	weights = len(labels) / (num_classes * counts)
	return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(
	expression: np.ndarray,
	labels: np.ndarray,
	batch_size: int,
	seed: int,
) -> DataLoader:
	dataset = TensorDataset(torch.from_numpy(expression), torch.from_numpy(labels))
	generator = torch.Generator().manual_seed(seed)
	return DataLoader(
		dataset,
		batch_size=min(batch_size, len(dataset)),
		shuffle=True,
		num_workers=0,
		generator=generator,
	)


def train_mlp(
	x_train: np.ndarray,
	y_train: np.ndarray,
	input_dim: int,
	num_classes: int,
	hidden_dim: int,
	max_epochs: int,
	batch_size: int,
	learning_rate: float,
	weight_decay: float,
	seed: int,
	device: torch.device,
	checkpoint_expression: np.ndarray,
	checkpoint_epochs: tuple[int, ...],
	checkpoint_callback: Callable[[int, np.ndarray], None] | None = None,
) -> dict[int, np.ndarray]:
	"""Train the custom MLP for exactly ``max_epochs`` epochs."""
	set_global_seed(seed)
	model = AnnotationMLP(input_dim, num_classes, hidden_dim).to(device)
	loss_fn = nn.CrossEntropyLoss(class_weights(y_train, num_classes, device))
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=learning_rate, weight_decay=weight_decay
	)
	loader = make_loader(x_train, y_train, batch_size, seed)
	checkpoint_set = set(checkpoint_epochs)
	checkpoint_predictions = {}

	for epoch in range(1, max_epochs + 1):
		model.train()
		for batch_x, batch_y in loader:
			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device)
			optimizer.zero_grad(set_to_none=True)
			loss = loss_fn(model(batch_x), batch_y)
			loss.backward()
			optimizer.step()

		if epoch in checkpoint_set:
			prediction = predict_mlp(model, checkpoint_expression, device)
			checkpoint_predictions[epoch] = prediction
			if checkpoint_callback is not None:
				checkpoint_callback(epoch, prediction)

	return checkpoint_predictions


def predict_mlp(model: AnnotationMLP, expression: np.ndarray, device: torch.device) -> np.ndarray:
	model.eval()
	with torch.no_grad():
		logits = model(torch.from_numpy(expression).to(device))
	return logits.argmax(dim=1).cpu().numpy()


def evaluate_leave_one_sample_out(
	samples: list[SampleData],
	seed: int,
	hidden_dim: int,
	max_epochs: int,
	batch_size: int,
	learning_rate: float,
	weight_decay: float,
	epoch_sweep: tuple[int, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
	metric_rows = []
	sweep_rows = []
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Training custom MLP on {device}; hidden dimension={hidden_dim}")
	checkpoint_epochs = tuple(sorted(set(epoch_sweep) | {max_epochs}))
	training_epochs = max(checkpoint_epochs)

	for fold_index, test_sample in enumerate(samples, start=1):
		train_samples = [sample for sample in samples if sample is not test_sample]
		x_train = np.concatenate([sample.expression for sample in train_samples])
		y_train_labels = np.concatenate([sample.labels for sample in train_samples])
		train_classes = sorted(set(y_train_labels), key=str.casefold)
		class_to_index = {label: index for index, label in enumerate(train_classes)}
		evaluable = np.asarray([label in class_to_index for label in test_sample.labels])
		unseen_labels = sorted(set(test_sample.labels[~evaluable]), key=str.casefold)
		if not evaluable.any():
			raise ValueError(
				f"Fold {test_sample.sample_id} has no test labels represented in training samples"
			)

		# Fit per-gene mean/std on non-test samples only, then train fixed epochs.
		train_mean, train_std = fit_standardizer(x_train)
		x_train = standardize(x_train, train_mean, train_std)
		y_train = encode_labels(y_train_labels, class_to_index)
		x_test = standardize(test_sample.expression[evaluable], train_mean, train_std)
		y_true = test_sample.labels[evaluable]

		def record_checkpoint(epoch: int, predicted_indices: np.ndarray) -> None:
			y_pred = np.asarray([train_classes[index] for index in predicted_indices])
			row = {
				"fold": fold_index,
				"test_sample": test_sample.sample_id,
				"epochs": epoch,
				"train_spots": len(y_train),
				"test_spots": len(test_sample.labels),
				"evaluated_spots": int(evaluable.sum()),
				"unseen_label_spots": int((~evaluable).sum()),
				"unseen_labels": ";".join(unseen_labels),
				"accuracy": accuracy_score(y_true, y_pred),
				"balanced_accuracy": recall_score(
					y_true,
					y_pred,
					labels=sorted(set(y_true), key=str.casefold),
					average="macro",
					zero_division=0,
				),
				"macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
				"weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
			}
			if epoch == max_epochs:
				metric_rows.append(row)
			if epoch in epoch_sweep:
				sweep_rows.append(row.copy())
				print(
					f"  TEST checkpoint | sample={test_sample.sample_id} | "
					f"epoch={epoch} | accuracy={row['accuracy']:.4f} | "
					f"balanced_accuracy={row['balanced_accuracy']:.4f} | "
					f"macro-F1={row['macro_f1']:.4f} | "
					f"weighted-F1={row['weighted_f1']:.4f}"
				)

		train_mlp(
			x_train=x_train,
			y_train=y_train,
			input_dim=x_train.shape[1],
			num_classes=len(train_classes),
			hidden_dim=hidden_dim,
			max_epochs=training_epochs,
			batch_size=batch_size,
			learning_rate=learning_rate,
			weight_decay=weight_decay,
			seed=seed + fold_index,
			device=device,
			checkpoint_expression=x_test,
			checkpoint_epochs=checkpoint_epochs,
			checkpoint_callback=record_checkpoint,
		)

		final_row = metric_rows[-1]
		print(
			f"Fold {fold_index}/{len(samples)} test={test_sample.sample_id}, "
			f"epochs={max_epochs}: "
			f"accuracy={final_row['accuracy']:.4f}, "
			f"macro-F1={final_row['macro_f1']:.4f}"
		)

	metrics = pd.DataFrame(metric_rows)
	metrics = append_mean_row(metrics, max_epochs)

	if not sweep_rows:
		return metrics, None
	sweep = pd.DataFrame(sweep_rows)
	sweep_parts = []
	for epoch in epoch_sweep:
		epoch_rows = sweep[sweep["epochs"] == epoch].copy()
		sweep_parts.append(append_mean_row(epoch_rows, epoch))
	return metrics, pd.concat(sweep_parts, ignore_index=True)


def append_mean_row(metrics: pd.DataFrame, epochs: int) -> pd.DataFrame:
	mean_row = {
		"fold": "mean",
		"test_sample": "all_samples",
		"epochs": epochs,
		"train_spots": pd.NA,
		"test_spots": metrics["test_spots"].sum(),
		"evaluated_spots": metrics["evaluated_spots"].sum(),
		"unseen_label_spots": metrics["unseen_label_spots"].sum(),
		"unseen_labels": "",
	}
	for column in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
		mean_row[column] = metrics[column].mean()
	return pd.concat([metrics, pd.DataFrame([mean_row])], ignore_index=True)


def run_evaluation(
	data_dir: str | Path,
	gene_list_path: str | Path,
	output_dir: str | Path = DEFAULT_OUTPUT_DIR,
	sample_ids: list[str] | None = None,
	label_key: str = "label",
	undefined_labels: tuple[str, ...] = DEFAULT_UNDEFINED_LABELS,
	gene_num: int | None = DEFAULT_GENE_NUM,
	seed: int = 42,
	hidden_dim: int = DEFAULT_HIDDEN_DIM,
	epochs: int = 100,
	batch_size: int = 128,
	learning_rate: float = DEFAULT_LEARNING_RATE,
	weight_decay: float = 1e-4,
	test_epoch_sweep: bool = False,
) -> pd.DataFrame:
	set_global_seed(seed)
	gene_path, requested_genes = read_gene_list(gene_list_path, gene_num=gene_num)
	paths = find_annotated_paths(data_dir, sample_ids)
	genes = common_selected_genes(paths, requested_genes)
	undefined = {label.strip().casefold() for label in undefined_labels}
	samples = [load_sample(path, genes, label_key, undefined) for path in paths]

	print(
		f"Loaded {len(samples)} annotated samples and {sum(len(s.labels) for s in samples)} "
		f"defined spots; using {len(genes)}/{len(requested_genes)} requested genes"
	)
	for sample in samples:
		counts = pd.Series(sample.labels).value_counts().to_dict()
		print(f"  {sample.sample_id}: {len(sample.labels)} spots, labels={counts}")

	metrics, sweep_metrics = evaluate_leave_one_sample_out(
		samples=samples,
		seed=seed,
		hidden_dim=hidden_dim,
		max_epochs=epochs,
		batch_size=batch_size,
		learning_rate=learning_rate,
		weight_decay=weight_decay,
		epoch_sweep=TEST_EPOCH_SWEEP if test_epoch_sweep else (),
	)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	prefix = gene_path.stem
	metrics_path = output_dir / f"{prefix}_sample_folds.csv"
	metrics.to_csv(metrics_path, index=False)
	print(f"Saved fold metrics: {metrics_path}")
	if sweep_metrics is not None:
		sweep_path = output_dir / f"{prefix}_epoch_sweep.csv"
		sweep_metrics.to_csv(sweep_path, index=False)
		print(f"Saved temporary test epoch sweep: {sweep_path}")
	return metrics


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"gene_lists",
		nargs="+",
		help="One or more gene-list JSON paths or names under datasets/BC1/rank_list",
	)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--samples", nargs="+", help="Optional annotated sample IDs")
	parser.add_argument("--label-key", default="label")
	parser.add_argument(
		"--undefined-labels",
		nargs="*",
		default=list(DEFAULT_UNDEFINED_LABELS),
		help="Case-insensitive annotation values to remove",
	)
	parser.add_argument(
		"--gene-num",
		type=int,
		default=DEFAULT_GENE_NUM,
		help=f"Use the first N genes from the ranked list (default: {DEFAULT_GENE_NUM})",
	)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument(
		"--test-epoch-sweep",
		action="store_true",
		help=(
			"TEMPORARY diagnostic: evaluate the outer test sample at epochs "
			+ ", ".join(map(str, TEST_EPOCH_SWEEP))
		),
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	comparison_rows = []
	for index, gene_list in enumerate(args.gene_lists, start=1):
		if len(args.gene_lists) > 1:
			print(
				f"\n==================== Gene list {index}/{len(args.gene_lists)}: "
				f"{gene_list} ===================="
			)
		metrics = run_evaluation(
			data_dir=args.data_dir,
			gene_list_path=gene_list,
			output_dir=args.output_dir,
			sample_ids=args.samples,
			label_key=args.label_key,
			undefined_labels=tuple(args.undefined_labels),
			gene_num=args.gene_num,
			seed=args.seed,
			hidden_dim=DEFAULT_HIDDEN_DIM,
			epochs=args.epochs,
			batch_size=args.batch_size,
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			test_epoch_sweep=args.test_epoch_sweep,
		)
		mean_metrics = metrics.iloc[-1]
		comparison_rows.append(
			{
				"gene_list": resolve_gene_list_path(gene_list).stem,
				"accuracy": mean_metrics["accuracy"],
				"balanced_accuracy": mean_metrics["balanced_accuracy"],
				"macro_f1": mean_metrics["macro_f1"],
				"weighted_f1": mean_metrics["weighted_f1"],
			}
		)

	if len(comparison_rows) > 1:
		comparison_path = Path(args.output_dir) / "gene_list_comparison.csv"
		pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
		print(f"\nSaved gene-list comparison: {comparison_path}")


if __name__ == "__main__":
	main()


'''
python evaluation/train_cluster.py \
	tri_hmg \
	tri_hvg \
	tri_hmhvg \
	stem_hmhvg \
	her2st_DEG_200 \
	conch_cca_pca50 \
	conch_mlp \
	spatial_moran \
	conch_fscore_0p5 \
	--epochs 300

python evaluation/train_cluster.py \
	--epochs 300 \
	--test-epoch-sweep \
	tri_hmg \
	her2st_DEG_200 \
	conch_fscore_0p5 
'''
