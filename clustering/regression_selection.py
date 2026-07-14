#!/usr/bin/env python3
"""Rank genes by their predictability from CONCH image features.

All input samples are first aligned to the intersection of their genes.  Scores
are then computed independently for every patient.  The cross-patient ranking
uses the same rule as ``selection.py``: frequency in patient top-k (descending),
then mean within-patient rank (ascending).

Methods
-------
``cca``
    One sklearn CCA fit per gene, using the original CONCH embedding.
``cca_pca50``
    As above after patient-wise PCA (at most 50 components).
``ridge`` / ``mlp``
    One model per gene with five-fold shuffled CV. Each fold is scored by
    Pearson correlation on its held-out 20%; the five correlations are averaged.
    MLP uses PyTorch/CUDA and trains 256 independent gene models per batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "datasets" / "BC1" / "features" / "conch"
DEFAULT_MAPPING_CSV = PROJECT_ROOT / "datasets" / "BC1" / "BC1.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "BC1" / "rank_list"
METHODS = ("cca", "cca_pca50", "ridge", "mlp")


@contextmanager
def tqdm_joblib(progress_bar: tqdm):
	"""Update tqdm when joblib workers finish a batch of tasks."""
	original_callback = joblib.parallel.BatchCompletionCallBack

	class TqdmBatchCompletionCallback(original_callback):
		def __call__(self, *args, **kwargs):
			progress_bar.update(n=self.batch_size)
			return super().__call__(*args, **kwargs)

	joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
	try:
		yield progress_bar
	finally:
		joblib.parallel.BatchCompletionCallBack = original_callback
		progress_bar.close()


@dataclass
class Sample:
	name: str
	patient: str
	genes: list[str]
	features: np.ndarray
	expression: np.ndarray


def _strings(values: np.ndarray) -> list[str]:
	values = np.asarray(values)
	if values.ndim > 1 and values.shape[1] == 1:
		values = values[:, 0]
	return [v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values]


def load_mapping(path: Path | None) -> dict[str, str]:
	if path is None:
		return {}
	with path.open(encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		if not {"sample_id", "patient"}.issubset(reader.fieldnames or []):
			raise ValueError(f"{path} must contain sample_id and patient columns")
		return {row["sample_id"].strip(): row["patient"].strip() for row in reader}


def _scalar_string(dataset: h5py.Dataset) -> str:
	value = dataset[()]
	if isinstance(value, (bytes, np.bytes_)):
		return value.decode("utf-8")
	return str(value)


def load_samples(input_dir: Path, mapping_csv: Path | None, expression_key: str) -> list[Sample]:
	paths = [input_dir] if input_dir.is_file() else sorted(input_dir.rglob("*.h5"))
	if not paths:
		raise FileNotFoundError(f"No .h5 files found under {input_dir}")
	mapping = load_mapping(mapping_csv)
	samples = []
	for path in paths:
		with h5py.File(path, "r") as handle:
			required = {"features", "gene_list", expression_key}
			missing = required - set(handle.keys())
			if missing:
				raise KeyError(f"{path} is missing datasets: {sorted(missing)}")
			features = np.asarray(handle["features"][:], dtype=np.float32)
			expression = np.asarray(handle[expression_key][:], dtype=np.float32)
			genes = _strings(handle["gene_list"][:])
			patient = _scalar_string(handle["patient"]) if "patient" in handle else mapping.get(path.stem, "")
		if not patient:
			raise ValueError(f"No patient label for sample {path.stem}; provide --mapping-csv")
		if features.shape[0] != expression.shape[0] or expression.shape[1] != len(genes):
			raise ValueError(f"Inconsistent shapes in {path}: features={features.shape}, expression={expression.shape}, genes={len(genes)}")
		samples.append(Sample(path.stem, patient, genes, features, expression))
	return samples


def intersect_and_group(samples: list[Sample]) -> tuple[list[str], dict[str, tuple[np.ndarray, np.ndarray]]]:
	"""Align every sample to a stable, dataset-wide gene intersection."""
	common_set = set(samples[0].genes)
	for sample in samples[1:]:
		common_set.intersection_update(sample.genes)
	common = [gene for gene in samples[0].genes if gene in common_set]
	if not common:
		raise ValueError("No common genes across the CONCH feature files")
	grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
	for sample in samples:
		index = {gene: i for i, gene in enumerate(sample.genes)}
		y = sample.expression[:, [index[gene] for gene in common]]
		grouped.setdefault(sample.patient, []).append((sample.features, y))
	return common, {
		patient: (np.concatenate([v[0] for v in values]), np.concatenate([v[1] for v in values]))
		for patient, values in grouped.items()
	}


def pearson_columns(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
	a = y_true - y_true.mean(axis=0)
	b = y_pred - y_pred.mean(axis=0)
	denominator = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
	return np.divide(np.sum(a * b, axis=0), denominator, out=np.full(y_true.shape[1], np.nan), where=denominator > 0)


def _cca_gene_score(x: np.ndarray, target: np.ndarray, max_iter: int) -> float:
	"""Fit one gene's CCA; kept top-level so joblib can run it in a worker."""
	if np.std(target) == 0:
		return np.nan
	model = CCA(n_components=1, scale=True, max_iter=max_iter)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", ConvergenceWarning)
		try:
			x_c, y_c = model.fit_transform(x, target[:, None])
			return float(pearson_columns(y_c, x_c)[0])
		except (ValueError, np.linalg.LinAlgError):
			return np.nan


def cca_scores(
	x: np.ndarray,
	y: np.ndarray,
	pca50: bool,
	max_iter: int,
	n_jobs: int,
	progress_desc: str,
) -> np.ndarray:
	x = StandardScaler().fit_transform(x)
	if pca50:
		n_components = min(50, x.shape[0] - 1, x.shape[1])
		if n_components < 1:
			raise ValueError("Too few spots for PCA")
		x = PCA(n_components=n_components, random_state=0).fit_transform(x)
	# Process workers parallelize genes. joblib automatically memory-maps the
	# shared X matrix when it is large, avoiding one full copy per task.
	with tqdm_joblib(tqdm(total=y.shape[1], desc=progress_desc, unit="gene")):
		scores = Parallel(n_jobs=n_jobs, verbose=0, max_nbytes="10M")(
			delayed(_cca_gene_score)(x, y[:, gene_i], max_iter)
			for gene_i in range(y.shape[1])
		)
	return np.asarray(scores)


def _cv_gene_score(
	y: np.ndarray,
	fold_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
	ridge_alpha: float,
) -> float:
	"""Train and evaluate one independent Ridge model over all CV folds."""
	scores = []
	for x_train, x_valid, train, valid in fold_data:
		if np.std(y[train]) == 0:
			scores.append(np.nan)
			continue
		regressor = Ridge(alpha=ridge_alpha)
		regressor.fit(x_train, y[train])
		prediction = regressor.predict(x_valid)
		scores.append(pearson_columns(y[valid, None], prediction[:, None])[0])
	return float(np.nanmean(scores)) if not np.all(np.isnan(scores)) else np.nan


def cv_scores(
	x: np.ndarray,
	y: np.ndarray,
	method: str,
	folds: int,
	seed: int,
	ridge_alpha: float,
	mlp_hidden: int,
	mlp_max_iter: int,
	n_jobs: int,
	mlp_gene_batch: int,
	mlp_spot_batch: int,
	mlp_device: str,
	mlp_patience: int,
	progress_desc: str,
) -> np.ndarray:
	if len(x) < folds:
		raise ValueError(f"Need at least {folds} spots, got {len(x)}")
	fold_data = []
	for train, valid in KFold(folds, shuffle=True, random_state=seed).split(x):
		scaler = StandardScaler().fit(x[train])
		fold_data.append((scaler.transform(x[train]), scaler.transform(x[valid]), train, valid))
	if method == "mlp":
		return batched_mlp_scores(
			y,
			fold_data,
			seed,
			mlp_hidden,
			mlp_max_iter,
			mlp_gene_batch,
			mlp_spot_batch,
			mlp_device,
			mlp_patience,
			progress_desc,
		)
	with tqdm_joblib(tqdm(total=y.shape[1], desc=progress_desc, unit="gene")):
		scores = Parallel(n_jobs=n_jobs, verbose=0, max_nbytes="10M")(
			delayed(_cv_gene_score)(
				y[:, gene_i],
				fold_data,
				ridge_alpha,
			)
			for gene_i in range(y.shape[1])
		)
	return np.asarray(scores)


def batched_mlp_scores(
	y: np.ndarray,
	fold_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
	seed: int,
	hidden: int,
	max_iter: int,
	gene_batch: int,
	spot_batch: int,
	device_name: str,
	patience: int,
	progress_desc: str,
) -> np.ndarray:
	"""Train independent one-gene MLPs in GPU batches of ``gene_batch``."""
	try:
		import torch
	except ImportError as exc:
		raise ImportError("GPU MLP requires PyTorch. Install a CUDA-enabled torch build.") from exc
	if device_name == "auto":
		device_name = "cuda" if torch.cuda.is_available() else "cpu"
	if device_name.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("--mlp-device requests CUDA, but torch.cuda.is_available() is False")
	device = torch.device(device_name)
	print(f"MLP device: {device}; {gene_batch} independent genes per GPU batch")
	fold_scores = np.full((len(fold_data), y.shape[1]), np.nan, dtype=np.float32)
	progress = tqdm(
		total=len(fold_data) * y.shape[1],
		desc=progress_desc,
		unit="gene-fold",
	)

	def forward(x_tensor, w1, b1, w2, b2):
		h = torch.relu(torch.einsum("nd,gdh->ngh", x_tensor, w1) + b1)
		return torch.einsum("ngh,gh->ng", h, w2) + b2

	for fold_i, (x_train_all, x_test, train, test) in enumerate(fold_data):
		rng = np.random.default_rng(seed + fold_i)
		order = rng.permutation(len(train))
		n_internal_val = max(1, int(round(len(train) * 0.1)))
		fit_pos, stop_pos = order[n_internal_val:], order[:n_internal_val]
		if len(fit_pos) < 2:
			raise ValueError("Too few training spots for MLP internal validation")
		x_fit = torch.as_tensor(x_train_all[fit_pos], dtype=torch.float32, device=device)
		x_stop = torch.as_tensor(x_train_all[stop_pos], dtype=torch.float32, device=device)
		x_test_tensor = torch.as_tensor(x_test, dtype=torch.float32, device=device)
		for start in range(0, y.shape[1], gene_batch):
			end = min(start + gene_batch, y.shape[1])
			g = end - start
			y_train = y[train, start:end]
			y_fit = torch.as_tensor(y_train[fit_pos], dtype=torch.float32, device=device)
			y_stop = torch.as_tensor(y_train[stop_pos], dtype=torch.float32, device=device)
			torch.manual_seed(seed + fold_i * 100_000 + start)
			bound1 = np.sqrt(6.0 / (x_fit.shape[1] + hidden))
			bound2 = np.sqrt(6.0 / (hidden + 1))
			w1 = torch.empty(g, x_fit.shape[1], hidden, device=device).uniform_(-bound1, bound1).requires_grad_()
			b1 = torch.zeros(g, hidden, device=device, requires_grad=True)
			w2 = torch.empty(g, hidden, device=device).uniform_(-bound2, bound2).requires_grad_()
			b2 = torch.zeros(g, device=device, requires_grad=True)
			optimizer = torch.optim.Adam((w1, b1, w2, b2), lr=1e-4, weight_decay=1e-4)
			best_loss = torch.full((g,), torch.inf, device=device)
			stale = torch.zeros(g, dtype=torch.long, device=device)
			best = [parameter.detach().clone() for parameter in (w1, b1, w2, b2)]
			for _ in range(max_iter):
				active = stale < patience
				if not bool(active.any()):
					break
				permutation = torch.randperm(len(x_fit), device=device)
				for spot_start in range(0, len(x_fit), spot_batch):
					idx = permutation[spot_start:spot_start + spot_batch]
					prediction = forward(x_fit[idx], w1, b1, w2, b2)
					loss_by_gene = ((prediction - y_fit[idx]) ** 2).mean(dim=0)
					loss = loss_by_gene[active].mean()
					optimizer.zero_grad(set_to_none=True)
					loss.backward()
					optimizer.step()
				with torch.no_grad():
					val_loss = ((forward(x_stop, w1, b1, w2, b2) - y_stop) ** 2).mean(dim=0)
					improved = val_loss < best_loss - 1e-4
					best_loss[improved] = val_loss[improved]
					stale[improved] = 0
					stale[~improved] += 1
					for saved, current in zip(best, (w1, b1, w2, b2)):
						saved[improved] = current[improved]
			with torch.no_grad():
				prediction = forward(x_test_tensor, *best).cpu().numpy()
			fold_scores[fold_i, start:end] = pearson_columns(y[test, start:end], prediction)
			progress.update(end - start)
			del w1, b1, w2, b2, best, optimizer
	progress.close()
	return np.nanmean(fold_scores, axis=0)


def aggregate_patient_ranks(patient_tables: list[pd.DataFrame], top_k: int) -> pd.DataFrame:
	rows = []
	for table in patient_tables:
		rows.append(table.dropna(subset=["score"]).head(top_k)[["gene", "rank"]])
	selected = pd.concat(rows, ignore_index=True)
	return (selected.groupby("gene", as_index=False).agg(frequency=("rank", "size"), avg_rank=("rank", "mean"))
		.sort_values(["frequency", "avg_rank", "gene"], ascending=[False, True, True]).reset_index(drop=True))


def run(args: argparse.Namespace) -> None:
	samples = load_samples(args.input_dir, args.mapping_csv, args.expression_key)
	genes, patients = intersect_and_group(samples)
	print(f"Loaded {len(samples)} samples, {len(patients)} patients; common genes: {len(genes)}")
	args.output_dir.mkdir(parents=True, exist_ok=True)
	for method in args.methods:
		patient_tables = []
		for patient in sorted(patients):
			x, y = patients[patient]
			print(f"[{method}] {patient}: {len(x)} spots")
			if method.startswith("cca"):
				scores = cca_scores(
					x,
					y,
					method == "cca_pca50",
					args.cca_max_iter,
					args.n_jobs,
					f"{patient} {method}",
				)
			else:
				scores = cv_scores(
					x,
					y,
					method,
					args.folds,
					args.seed,
					args.ridge_alpha,
					args.mlp_hidden,
					args.mlp_max_iter,
					args.n_jobs,
					args.mlp_gene_batch,
					args.mlp_spot_batch,
					args.mlp_device,
					args.mlp_patience,
					f"{patient} {method}",
				)
			table = pd.DataFrame({"patient": patient, "gene": genes, "score": scores})
			table = table.sort_values(["score", "gene"], ascending=[False, True], na_position="last").reset_index(drop=True)
			table["rank"] = np.arange(1, len(table) + 1)
			patient_tables.append(table)
		aggregate = aggregate_patient_ranks(patient_tables, args.top_k)
		with (args.output_dir / f"conch_{method}.json").open("w", encoding="utf-8") as handle:
			json.dump(aggregate["gene"].head(args.top_k).tolist(), handle, ensure_ascii=False, indent=4)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
	parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--expression-key", choices=("exp", "count", "counts"), default="exp")
	parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
	parser.add_argument("--top-k", type=int, default=250)
	parser.add_argument("--folds", type=int, default=5)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--ridge-alpha", type=float, default=1.0)
	parser.add_argument("--mlp-hidden", type=int, default=50)
	parser.add_argument("--mlp-max-iter", type=int, default=100)
	parser.add_argument("--mlp-gene-batch", type=int, default=256)
	parser.add_argument("--mlp-spot-batch", type=int, default=256)
	parser.add_argument(
		"--mlp-device",
		default="cuda",
		help="PyTorch device for MLP: auto, cuda, cuda:0, or cpu.",
	)
	parser.add_argument("--mlp-patience", type=int, default=10)
	parser.add_argument("--cca-max-iter", type=int, default=300)
	parser.add_argument(
		"--n-jobs",
		type=int,
		default=-1,
		help="Parallel per-gene workers; -1 uses all available CPU cores.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	run(parse_args())


r'''
默认运行全部方法（CCA、PCA50+CCA、Ridge、MLP）：
python clustering/regression_selection.py

明确指定输入、患者映射和输出目录：
python clustering/regression_selection.py \
	--input-dir datasets/CSCC/features/conch \
	--mapping-csv datasets/CSCC/CSCC.csv \
	--output-dir datasets/CSCC/rank_list \
	--methods cca_pca50 ridge mlp \
	--n-jobs -1 \
	--mlp-hidden 50 \
	--mlp-max-iter 100 \
	--mlp-device cuda \
	--mlp-gene-batch 256 \
	--top-k 250


使用 8 个 CPU worker 运行原始维度 CCA：
python clustering/regression_selection.py \
	--methods cca_pca50 \
	--n-jobs -1

运行 Ridge 和 MLP，并设置交叉验证与 MLP 参数：
python clustering/regression_selection.py \
	--methods ridge mlp \
	--folds 5 \
	--n-jobs -1 \
	--mlp-hidden 50 \
	--mlp-max-iter 100 \
	--mlp-device cuda \
	--mlp-gene-batch 256 \
	--top-k 250 

使用原始 count 而不是 exp 作为基因表达：
python clustering/regression_selection.py \
	--expression-key count
'''
