import argparse
import contextlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from datasets import (
	get_sample_ids_by_patient,
	get_train_test_datasets,
	load_data,
	resolve_rank_list_path,
)
from module import STPred


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP_K_ORDER = ["top10", "top50", "top100", "top250"]
MATMUL_PRECISION = "medium"
METRIC_FILES = {
	"Mean_PCC": PROJECT_ROOT / "fscore_pcc.csv",
	"Mean_SCC": PROJECT_ROOT / "fscore_scc.csv",
}
DISPLAY_NAME_OVERRIDES = {
	"conch": "Conch",
	"conch_v1": "Conch",
	"midnight": "Midnight",
	"resnet50": "Resnet",
	"resnet50_trunc": "Resnet",
	"uni": "Uni",
	"uni_v1": "Uni",
}


@contextlib.contextmanager
def file_lock(lock_path):
	lock_path.parent.mkdir(parents=True, exist_ok=True)
	with open(lock_path, "w", encoding="utf-8") as lock_file:
		try:
			import fcntl

			fcntl.flock(lock_file, fcntl.LOCK_EX)
			yield
			fcntl.flock(lock_file, fcntl.LOCK_UN)
		except ImportError:
			yield


class BestValMetricCallback(pl.Callback):
	def __init__(self, metric_name="val_pcc"):
		self.metric_name = metric_name
		self.best_value = None
		self.best_epoch = None

	def on_validation_epoch_end(self, trainer, pl_module):
		if trainer.sanity_checking:
			return

		metric = trainer.callback_metrics.get(self.metric_name)
		if metric is None:
			return

		metric_value = float(metric.detach().cpu())
		if self.best_value is None or metric_value > self.best_value:
			self.best_value = metric_value
			self.best_epoch = trainer.current_epoch

		pl_module.log("best_val_pcc", self.best_value, prog_bar=True)
		pl_module.log("best_epoch_by_val_pcc", float(self.best_epoch))


def set_global_seed(seed: int):
	if seed is None:
		return
	print(f"Set global seed: {seed}")
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	try:
		torch.cuda.manual_seed_all(seed)
	except Exception:
		pass
	pl.seed_everything(seed, workers=True)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def configure_torch_performance():
	if torch.cuda.is_available():
		torch.set_float32_matmul_precision(MATMUL_PRECISION)
		print(f"Set float32 matmul precision: {MATMUL_PRECISION}")


def display_experiment_part(name):
	name = str(name)
	base_name = name.removesuffix("_fscore")
	return DISPLAY_NAME_OVERRIDES.get(base_name, base_name.replace("_", " ").title().replace(" ", ""))


def make_experiment_name(feature_name, json_stem):
	feature_label = display_experiment_part(feature_name or "feature")
	rank_label = display_experiment_part(json_stem)
	return f"{feature_label}_{rank_label}"


def make_metric_frame(df_ranking, metric_col, experiment_name):
	return df_ranking[["Patient", "Top_K", metric_col]].rename(
		columns={metric_col: experiment_name}
	)


def update_metric_csv(path, df_ranking, metric_col, experiment_name):
	with file_lock(path.with_suffix(path.suffix + ".lock")):
		new_values = make_metric_frame(df_ranking, metric_col, experiment_name)
		if path.exists():
			result = pd.read_csv(path)
			if experiment_name in result.columns:
				result = result.drop(columns=[experiment_name])
			result = result.merge(new_values, on=["Patient", "Top_K"], how="outer")
		else:
			result = new_values

		row_order = {
			(str(row["Patient"]), str(row["Top_K"])): idx
			for idx, row in df_ranking[["Patient", "Top_K"]].iterrows()
		}
		result["_row_order"] = result.apply(
			lambda row: row_order.get((str(row["Patient"]), str(row["Top_K"])), len(row_order)),
			axis=1,
		)
		result = result.sort_values("_row_order").drop(columns=["_row_order"])
		result.to_csv(path, index=False)


def save_metric_csvs(df_ranking, feature_name, json_stem):
	experiment_name = make_experiment_name(feature_name, json_stem)
	for metric_col, output_path in METRIC_FILES.items():
		update_metric_csv(output_path, df_ranking, metric_col, experiment_name)
		print(f"Saved {metric_col} to {output_path} column {experiment_name}")


def load_bc1_data_by_patient(root_dir, feature_name=None):
	data_dict = load_data(root_dir, feature_name=feature_name)
	patient_to_sample_ids = get_sample_ids_by_patient(data_dict)
	patient_names = sorted(patient_to_sample_ids)
	if not patient_names:
		raise ValueError(f"No patient labels found in {root_dir}.")
	return data_dict, patient_to_sample_ids, patient_names


def run_single_patient_evaluation(
	data_dict,
	test_sample_ids,
	test_patient,
	json_path,
	model_key="ff",
	test_as_val=False,
):
	json_path = resolve_rank_list_path(json_path)
	with open(json_path, "r", encoding="utf-8") as f:
		target_genes = json.load(f)

	print(f"Loaded {len(data_dict)} samples; rank list genes: {len(target_genes)}")

	train_dataset, test_dataset = get_train_test_datasets(
		data_dict=data_dict,
		test_sample_ids=test_sample_ids,
		json_path=json_path,
	)

	num_genes = len(train_dataset.common_genes)
	input_features = train_dataset.img_feature_dim
	print(f"Effective genes for this fold: {num_genes}; input features: {input_features}")

	use_pin_memory = torch.cuda.is_available()
	train_loader = DataLoader(
		train_dataset,
		batch_size=512,
		shuffle=True,
		num_workers=4,
		persistent_workers=True,
		pin_memory=use_pin_memory,
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=512,
		shuffle=False,
		num_workers=4,
		persistent_workers=True,
		pin_memory=use_pin_memory,
	)

	model_params = {
		"in_features": input_features,
		"out_features": num_genes,
		"hidden_dims": [384],
		"dropout_p": 0.4,
	}
	model = STPred(model_key=model_key, model_params=model_params, lr=5e-5)

	callbacks = [BestValMetricCallback()] if test_as_val else None
	trainer = pl.Trainer(
		max_epochs=150,
		accelerator="auto",
		devices="auto",
		enable_checkpointing=False,
		logger=False,
		callbacks=callbacks,
		log_every_n_steps=10,
	)
	if test_as_val:
		trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
	else:
		trainer.fit(model, train_dataloaders=train_loader)
	trainer.test(model, dataloaders=test_loader)

	preds = torch.cat([x["pred"] for x in model.test_outputs]).numpy()
	trues = torch.cat([x["true"] for x in model.test_outputs]).numpy()

	gene_rows = []
	fold_pcc_list = []
	for i, gene in enumerate(train_dataset.common_genes):
		y_true = trues[:, i]
		y_pred = preds[:, i]
		if np.std(y_true) < 1e-6 or np.std(y_pred) < 1e-6:
			pcc, scc = 0.0, 0.0
		else:
			pcc, _ = pearsonr(y_true, y_pred)
			scc, _ = spearmanr(y_true, y_pred)
			pcc = 0.0 if np.isnan(pcc) else pcc
			scc = 0.0 if np.isnan(scc) else scc

		gene_rows.append({"Gene": gene, "Mean_PCC": pcc, "SCC": scc})
		fold_pcc_list.append(pcc)

	df_sorted = (
		pd.DataFrame(gene_rows)
		.sort_values(by="Mean_PCC", ascending=False)
		.reset_index(drop=True)
	)
	top_k_rows = []
	for top_k in [10, 50, 100, 250]:
		effective_k = min(top_k, len(df_sorted))
		top_k_rows.append(
			{
				"Patient": test_patient,
				"Top_K": f"top{top_k}",
				"Mean_PCC": float(df_sorted.head(effective_k)["Mean_PCC"].mean()),
				"Mean_SCC": float(df_sorted.head(effective_k)["SCC"].mean()),
			}
		)

	return pd.DataFrame(top_k_rows), float(np.mean(fold_pcc_list))


def run_fold_evaluation(
	data_dir,
	json_paths,
	model_key="ff",
	seed: int = 42,
	feature_name=None,
	test_as_val=False,
):
	if isinstance(json_paths, (str, Path)):
		json_paths = [json_paths]

	set_global_seed(seed)
	data_dict, patient_to_sample_ids, patient_names = load_bc1_data_by_patient(
		data_dir, feature_name=feature_name
	)
	print(f"Will evaluate {len(patient_names)} patients: {', '.join(patient_names)}")

	for json_path in json_paths:
		json_path = resolve_rank_list_path(json_path)
		json_stem = json_path.stem
		print(f"\n==================== Rank list: {json_stem} ====================")

		ranking_rows = []
		for patient_name in patient_names:
			print(f"\n----- Test patient: {patient_name} -----")
			df_patient_rank, _ = run_single_patient_evaluation(
				data_dict=data_dict,
				test_sample_ids=patient_to_sample_ids[patient_name],
				test_patient=patient_name,
				json_path=json_path,
				model_key=model_key,
				test_as_val=test_as_val,
			)
			ranking_rows.append(df_patient_rank)

		df_ranking = pd.concat(ranking_rows, ignore_index=True)
		average_rows = (
			df_ranking.groupby("Top_K", as_index=False)[["Mean_PCC", "Mean_SCC"]]
			.mean()
			.assign(Patient="Average")
		)
		average_rows = average_rows[["Patient", "Top_K", "Mean_PCC", "Mean_SCC"]]
		average_rows["Top_K"] = pd.Categorical(
			average_rows["Top_K"], categories=TOP_K_ORDER, ordered=True
		)
		average_rows = average_rows.sort_values("Top_K")
		average_rows["Top_K"] = average_rows["Top_K"].astype(str)
		df_ranking = pd.concat([df_ranking, average_rows], ignore_index=True)

		save_metric_csvs(df_ranking, feature_name, json_stem)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run leave-one-patient-out ST gene prediction evaluation."
	)
	parser.add_argument(
		"--data-dir",
		default=PROJECT_ROOT / "datasets" / "BC1" / "features",
		help="Directory containing feature .h5 files, or feature subfolders.",
	)
	parser.add_argument(
		"--feature-name",
		default="conch",
		help="Feature subfolder/name to load.",
	)
	parser.add_argument(
		"--json-paths",
		nargs="+",
		default=["resnet50"],
		help="One or more rank list names or JSON paths.",
	)
	parser.add_argument(
		"--model-key",
		default="ff",
		choices=["ff", "linear"],
		help="Model architecture key.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed.",
	)
	parser.add_argument(
		"--test-as-val",
		action="store_true",
		help="Use the test loader as validation during fit, for observing the best epoch.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	configure_torch_performance()
	run_fold_evaluation(
		data_dir=args.data_dir,
		feature_name=args.feature_name,
		json_paths=args.json_paths,
		model_key=args.model_key,
		seed=args.seed,
		test_as_val=args.test_as_val,
	)
'''
python evaluation/train.py \
	--data-dir datasets/BC1/features \
	--feature-name conch \
	--json-paths conch \
	--test-as-val
'''
