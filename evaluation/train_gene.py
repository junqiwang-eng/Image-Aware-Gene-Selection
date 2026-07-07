import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from datasets import (
	get_sample_ids_by_patient,
	get_train_test_datasets,
	load_data,
	resolve_rank_list_path,
)
from module import STPred


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

	json_name = json_path.stem
	task_name = f"{test_patient}_{json_name}"
	wandb_logger = WandbLogger(
		project="ST_Gene_Prediction",
		name=f"Test_{test_patient}",
		group=task_name,
		config={
			"model_type": model_key,
			"in_features": input_features,
			"out_features": num_genes,
			"lr": 5e-5,
			"test_patient": test_patient,
		},
	)

	trainer = pl.Trainer(
		max_epochs=150,
		accelerator="auto",
		devices="auto",
		enable_checkpointing=False,
		logger=wandb_logger,
		log_every_n_steps=10,
	)
	trainer.fit(model, train_dataloaders=train_loader)
	trainer.test(model, dataloaders=test_loader)

	preds = torch.cat([x["pred"] for x in model.test_outputs]).numpy()
	trues = torch.cat([x["true"] for x in model.test_outputs]).numpy()

	gene_metrics_dict = {}
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

		gene_metrics_dict[gene] = {"PCC": float(pcc), "SCC": float(scc)}
		fold_pcc_list.append(pcc)

	wandb_logger.log_metrics({"fold_mean_pcc_test": float(np.mean(fold_pcc_list))})

	import wandb

	wandb.finish()

	return gene_metrics_dict, float(np.mean(fold_pcc_list))


def run_fold_evaluation(
	data_dir,
	json_paths,
	model_key="ff",
	seed: int = 42,
	feature_name=None,
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

		all_genes_data = defaultdict(dict)
		active_patients = []

		for patient_name in patient_names:
			print(f"\n----- Test patient: {patient_name} -----")
			gene_metrics_dict, _ = run_single_patient_evaluation(
				data_dict=data_dict,
				test_sample_ids=patient_to_sample_ids[patient_name],
				test_patient=patient_name,
				json_path=json_path,
				model_key=model_key,
			)

			active_patients.append(patient_name)
			for gene, metrics in gene_metrics_dict.items():
				all_genes_data[gene][patient_name] = metrics

		rows = []
		for gene, records in all_genes_data.items():
			row = {"Gene": gene}
			pcc_values = []
			scc_values = []
			for patient_name in active_patients:
				metrics = records.get(patient_name, {})
				pcc_val = metrics.get("PCC", np.nan)
				scc_val = metrics.get("SCC", np.nan)
				row[f"{patient_name}_PCC"] = pcc_val
				row[f"{patient_name}_SCC"] = scc_val
				if not np.isnan(pcc_val):
					pcc_values.append(pcc_val)
				if not np.isnan(scc_val):
					scc_values.append(scc_val)

			row["Average_PCC"] = np.mean(pcc_values) if pcc_values else 0.0
			row["Average_SCC"] = np.mean(scc_values) if scc_values else 0.0
			rows.append(row)

		df_final = pd.DataFrame(rows)
		df_final = df_final.sort_values(by="Average_PCC", ascending=False).reset_index(drop=True)

		patient_cols = []
		for patient_name in active_patients:
			patient_cols.extend([f"{patient_name}_PCC", f"{patient_name}_SCC"])
		df_final = df_final[["Gene", "Average_PCC", "Average_SCC"] + patient_cols]

		ranking_file = f"{json_stem}_BC1_detailed_genes_metrics.csv"
		df_final.to_csv(ranking_file, index=False)
		print(f"\nSaved result: {ranking_file}")
		print(df_final.head(10).to_string(index=False))


if __name__ == "__main__":
	run_fold_evaluation(
		data_dir="D:/Work/Gene select/datasets/features/BC1",
		feature_name="conch",
		json_paths=["conch"],
		model_key="ff",
	)
