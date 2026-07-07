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


def subset_data_dict(data_dict, sample_ids):
	return {sample_id: data_dict[sample_id] for sample_id in sample_ids}


def run_single_json_evaluation(
	data_dict,
	patient_name,
	json_path,
	model_key="ff",
):
	json_path = resolve_rank_list_path(json_path)
	all_sample_ids = list(data_dict.keys())
	with open(json_path, "r", encoding="utf-8") as f:
		target_genes = json.load(f)

	print(f"Loaded {len(all_sample_ids)} samples; rank list genes: {len(target_genes)}")

	all_folds_pcc = defaultdict(list)
	all_folds_scc = defaultdict(list)
	use_pin_memory = torch.cuda.is_available()

	for fold_idx, test_id in enumerate(all_sample_ids):
		print(f"\n>>> Fold {fold_idx + 1}/{len(all_sample_ids)} | test sample: {test_id} <<<")

		train_dataset, test_dataset = get_train_test_datasets(
			data_dict=data_dict,
			test_sample_ids=[test_id],
			json_path=json_path,
		)

		num_genes = len(train_dataset.common_genes)
		input_features = train_dataset.img_feature_dim
		print(f"Effective genes for this fold: {num_genes}; input features: {input_features}")

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
		task_name = f"{patient_name}_{json_name}"
		wandb_logger = WandbLogger(
			project="ST_Gene_Prediction",
			name=f"Fold_{fold_idx + 1}_Test_{test_id}",
			group=task_name,
			config={
				"model_type": model_key,
				"in_features": input_features,
				"out_features": num_genes,
				"lr": 5e-5,
				"test_sample": test_id,
				"patient": patient_name,
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

			all_folds_pcc[gene].append(pcc)
			all_folds_scc[gene].append(scc)
			fold_pcc_list.append(pcc)

		wandb_logger.log_metrics({"fold_mean_pcc_test": float(np.mean(fold_pcc_list))})

		import wandb

		wandb.finish()

	summary_results = []
	for gene in all_folds_pcc.keys():
		summary_results.append(
			{
				"Gene": gene,
				"Mean_PCC": np.mean(all_folds_pcc[gene]),
				"Mean_SCC": np.mean(all_folds_scc[gene]),
			}
		)

	df_sorted = (
		pd.DataFrame(summary_results)
		.sort_values(by="Mean_PCC", ascending=False)
		.reset_index(drop=True)
	)
	top_k_rows = []
	for top_k in [10, 50, 100, 250]:
		effective_k = min(top_k, len(df_sorted))
		top_k_rows.append(
			{
				"Patient": patient_name,
				"Top_K": f"top{top_k}",
				"Mean_PCC": float(df_sorted.head(effective_k)["Mean_PCC"].mean()),
			}
		)

	return pd.DataFrame(top_k_rows)


def run_fold_evaluation(
	data_dir,
	json_paths,
	model_key="ff",
	seed: int = 42,
	feature_name=None,
):
	if isinstance(json_paths, (str, Path)):
		json_paths = [json_paths]

	data_dict = load_data(data_dir, feature_name=feature_name)
	patient_to_sample_ids = get_sample_ids_by_patient(data_dict)
	patient_names = sorted(patient_to_sample_ids)
	if not patient_names:
		raise ValueError(f"No patient labels found in {data_dir}.")

	print(f"Will evaluate {len(patient_names)} patients: {', '.join(patient_names)}")

	for json_path in json_paths:
		json_path = resolve_rank_list_path(json_path)
		json_stem = json_path.stem
		print(f"\n==================== Rank list: {json_stem} ====================")

		patient_tables = []
		for patient_name in patient_names:
			print(f"\n----- Patient: {patient_name} -----")
			patient_data = subset_data_dict(data_dict, patient_to_sample_ids[patient_name])
			set_global_seed(seed)
			df_patient = run_single_json_evaluation(
				data_dict=patient_data,
				patient_name=patient_name,
				json_path=json_path,
				model_key=model_key,
			)
			patient_tables.append(df_patient.reset_index(drop=True))

		df_output = pd.concat(patient_tables, axis=0, ignore_index=True)
		output_file = f"{json_stem}_BC1_patient_topk_pcc.csv"
		df_output.to_csv(output_file, index=False)
		print(f"\nSaved result: {output_file}")


if __name__ == "__main__":
	run_fold_evaluation(
		data_dir="D:/Work/Gene select/datasets/features/BC1",
		feature_name="conch",
		json_paths=["conch"],
		model_key="ff",
	)
