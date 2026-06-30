import pandas as pd
import sys

import dotenv

dotenv.load_dotenv()

task_id = sys.argv[1]

df = pd.read_csv("./dataset/hest1k/HEST_v1_1_0.csv")

from huggingface_hub import login


import os

login(os.environ["HUGGINGFACE_TOKEN"])

import datasets

# task3 xenium
tasks_id_dict = {
    "CSCC": [f"NCBI{i}" for i in range(759, 771)],
    "BC1": [f"SPA{i}" for i in range(119, 155)],
    "BC2": [f"SPA{i}" for i in range(51, 119)],
}


def download_hest_data(ids_to_query, local_dir):
    list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
    dataset = datasets.load_dataset(
        "MahmoodLab/hest", cache_dir=local_dir, patterns=list_patterns
    )
    sub_meta_df = df.query("id in @ids_to_query")
    sub_meta_df.to_csv(f"{local_dir}/meta.csv", index=False)
    sub_meta_df.patient.value_counts().to_csv(f"{local_dir}/patient_counts.csv")


local_dir = f"./dataset/hest1k/{task_id}"
download_hest_data(tasks_id_dict[task_id], local_dir)
