import json
import csv
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANK_LIST_DIR = PROJECT_ROOT / "datasets" / "BC1" / "rank_list"


def _decode_h5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_h5_string(value.item())
        return [_decode_h5_string(item) for item in value.tolist()]
    return str(value)


def _read_optional_string(f, key, default):
    if key not in f:
        return default
    return _decode_h5_string(f[key][()])


def load_patient_mapping(csv_path):
    if csv_path is None:
        return {}

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Patient mapping CSV not found: {csv_path}")

    patient_by_sample = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"sample_id", "patient"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing_columns)}")

        for row in reader:
            sample_id = row["sample_id"].strip()
            patient = row["patient"].strip()
            if sample_id and patient:
                patient_by_sample[sample_id] = patient
    return patient_by_sample


def _resolve_data_dir(data_dir, feature_name=None):
    data_dir = Path(data_dir)
    if feature_name is not None:
        feature_dir = data_dir / feature_name
        if feature_dir.is_dir():
            return feature_dir

    if list(data_dir.glob("*.h5")):
        return data_dir

    feature_dirs = sorted([path for path in data_dir.iterdir() if path.is_dir()])
    if len(feature_dirs) == 1:
        return feature_dirs[0]

    available = [path.name for path in feature_dirs]
    raise FileNotFoundError(
        f"No .h5 files found directly under {data_dir}. "
        f"Please pass feature_name. Available feature folders: {available}"
    )


def resolve_rank_list_path(json_path, rank_list_dir=DEFAULT_RANK_LIST_DIR):
    if json_path is None:
        return None

    path = Path(json_path)
    if path.exists():
        return path

    rank_list_dir = Path(rank_list_dir)
    candidates = []
    if path.suffix == ".json":
        candidates.append(rank_list_dir / path.name)
    else:
        name_parts = path.name.rsplit("_", 1)
        candidates.extend(
            [
                rank_list_dir / f"{path.name}_fscore.json",
                rank_list_dir / f"{path.name}.json",
            ]
        )
        if len(name_parts) == 2:
            feature_name, resolution = name_parts
            candidates.append(rank_list_dir / f"{feature_name}_fscore_{resolution}.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot find rank list '{json_path}'. Tried: {[str(p) for p in candidates]}"
    )


# 1. 核心数据加载：一次性真正读入内存，读完自动关闭文件
def load_data(data_dir, feature_name=None, mapping_csv=None):
    data_dir = _resolve_data_dir(data_dir, feature_name=feature_name)
    data_paths = sorted(data_dir.glob("*.h5"))
    data_dict = {}
    patient_by_sample = load_patient_mapping(mapping_csv)

    for path in data_paths:
        sample_id = path.stem
        # with 语句结束时，.h5 文件在最开始就会自动关闭
        with h5py.File(path, "r") as f:
            sample = _read_optional_string(f, "sample", sample_id)
            patient = _read_optional_string(
                f,
                "patient",
                patient_by_sample.get(sample_id, path.parent.name),
            )
            data_dict[sample_id] = {
                "gene_list": f["gene_list"][:],
                "count": f["count"][:],
                "exp": f["exp"][:],
                "sample": sample,
                "patient": patient,
                "img_feat": f["img_feat"][:] if "img_feat" in f else f["features"][:],
            }
    return data_dict


def get_sample_ids_by_patient(data_dict):
    patient_to_sample_ids = {}
    for sample_id, sample_data in data_dict.items():
        patient = sample_data.get("patient", sample_id)
        patient_to_sample_ids.setdefault(patient, []).append(sample_id)
    return patient_to_sample_ids


# 2. 基因匹配与映射工具函数
def get_common_genes(data_dict):
    sample_ids = list(data_dict.keys())
    geneset_list = [
        set(
            g.decode("utf-8") if isinstance(g, bytes) else str(g)
            for g in data_dict[sample_id]["gene_list"][:]
        )
        for sample_id in sample_ids
    ]
    common_genes = list(set.intersection(*geneset_list))
    return common_genes


def idx_map2common_genes(data_dict, common_genes):
    idx_map_dict = {}
    for sample_id, ds in data_dict.items():
        gene_list = [
            g.decode("utf-8") if isinstance(g, bytes) else str(g)
            for g in ds["gene_list"][:]
        ]
        gene_to_index = {gene: idx for idx, gene in enumerate(gene_list)}
        idx_map_dict[sample_id] = torch.tensor(
            [gene_to_index[gene] for gene in common_genes], dtype=torch.int32
        )
    return idx_map_dict


# 3. 标准 PyTorch Dataset（支持指定样本独立加载与独立归一化）
class IMG2STDataset(Dataset):

    def __init__(
        self,
        data_dict,
        sample_ids=None,
        ntop_genes=250,
        use_gene_list=None,
        json_path=None,
    ):
        """
        sample_ids: 明确指定当前 Dataset 仅加载哪些患者的数据。
                    例如训练集传入 ['patientB', 'patientC', ...]，测试集传入 ['patientA']
                    若为 None，则默认加载 data_dict 中的全量样本（保持对旧代码的向下兼容）
        """
        # 如果没传，默认用全部样本
        if sample_ids is None:
            sample_ids = list(data_dict.keys())
        if not sample_ids:
            raise ValueError("Dataset received no sample ids. Check patient labels and split settings.")

        self.sample_id_map = {
            sample_id: i for i, sample_id in enumerate(sample_ids)
        }

        # 🚀 核心修改：只拼接当前被选中的样本数据，实现物理隔离
        self.img_feat = torch.cat(
            [
                torch.tensor(data_dict[sample_id]["img_feat"])
                for sample_id in sample_ids
            ]
        )
        self.img_feature_dim = self.img_feat.shape[1]

        self.common_genes = get_common_genes(data_dict)
        self.idx_map = idx_map2common_genes(data_dict, self.common_genes)

        self.count = torch.cat(
            [
                torch.tensor(data_dict[sample_id]["count"]).float()[
                    :, self.idx_map[sample_id]
                ]
                for sample_id in sample_ids
            ]
        )

        # 优先读取指定的 JSON 基因列表
        if json_path is not None:
            json_path = resolve_rank_list_path(json_path)
            with open(json_path, "r", encoding="utf-8") as f:
                use_gene_list = json.load(f)

        # 如果没有传入指定的基因列表，则在当前“纯净样本内部”计算高变基因（HVG）
        if use_gene_list is None:
            orig_exp = torch.cat(
                [
                    torch.tensor(data_dict[sample_id]["exp"]).float()[
                        :, self.idx_map[sample_id]
                    ]
                    for sample_id in sample_ids
                ]
            )
            gene_variances = torch.var(orig_exp, dim=0)
            use_gene_idx_list = torch.argsort(gene_variances, descending=True)[
                :ntop_genes
            ]
        else:
            # 如果传入了基因列表（如你的前 250 基因），则精准匹配其在交集基因里的索引
            common_genes_str = [str(g) for g in self.common_genes]
            
            # 👇 【核心修改：加入安全容错过滤】
            # 只保留那些“确实同时存在于当前 .h5 文件交集里”的 JSON 基因
            matched_gene_idx = []
            actual_used_genes = []
            missing_genes = []

            for gene in use_gene_list:
                gene_str = str(gene)
                if gene_str in common_genes_str:
                    matched_gene_idx.append(common_genes_str.index(gene_str))
                    actual_used_genes.append(gene_str)
                else:
                    missing_genes.append(gene_str)

            # 如果发现有基因对不上，打印温馨提示，但绝对不报错，让程序丝滑往下走
            if len(missing_genes) > 0:
                print(f"⚠️  [Dataset 提示] 有 {len(missing_genes)} 个候选基因未在当前样本交集中找到（已自动跳过）。"
                      f"例如: {missing_genes[:3]}")
                print(f"📊 实际用于当前 Fold 训练/测试的有效基因数量: {len(matched_gene_idx)}")

            use_gene_idx_list = torch.tensor(matched_gene_idx, dtype=torch.long)

        # 切分表达量矩阵
        self.count = self.count[:, use_gene_idx_list]

        # 🚀 核心变化：这里的归一化（Log1p）和分母缩放将绝对在选定样本内部独立进行，杜绝泄露
        self.exp = torch.log(
            1.0e4 * self.count / self.count.sum(dim=1, keepdims=True) + 1.0
        )
        self.common_genes = [
            list(self.common_genes)[i] for i in use_gene_idx_list
        ]

        # 生成对应当前样本集的 sample_ids 标签矩阵
        self.sample_ids = torch.cat(
            [
                torch.tensor(
                    [self.sample_id_map[sample_id]]
                    * len(data_dict[sample_id]["count"])
                )
                for sample_id in sample_ids
            ]
        )

    def __len__(self):
        return len(self.img_feat)

    def __getitem__(self, idx):
        return {
            "img_feat": self.img_feat[idx],
            "exp": self.exp[idx],
            "count": self.count[idx],
            "sample_id": self.sample_ids[idx],
        }


# 4. 按样本划分 Train / Test（彻底物理隔离版）
def get_train_test_datasets(data_dict, test_sample_ids, **kwargs):
    """
    通过留一法（LOOCV）思路，干净利落地分离训练集和测试集。
    test_sample_ids: 被留出来当作测试集独立验证的患者 ID 列表，例如 ['patientA']
    """
    # 训练集样本 = 全部样本 - 测试集样本
    train_sample_ids = [
        sid for sid in data_dict.keys() if sid not in test_sample_ids
    ]


    # 物理创建纯净的训练集 Dataset
    train_dataset = IMG2STDataset(
        data_dict=data_dict, sample_ids=train_sample_ids, **kwargs
    )

    # 物理创建纯净的测试集 Dataset
    test_dataset = IMG2STDataset(
        data_dict=data_dict, sample_ids=test_sample_ids, **kwargs
    )

    # 直接返回两个完备的 PyTorch Dataset，外部的 DataLoader 无需任何改动即可完美对接
    return train_dataset, test_dataset
