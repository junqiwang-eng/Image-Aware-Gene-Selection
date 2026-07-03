from pathlib import Path
import pickle
import scanpy as sc
import h5py
import sys
import warnings
import numpy as np


def load_patches(file_name):
    with h5py.File(file_name) as f:
        barcode = f["barcode"][:]
        coords = f["coords"][:]
        patch_list = f["img"][:]
    return patch_list, barcode, coords


def preprocessing_adata(file_name):

    adata = sc.read_h5ad(file_name)

    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    # 使用错误抑制来处理 calculate_qc_metrics 中的 log1p 警告
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with np.errstate(divide='ignore', invalid='ignore'):
            sc.pp.calculate_qc_metrics(
                adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
            )


    adata = adata[adata.obs.pct_counts_mt < 15, :].copy()


    sc.pp.filter_cells(adata, min_genes=100)
    min_cells = max(1, int(np.ceil(adata.n_obs * 0.01)))
    sc.pp.filter_genes(adata, min_cells=min_cells)
 

    # Regularized count data

    cell_sums = adata.X.sum(axis=1)
    if hasattr(cell_sums, "A1"):
        cell_sums = cell_sums.A1
    else:
        cell_sums = np.asarray(cell_sums).ravel()
    nonzero_cells = cell_sums != 0
    adata = adata[nonzero_cells].copy()   
    

    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)


    return adata


def matching_adata_patch(adata, patch_list, barcode):
    code_list = []
    for bar in barcode:
        code_list.append(bar[0].decode("utf-8"))

    adata.obs["patch_index"] = None
    for code in adata.obs_names:
        try:
            idx = code_list.index(code)
            adata.obs.loc[code, "patch_index"] = idx
        except:
            pass

    adata_wpatch = adata[adata.obs["patch_index"].notnull()]


    spot_adata = adata_wpatch
    exp = spot_adata.X
    count = spot_adata.layers["counts"]

    patch_idx = spot_adata.obs["patch_index"].values
    patch = patch_list[patch_idx.astype(int)]
    img_coords = spot_adata.obsm["spatial"]
    gene_list = adata.var_names

    return exp, count, patch, img_coords, gene_list, spot_adata.obs_names


if len(sys.argv) != 2:
    print("Usage: python make_paired.py <data_directory>")
    sys.exit(1)

data_directory = sys.argv[1]
save_dir = Path(data_directory) / "paired_data"
save_dir.mkdir(parents=True, exist_ok=True)
patch_paths = sorted(Path(data_directory + "/patches").glob("*.h5"))
adata_dir = data_directory + "/st"

for patch_path in patch_paths:
    patch_list, barcode, coords = load_patches(patch_path)
    adata = preprocessing_adata(adata_dir + "/" + patch_path.stem + ".h5ad")
    exp, count, patch, img_coords, gene_list, barcode = matching_adata_patch(
        adata, patch_list, barcode
    )
    adata = sc.read_h5ad(adata_dir + "/" + patch_path.stem + ".h5ad")

    with open(
        f"{save_dir}/{patch_path.stem}.pkl",
        "wb",
    ) as f:
        pickle.dump(
            {
                "exp": exp,
                "count": count,
                "patch": patch,
                "img_coords": img_coords,
                "gene_list": gene_list,
                "barcode": barcode,
            },
            f,
        )

# python preprocessing/make_paired.py datasets/BC1
