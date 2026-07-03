#!/usr/bin/env python3
"""
Combine one or more extracted feature .h5 files into AnnData .h5ad files.

The feature .h5 files produced by preprocessing/feature_extraction.py contain:
barcode, coords, count, exp, features, gene_list.

This script stores:
- exp -> adata.X
- count -> adata.layers["counts"]
- coords -> adata.obsm["spatial"]
- each feature matrix -> adata.obsm["X_<feature_name>"]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData


REQUIRED_KEYS = ("barcode", "coords", "count", "exp", "features", "gene_list")


def _decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return str(value)


def decode_strings(values: np.ndarray) -> list[str]:
    """Decode HDF5 string arrays, including possible (n, 1) barcode arrays."""
    arr = np.asarray(values)
    if arr.ndim > 1 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return [_decode_scalar(value) for value in arr.tolist()]


def read_feature_h5(path: Path) -> dict[str, np.ndarray | list[str]]:
    with h5py.File(path, "r") as file:
        missing = [key for key in REQUIRED_KEYS if key not in file]
        if missing:
            raise KeyError(f"{path} is missing required datasets: {missing}")

        data = {
            "barcode": decode_strings(file["barcode"][:]),
            "coords": file["coords"][:],
            "count": file["count"][:],
            "exp": file["exp"][:],
            "features": file["features"][:],
            "gene_list": decode_strings(file["gene_list"][:]),
        }

    n_obs = len(data["barcode"])
    if data["features"].shape[0] != n_obs:
        raise ValueError(
            f"{path} has {data['features'].shape[0]} feature rows but "
            f"{n_obs} barcodes."
        )
    return data


def feature_name_from_path(path: Path) -> str:
    """Use the parent directory name as the model/feature name."""
    return clean_feature_name(path.parent.name)


def clean_feature_name(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_")


def assert_same_spots(reference: dict, current: dict, path: Path) -> None:
    if reference["barcode"] != current["barcode"]:
        raise ValueError(f"{path} has different barcode order/content.")

    if not np.array_equal(reference["coords"], current["coords"]):
        raise ValueError(f"{path} has different coords.")

    if reference["gene_list"] != current["gene_list"]:
        raise ValueError(f"{path} has different gene_list.")


def make_adata(
    feature_paths: Sequence[Path],
    feature_names: Sequence[str] | None = None,
) -> AnnData:
    if not feature_paths:
        raise ValueError("At least one feature .h5 file is required.")

    if feature_names is None:
        feature_names = [feature_name_from_path(path) for path in feature_paths]
    if len(feature_names) != len(feature_paths):
        raise ValueError("--names must have the same length as --feature-files.")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError(f"Feature names must be unique, got: {feature_names}")

    reference = read_feature_h5(feature_paths[0])
    obs = pd.DataFrame(index=pd.Index(reference["barcode"], name="barcode"))
    var = pd.DataFrame(index=pd.Index(reference["gene_list"], name="gene"))

    adata = AnnData(X=reference["exp"], obs=obs, var=var)
    adata.layers["counts"] = reference["count"]
    adata.obsm["spatial"] = reference["coords"]

    for path, name in zip(feature_paths, feature_names):
        current = reference if path == feature_paths[0] else read_feature_h5(path)
        assert_same_spots(reference, current, path)
        adata.obsm[f"X_{name}"] = current["features"]

    adata.uns["feature_sources"] = {
        name: str(path) for name, path in zip(feature_names, feature_paths)
    }
    return adata


def combine_feature_files(
    feature_paths: Sequence[Path],
    output_path: Path,
    feature_names: Sequence[str] | None = None,
) -> Path:
    adata = make_adata(feature_paths, feature_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    return output_path


def find_common_stems(feature_dirs: Sequence[Path]) -> list[str]:
    stem_sets = []
    for directory in feature_dirs:
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        stems = {path.stem for path in directory.glob("*.h5")}
        if not stems:
            raise FileNotFoundError(f"No .h5 files found in {directory}")
        stem_sets.append(stems)
    common_stems = sorted(set.intersection(*stem_sets))
    if not common_stems:
        raise FileNotFoundError("No common .h5 sample names found across directories.")
    return common_stems


def combine_feature_dirs(
    feature_dirs: Sequence[Path],
    output_dir: Path,
    feature_names: Sequence[str] | None = None,
) -> list[Path]:
    if feature_names is None:
        feature_names = [clean_feature_name(directory.name) for directory in feature_dirs]
    if len(feature_names) != len(feature_dirs):
        raise ValueError("--names must have the same length as --feature-dirs.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for stem in find_common_stems(feature_dirs):
        feature_paths = [directory / f"{stem}.h5" for directory in feature_dirs]
        output_path = output_dir / f"{stem}.h5ad"
        combine_feature_files(feature_paths, output_path, feature_names)
        output_paths.append(output_path)
        print(f"wrote {output_path}")
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple feature .h5 files into .h5ad."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--feature-files",
        nargs="+",
        type=Path,
        help="Feature .h5 files for the same sample.",
    )
    mode.add_argument(
        "--feature-dirs",
        nargs="+",
        type=Path,
        help="Directories containing same-named feature .h5 files.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Names used in obsm as X_<name>. Defaults to parent/dir names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .h5ad path for --feature-files, or output directory for --feature-dirs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.feature_files:
        output_path = combine_feature_files(args.feature_files, args.output, args.names)
        print(f"wrote {output_path}")
    else:
        output_paths = combine_feature_dirs(args.feature_dirs, args.output, args.names)
        print(f"wrote {len(output_paths)} h5ad files to {args.output}")


if __name__ == "__main__":
    main()


"""
Example: batch combine same-named feature .h5 files from multiple directories.

python clustering/combination.py \
    --feature-dirs \
        datasets/BC1/features/resnet50_trunc \
        datasets/BC1/features/uni_v1 \
        datasets/BC1/features/conch_v1 \
        datasets/BC1/features/midnight \
    --names resnet50 uni conch midnight \
    --output datasets/BC1/combined_features
"""
