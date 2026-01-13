import ast
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_SPLITS: Tuple[str, ...] = ("train", "val", "test")


def project_root() -> Path:
    """Return the save_features directory (root for all code)."""
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    """Return the workspace root where data/features live."""
    return project_root().parent


def default_data_root() -> Path:
    """Return the default data directory (data/)."""
    return workspace_root() / "data"


def default_images_dir(data_root: Path | None = None) -> Path:
    """Return the default path to the image directory."""
    base = data_root or default_data_root()
    return base / "images"


def default_output_root(model_name: str) -> Path:
    """Return features/{model_name}."""
    return workspace_root() / "features" / model_name


def resolve_csv_paths(
    data_root: Path, splits: Sequence[str] | None = None
) -> Dict[str, Path]:
    """Construct a mapping from split names to CSV paths."""
    split_names = splits or DEFAULT_SPLITS
    mapping: Dict[str, Path] = {}
    for split in split_names:
        csv_path = data_root / f"spica_{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} does not exist")
        mapping[split] = csv_path
    return mapping


def parse_refs(refs_raw: str) -> List[str]:
    """Parse the refs column into a Python list and skip non-string entries."""
    try:
        parsed = ast.literal_eval(refs_raw)
    except (SyntaxError, ValueError):
        return []

    refs: List[str] = []
    if isinstance(parsed, (list, tuple)):
        for ref in parsed:
            if ref is None:
                continue
            text = str(ref).strip()
            if text:
                refs.append(text)
    return refs


def iter_dataframe_chunks(df: pd.DataFrame, num_splits: int):
    """
    Split a DataFrame into num_splits parts and yield (chunk_idx, chunk_df).
    Each chunk_df receives a __global_index__ column.
    """
    if num_splits <= 0:
        raise ValueError("num_splits must be a positive integer")

    split_indices = np.array_split(np.arange(len(df)), num_splits)
    for chunk_idx, indices in enumerate(split_indices):
        chunk_df = df.iloc[indices].copy()
        chunk_df["__global_index__"] = indices
        chunk_df = chunk_df.reset_index(drop=True)
        yield chunk_idx, chunk_df


def ensure_output_dir(base_dir: Path, dataset_name: str) -> Path:
    """
    Create (if needed) and return the output directory for a dataset,
    e.g., features/beit3/train.
    """
    target = base_dir / dataset_name
    target.mkdir(parents=True, exist_ok=True)
    return target


def chunk_file_prefix(dataset_name: str, chunk_idx: int) -> str:
    """Return the file-name prefix for a chunk (always clip_)."""
    return f"clip"


def make_mt_key(split_name: str, imgid: str, global_index: int) -> str:
    """Generate the dict key for MT features."""
    return f"{split_name}__{imgid}__mt__{global_index}"


def make_ref_key(
    split_name: str, imgid: str, global_index: int, local_ref_idx: int
) -> str:
    """Generate the dict key for reference-text features."""
    return f"{split_name}__{imgid}__ref__{global_index}__{local_ref_idx}"

