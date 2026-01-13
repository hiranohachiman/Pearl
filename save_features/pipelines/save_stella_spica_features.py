from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from pipelines.spica_utils import (
    DEFAULT_SPLITS,
    chunk_file_prefix,
    default_data_root,
    default_output_root,
    ensure_output_dir,
    iter_dataframe_chunks,
    make_mt_key,
    make_ref_key,
    parse_refs,
    resolve_csv_paths,
)


def load_stella_components(
    model_path: Path,
    vector_dim: int,
    dense_subdir: str,
    device: torch.device,
):
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model.eval()

    projection = torch.nn.Linear(
        in_features=model.config.hidden_size, out_features=vector_dim
    )
    dense_path = model_path / dense_subdir / "pytorch_model.bin"
    if not dense_path.exists():
        raise FileNotFoundError(f"Linear layer weights not found: {dense_path}")
    state_dict = torch.load(dense_path, map_location=device)
    cleaned = {k.replace("linear.", ""): v for k, v in state_dict.items()}
    projection.load_state_dict(cleaned)
    projection = projection.to(device).eval()
    return model, tokenizer, projection


def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    projection,
    device: torch.device,
    max_length: int = 512,
):
    cleaned = [t if t else "[EMPTY]" for t in texts]
    tokens = tokenizer(
        cleaned,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    attention_mask = tokens["attention_mask"]

    with torch.no_grad():
        outputs = model(**tokens)
        last_hidden = outputs.last_hidden_state
        masked = last_hidden * attention_mask.unsqueeze(-1)
        sentence_vec = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        projected = projection(sentence_vec)
        return normalize(projected.cpu().numpy())


def process_dataset(
    csv_path: Path,
    dataset_name: str,
    output_root: Path,
    model,
    tokenizer,
    projection,
    batch_size: int,
    num_splits: int,
    device: torch.device,
) -> None:
    df = pd.read_csv(csv_path)
    print(f"[STELLA:{dataset_name}] samples: {len(df)}")
    dataset_output_dir = ensure_output_dir(output_root, dataset_name)

    total_mt = 0
    total_ref = 0

    for chunk_idx, chunk_df in iter_dataframe_chunks(df, num_splits):
        chunk_prefix = chunk_file_prefix(dataset_name, chunk_idx)
        mt_features_dict: Dict[str, np.ndarray] = {}
        ref_features_dict: Dict[str, np.ndarray] = {}

        for start_row in tqdm(
            range(0, len(chunk_df), batch_size),
            desc=f"{dataset_name}-chunk{chunk_idx + 1}",
        ):
            end_row = min(start_row + batch_size, len(chunk_df))
            batch_df = chunk_df.iloc[start_row:end_row]

            mt_rows: List[dict] = []
            for _, row in batch_df.iterrows():
                mt_rows.append(
                    {
                        "imgid": str(row["imgid"]),
                        "global_index": int(row["__global_index__"]),
                        "mt": str(row["mt"]) if row["mt"] is not None else "",
                        "refs": parse_refs(row["refs"]),
                    }
                )

            if not mt_rows:
                continue

            mt_texts = [row["mt"] for row in mt_rows]
            mt_features = encode_texts(mt_texts, tokenizer, model, projection, device)

            for idx, row in enumerate(mt_rows):
                # MT keys are the MT text itself
                mt_features_dict[row["mt"]] = mt_features[idx]

            flattened_refs = [
                ref for row in mt_rows for ref in (row["refs"] or [])
            ]
            if flattened_refs:
                ref_features = encode_texts(
                    flattened_refs, tokenizer, model, projection, device
                )
                offset = 0
                for row in mt_rows:
                    refs = row["refs"] or []
                    for local_idx, ref_text in enumerate(refs):
                        # Ref keys are the reference text itself
                        ref_features_dict[ref_text] = ref_features[offset]
                        offset += 1

        output_mt_path = dataset_output_dir / f"{chunk_prefix}_mt_features_{chunk_idx + 1}.npz"
        output_ref_path = dataset_output_dir / f"{chunk_prefix}_ref_features_{chunk_idx + 1}.npz"

        np.savez_compressed(output_mt_path, features=mt_features_dict)
        np.savez_compressed(output_ref_path, features=ref_features_dict)

        total_mt += len(mt_features_dict)
        total_ref += len(ref_features_dict)

        print(
            f"[STELLA:{dataset_name}] chunk {chunk_idx + 1}: "
            f"mt={len(mt_features_dict)}, ref={len(ref_features_dict)}"
        )

    print(f"[STELLA:{dataset_name}] completed: MT {total_mt}, reference {total_ref}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STELLA feature extractor for the SPICA dataset (MT/Ref only)"
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root("stella"),
        help="Output root directory for features",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names to process",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Directory that contains the STELLA model",
    )
    parser.add_argument(
        "--vector-dim", type=int, default=768, help="Output dimension after projection"
    )
    parser.add_argument(
        "--dense-subdir",
        type=str,
        default="2_Dense_768",
        help="Subdirectory that stores the linear-layer weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    csv_map = resolve_csv_paths(data_root, args.splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer, projection = load_stella_components(
        args.model_path, args.vector_dim, args.dense_subdir, device
    )

    for split_name, csv_path in csv_map.items():
        process_dataset(
            csv_path=csv_path,
            dataset_name=split_name,
            output_root=output_root,
            model=model,
            tokenizer=tokenizer,
            projection=projection,
            batch_size=args.batch_size,
            num_splits=args.num_splits,
            device=device,
        )


if __name__ == "__main__":
    main()

