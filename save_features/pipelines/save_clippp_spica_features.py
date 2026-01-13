from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from pipelines.spica_utils import (
    DEFAULT_SPLITS,
    chunk_file_prefix,
    default_data_root,
    default_images_dir,
    default_output_root,
    ensure_output_dir,
    iter_dataframe_chunks,
    make_mt_key,
    make_ref_key,
    parse_refs,
    resolve_csv_paths,
)

PAC_SCORE_ROOT = Path(__file__).resolve().parents[2] / "pacscore"
if not PAC_SCORE_ROOT.exists():
    raise FileNotFoundError(
        f"pacscore directory not found: {PAC_SCORE_ROOT}. "
        "Place pacscore under the repository root."
    )
if str(PAC_SCORE_ROOT) not in sys.path:
    sys.path.append(str(PAC_SCORE_ROOT))

from models import clip  # noqa: E402
from models.clip_lora import clip_lora  # noqa: E402


def load_clippp_model(
    model_name: str,
    lora_rank: int,
    checkpoint_path: Path,
    device: torch.device,
):
    if device.type != "cuda":
        raise RuntimeError("The CLIP++ pipeline requires a CUDA device")

    model, preprocess = clip_lora.load(model_name, device="cuda", lora=lora_rank)
    model = model.float()
    state_dict = torch.load(checkpoint_path, map_location=device)
    weights = state_dict.get("state_dict", state_dict)
    model.load_state_dict(weights, strict=True)
    model = model.to(device).eval()
    tokenizer = clip.tokenize
    return model, preprocess, tokenizer


def process_dataset(
    csv_path: Path,
    dataset_name: str,
    images_dir: Path,
    output_root: Path,
    model,
    preprocess,
    tokenizer,
    batch_size: int,
    num_splits: int,
    device: torch.device,
) -> None:
    df = pd.read_csv(csv_path)
    print(f"[CLIP++:{dataset_name}] samples: {len(df)}")
    dataset_output_dir = ensure_output_dir(output_root, dataset_name)

    total_img = 0
    total_mt = 0
    total_ref = 0

    for chunk_idx, chunk_df in iter_dataframe_chunks(df, num_splits):
        chunk_prefix = chunk_file_prefix(dataset_name, chunk_idx)
        img_features_dict: Dict[str, np.ndarray] = {}
        mt_features_dict: Dict[str, np.ndarray] = {}
        ref_features_dict: Dict[str, np.ndarray] = {}

        for start_row in tqdm(
            range(0, len(chunk_df), batch_size),
            desc=f"{dataset_name}-chunk{chunk_idx + 1}",
        ):
            end_row = min(start_row + batch_size, len(chunk_df))
            batch_df = chunk_df.iloc[start_row:end_row]

            valid_rows: List[dict] = []
            batch_images = []
            for _, row in batch_df.iterrows():
                imgid = str(row["imgid"])
                image_path = images_dir / imgid
                if not image_path.exists():
                    print(f"[WARN] image not found: {image_path}")
                    continue
                refs = parse_refs(row["refs"])
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as exc:
                    print(f"[WARN] failed to load image ({imgid}): {exc}")
                    continue

                batch_images.append(preprocess(image))
                valid_rows.append(
                    {
                        "imgid": imgid,
                        "global_index": int(row["__global_index__"]),
                        "mt": str(row["mt"]) if row["mt"] is not None else "",
                        "refs": refs,
                    }
                )

            if not valid_rows:
                continue

            batch_mt_texts = [
                row["mt"] for row in valid_rows
            ]
            flattened_refs = [
                ref
                for row in valid_rows
                for ref in (row["refs"] or [])
            ]

            with torch.no_grad():
                image_input = torch.stack(batch_images).to(device).float()
                img_features = model.encode_image(image_input).cpu().numpy()

                mt_tokens = tokenizer(batch_mt_texts, truncate=True).to(device)
                mt_features = model.encode_text(mt_tokens).cpu().numpy()

                ref_features = None
                if flattened_refs:
                    ref_tokens = tokenizer(flattened_refs, truncate=True).to(device)
                    ref_features = model.encode_text(ref_tokens).cpu().numpy()

            for i, row in enumerate(valid_rows):
                img_features_dict[row["imgid"]] = img_features[i]
                mt_key = row["mt"]
                mt_features_dict[mt_key] = mt_features[i]

            if flattened_refs and ref_features is not None:
                offset = 0
                for row in valid_rows:
                    refs = row["refs"] or []
                    for local_idx, ref_text in enumerate(refs):
                        ref_key = ref_text
                        ref_features_dict[ref_key] = ref_features[offset]
                        offset += 1

        output_img_path = dataset_output_dir / f"{chunk_prefix}_img_features_{chunk_idx + 1}.npz"
        output_mt_path = dataset_output_dir / f"{chunk_prefix}_mt_features_{chunk_idx + 1}.npz"
        output_ref_path = dataset_output_dir / f"{chunk_prefix}_ref_features_{chunk_idx + 1}.npz"

        np.savez_compressed(output_img_path, features=img_features_dict)
        np.savez_compressed(output_mt_path, features=mt_features_dict)
        np.savez_compressed(output_ref_path, features=ref_features_dict)

        total_img += len(img_features_dict)
        total_mt += len(mt_features_dict)
        total_ref += len(ref_features_dict)

        print(
            f"[CLIP++:{dataset_name}] chunk {chunk_idx + 1}: "
            f"img={len(img_features_dict)}, mt={len(mt_features_dict)}, ref={len(ref_features_dict)}"
        )

    print(
        f"[CLIP++:{dataset_name}] completed: "
        f"image {total_img}, MT {total_mt}, reference {total_ref}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP++ (PACScore) feature extractor for the SPICA dataset"
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root("clippp"),
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
        "--model-name", type=str, default="ViT-L/14", help="Model name for clip_lora.load"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=4, help="LoRA rank passed to clip_lora.load"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Pretrained PAC++ checkpoint file",
    )
    parser.add_argument(
        "--text-prefix",
        type=str,
        default="A photo depicts ",
        help="Prefix applied to text inputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    images_dir = args.images_dir or default_images_dir(data_root)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    csv_map = resolve_csv_paths(data_root, args.splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, preprocess, tokenizer = load_clippp_model(
        args.model_name,
        args.lora_rank,
        args.checkpoint_path,
        device,
    )

    for split_name, csv_path in csv_map.items():
        process_dataset(
            csv_path=csv_path,
            dataset_name=split_name,
            images_dir=images_dir,
            output_root=output_root,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_splits=args.num_splits,
            device=device,
        )


if __name__ == "__main__":
    main()

