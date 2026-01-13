from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchscale.architecture.config import EncoderConfig
from tqdm import tqdm
from transformers import XLMRobertaTokenizer

from pipelines.beit3 import utils as beit3_utils
from pipelines.beit3.modeling_finetune import BEiT3ForRetrieval

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


def build_model(device: torch.device, checkpoint_path: Path) -> BEiT3ForRetrieval:
    """Build the BEiT3 retrieval model and load its weights."""
    img_size = 224
    patch_size = 16
    vocab_size = 64010
    drop_path_rate = 0
    mlp_ratio = 4
    checkpoint_activations = False
    encoder_cfg = EncoderConfig(
        img_size=img_size,
        patch_size=patch_size,
        vocab_size=vocab_size,
        multiway=True,
        layernorm_embedding=False,
        normalize_output=True,
        no_output_layer=True,
        drop_path_rate=drop_path_rate,
        encoder_embed_dim=768,
        encoder_attention_heads=12,
        encoder_ffn_embed_dim=int(768 * mlp_ratio),
        encoder_layers=12,
        checkpoint_activations=checkpoint_activations,
    )

    model = BEiT3ForRetrieval(args=encoder_cfg, only_infer=True).to(device)
    beit3_utils.load_model_and_may_interpolate(
        str(checkpoint_path), model, "model|module", ""
    )
    model.eval()
    return model


def prepare_preprocess() -> transforms.Compose:
    """Define the image preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def process_dataset(
    csv_path: Path,
    dataset_name: str,
    images_dir: Path,
    output_root: Path,
    model: BEiT3ForRetrieval,
    tokenizer: XLMRobertaTokenizer,
    preprocess: transforms.Compose,
    batch_size: int,
    num_splits: int,
    device: torch.device,
) -> None:
    """Process a single split CSV and persist features."""
    df = pd.read_csv(csv_path)
    print(f"[{dataset_name}] samples: {len(df)}")
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
            for row_idx, row in batch_df.iterrows():
                imgid = str(row["imgid"])
                image_path = images_dir / imgid
                if not image_path.exists():
                    print(f"[WARN] image not found: {image_path}")
                    continue

                refs = parse_refs(row["refs"])
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as exc:  # pragma: no cover - surfacing PIL errors
                    print(f"[WARN] failed to load image ({imgid}): {exc}")
                    continue

                valid_rows.append(
                    {
                        "imgid": imgid,
                        "global_index": int(row["__global_index__"]),
                        "mt": str(row["mt"]) if row["mt"] is not None else "",
                        "refs": refs,
                    }
                )
                batch_images.append(preprocess(image))

            if not valid_rows:
                continue

            batch_mt_texts = [row["mt"] or "[EMPTY]" for row in valid_rows]
            flattened_refs = [
                ref for row in valid_rows for ref in (row["refs"] or [])
            ]

            with torch.no_grad():
                image_tensor = torch.stack(batch_images).to(device)
                img_features, _ = model(image=image_tensor)
                img_features = img_features.cpu().numpy()

                mt_tokens = tokenizer(
                    batch_mt_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                )
                mt_input_ids = mt_tokens["input_ids"].to(device)
                _, mt_features = model(text_description=mt_input_ids)
                mt_features = mt_features.cpu().numpy()

                ref_features = None
                if flattened_refs:
                    ref_tokens = tokenizer(
                        flattened_refs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77,
                    )
                    ref_input_ids = ref_tokens["input_ids"].to(device)
                    _, ref_feats = model(text_description=ref_input_ids)
                    ref_features = ref_feats.cpu().numpy()

            # Store features inside dictionaries
            for i, row in enumerate(valid_rows):
                img_features_dict[row["imgid"]] = img_features[i]
                # MT keys are the MT text itself
                mt_features_dict[row["mt"]] = mt_features[i]

            if flattened_refs and ref_features is not None:
                offset = 0
                for row in valid_rows:
                    refs = row["refs"] or []
                    for local_idx, ref_text in enumerate(refs):
                        # Ref keys are the reference text itself
                        ref_features_dict[ref_text] = ref_features[offset]
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
            f"[{dataset_name}] chunk {chunk_idx + 1}: "
            f"img={len(img_features_dict)}, "
            f"mt={len(mt_features_dict)}, "
            f"ref={len(ref_features_dict)}"
        )

    print(
        f"[{dataset_name}] completed: "
        f"image features {total_img}, MT features {total_mt}, reference features {total_ref}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BEiT3 feature extractor for the SPICA dataset"
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument(
        "--images-dir", type=Path, default=None, help="Image directory (defaults to data/images)"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root("beit3"),
        help="Output root directory for features",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names to process (choose from train/val/test)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        required=True,
        help="SentencePiece path for the XLM-R tokenizer",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Pretrained BEiT3 checkpoint file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    images_dir = args.images_dir or default_images_dir(data_root)
    output_root = args.output_root

    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")
    output_root.mkdir(parents=True, exist_ok=True)

    csv_map = resolve_csv_paths(data_root, args.splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(device, args.checkpoint_path)
    tokenizer = XLMRobertaTokenizer(str(args.tokenizer_path))
    preprocess = prepare_preprocess()

    for split_name, csv_path in csv_map.items():
        process_dataset(
            csv_path=csv_path,
            dataset_name=split_name,
            images_dir=images_dir,
            output_root=output_root,
            model=model,
            tokenizer=tokenizer,
            preprocess=preprocess,
            batch_size=args.batch_size,
            num_splits=args.num_splits,
            device=device,
        )


if __name__ == "__main__":
    main()

