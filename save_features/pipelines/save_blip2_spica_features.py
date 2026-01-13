from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from lavis.models import load_model_and_preprocess
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


def load_blip2_model(
    model_name: str, model_type: str, device: torch.device
):
    """Load the BLIP2 model along with preprocessors."""
    model, vis_processors, text_processors = load_model_and_preprocess(
        model_name,
        model_type,
        device=device if device.type == "cuda" else "cpu",
        is_eval=True,
    )
    if device.type == "cuda":
        model = model.to(device)
    model.eval()
    return model, vis_processors, text_processors


def process_dataset(
    csv_path: Path,
    dataset_name: str,
    images_dir: Path,
    output_root: Path,
    model,
    vis_processors,
    text_processors,
    batch_size: int,
    num_splits: int,
    device: torch.device,
) -> None:
    df = pd.read_csv(csv_path)
    print(f"[BLIP2:{dataset_name}] samples: {len(df)}")
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

                batch_images.append(vis_processors["eval"](image))
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

            batch_mt_texts = [row["mt"] or "[EMPTY]" for row in valid_rows]
            flattened_refs = [
                ref for row in valid_rows for ref in (row["refs"] or [])
            ]

            with torch.no_grad():
                image_input = torch.stack(batch_images).to(device)
                img_out = model.extract_features({"image": image_input}, mode="image")
                img_features = (
                    img_out.image_embeds_proj.detach().cpu().numpy()
                )

                mt_tokens = [text_processors["eval"](mt) for mt in batch_mt_texts]
                mt_out = model.extract_features({"text_input": mt_tokens}, mode="text")
                mt_features = (
                    mt_out.text_embeds_proj[:, 0, :].detach().cpu().numpy()
                )

                ref_features = None
                if flattened_refs:
                    ref_tokens = [
                        text_processors["eval"](ref) for ref in flattened_refs
                    ]
                    ref_out = model.extract_features(
                        {"text_input": ref_tokens}, mode="text"
                    )
                    ref_features = (
                        ref_out.text_embeds_proj[:, 0, :].detach().cpu().numpy()
                    )

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
            f"[BLIP2:{dataset_name}] chunk {chunk_idx + 1}: "
            f"img={len(img_features_dict)}, mt={len(mt_features_dict)}, ref={len(ref_features_dict)}"
        )

    print(
        f"[BLIP2:{dataset_name}] completed: "
        f"image {total_img}, MT {total_mt}, reference {total_ref}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BLIP2 feature extractor for the SPICA dataset"
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root("blip2"),
        help="Output root directory for features",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names to process",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument(
        "--model-name",
        type=str,
        default="blip2_image_text_matching",
        help="model_name argument for load_model_and_preprocess",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="pretrain",
        help="model_type argument for load_model_and_preprocess",
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

    model, vis_processors, text_processors = load_blip2_model(
        args.model_name, args.model_type, device
    )

    for split_name, csv_path in csv_map.items():
        process_dataset(
            csv_path=csv_path,
            dataset_name=split_name,
            images_dir=images_dir,
            output_root=output_root,
            model=model,
            vis_processors=vis_processors,
            text_processors=text_processors,
            batch_size=args.batch_size,
            num_splits=args.num_splits,
            device=device,
        )


if __name__ == "__main__":
    main()

