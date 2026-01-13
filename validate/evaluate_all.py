#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-contained evaluation script
Evaluates Composite, Flickr, and Nebula datasets
"""

import json
import argparse
import sys
import os
import warnings
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from scipy.stats import kendalltau, pearsonr, spearmanr
import scipy.stats

# Imports for PACScore
sys.path.append("./pacscore")
from pacscore.models.clip import clip
from pacscore.utils import collate_fn
from pacscore.evaluation import PACScore, RefPACScore
import pacscore.evaluation as evaluation

from pearl.models import load_checkpoint
from pearl.models.utils import load_clip_features

# ============================================================================
# Color output utilities
# ============================================================================
from termcolor import colored

def rprint(*args):
    combined_text = ' '.join(map(str, args))
    print(colored(combined_text, 'white', 'on_red', attrs=["bold"]))

def yprint(*args):
    combined_text = ' '.join(map(str, args))
    print(colored(combined_text, 'yellow', attrs=["bold"]))

def gprint(*args):
    combined_text = ' '.join(map(str, args))
    print(colored(combined_text, 'green', attrs=["bold"]))


# ============================================================================
# Metrics computation classes (Kendall, Pearson, Spearman)
# ============================================================================
class Kendall:
    def __init__(self, variant="c"):
        self.name = f"kendall_{variant}"
        self.variant = variant

    def compute(self, x: np.array, y: np.array) -> float:
        """Compute Kendall correlation

        Args:
            x: Predicted scores
            y: Ground truth scores

        Returns:
            Kendall Tau correlation value
        """
        if np.isnan(x).any() or np.isnan(y).any():
            return np.nan
        return torch.tensor(kendalltau(x, y, variant=self.variant)[0], dtype=torch.float32)


class Pearson:
    def __init__(self):
        self.name = "pearson"

    def compute(self, x: np.array, y: np.array) -> torch.Tensor:
        """Compute Pearson correlation

        Args:
            x: Predicted scores
            y: Ground truth scores

        Returns:
            Pearson correlation value
        """
        if np.isnan(x).any() or np.isnan(y).any():
            return np.nan
        return torch.tensor(pearsonr(x, y)[0], dtype=torch.float32)


class Spearman:
    def __init__(self):
        self.name = "spearman"

    def compute(self, x: np.array, y: np.array) -> float:
        """Compute Spearman correlation

        Args:
            x: Predicted scores
            y: Ground truth scores

        Returns:
            Spearman correlation value
        """
        if np.isnan(x).any() or np.isnan(y).any():
            return np.nan
        return torch.tensor(spearmanr(x, y)[0], dtype=torch.float32)


class RegressionReport:
    def __init__(self, kendall_type="c"):
        super().__init__()
        self.metrics = [Pearson(), Kendall("b"), Kendall("c"), Spearman()]

    def compute(self, x: np.array, y: np.array) -> dict:
        """Compute various correlation coefficients

        Args:
            x: Predicted scores
            y: Ground truth scores

        Returns:
            Dictionary containing correlation values for each metric
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {metric.name: metric.compute(x, y) for metric in self.metrics}


# ============================================================================
# Dataset classes
# ============================================================================
def is_image_ok(img_path):
    """Check if image file is valid"""
    try:
        img = Image.open(img_path)
        img.verify()
        return True
    except (IOError, SyntaxError) as e:
        return False


class SpicaDataset(Dataset):
    """CVPR-format dataset"""
    
    def __init__(self, dataset, img_dir_path, features_path, blip_features_path, 
                 beit3_features_path, stella_features_path):
        self.dataset = dataset
        self.img_dir_path = img_dir_path
        self.features_path = features_path
        self.blip_features_path = blip_features_path
        self.beit3_features_path = beit3_features_path
        self.stella_features_path = stella_features_path

        # Filter out corrupted image files
        for data in self.dataset:
            assert is_image_ok(os.path.join(img_dir_path, f"{data['imgid']}"))

        # Load CLIP++ features
        if self.features_path:
            features = load_clip_features(features_path)
            self.img_features = features["img_features"]
            self.mt_features = features["mt_features"]
            self.ref_features = features["ref_features"]
    
        # Load BLIP2 features
        if self.blip_features_path:
            blip_features = load_clip_features(blip_features_path)
            self.blip_img_features = blip_features["img_features"]
            self.blip_mt_features = blip_features["mt_features"]
            self.blip_ref_features = blip_features["ref_features"]
        
        # Load BEiT3 features
        if self.beit3_features_path:
            beit3_features = load_clip_features(beit3_features_path)
            self.beit3_img_features = beit3_features["img_features"]
            self.beit3_mt_features = beit3_features["mt_features"]
            self.beit3_ref_features = beit3_features["ref_features"]
        
        # Load Stella features
        if self.stella_features_path:
            stella_features = load_clip_features(stella_features_path, img=False)
            self.stella_mt_features = stella_features["mt_features"]
            self.stella_ref_features = stella_features["ref_features"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        imgid = data["imgid"]
        img_name = os.path.join(self.img_dir_path, f"{imgid}")

        labels = deepcopy(data)

        img = Image.open(img_name).convert("RGB")
        labels["img"] = img

        # Text for CLIP++ (with prefix)
        mt_text = "A photo depicts " + data["mt"]
        refs = data["refs"] if len(data["refs"]) > 0 else ["NO REFS!!!"]
        ref_texts = ["A photo depicts " + ref for ref in refs]

        if self.features_path:
            labels["img_features"] = self.img_features[imgid]
            labels["mt_features"] = self.mt_features[mt_text]
            refs_features = []
            for ref_text in ref_texts:
                if ref_text in self.ref_features:
                    refs_features.append(self.ref_features[ref_text])
                else:
                    refs_features.append(None)
            labels["refs_features"] = refs_features

        # Text for BLIP2, BEiT3, Stella (without prefix)
        mt_text_raw = data["mt"]
        refs_raw = data["refs"] if len(data["refs"]) > 0 else ["NO REFS!!!"]

        if self.blip_features_path:
            labels["blip_img_features"] = self.blip_img_features[imgid]
            labels["blip_mt_features"] = self.blip_mt_features[mt_text_raw]
            blip_refs_features = []
            for ref_text in refs_raw:
                if ref_text in self.blip_ref_features:
                    blip_refs_features.append(self.blip_ref_features[ref_text])
                else:
                    blip_refs_features.append(None)
            labels["blip_refs_features"] = blip_refs_features

        if self.beit3_features_path:
            labels["beit3_img_features"] = self.beit3_img_features[imgid]
            labels["beit3_mt_features"] = self.beit3_mt_features[mt_text_raw]
            beit3_refs_features = []
            for ref_text in refs_raw:
                if ref_text in self.beit3_ref_features:
                    beit3_refs_features.append(self.beit3_ref_features[ref_text])
                else:
                    beit3_refs_features.append(None)
            labels["beit3_refs_features"] = beit3_refs_features

        if self.stella_features_path:
            labels["stella_mt_features"] = self.stella_mt_features[mt_text_raw]
            stella_refs_features = []
            for ref_text in refs_raw:
                if ref_text in self.stella_ref_features:
                    stella_refs_features.append(self.stella_ref_features[ref_text])
                else:
                    stella_refs_features.append(None)
            labels["stella_refs_features"] = stella_refs_features

        return labels


def get_dataset(path, features_path, blip_features_path, beit3_features_path, stella_features_path):
    """Create dataset from CSV file"""
    df = pd.read_csv(path)
    df = df[["mt", "refs", "score", "imgid"]]
    
    refs_list = []
    for refs in df["refs"]:
        refs = eval(refs)
        refs_list.append(refs)

    df["refs"] = refs_list
    df["mt"] = df["mt"].astype(str)
    df["score"] = df["score"].astype(float)
    df["imgid"] = df["imgid"].astype(str)
    
    test_dataset = df.to_dict("records")
    test_dataset = SpicaDataset(
        test_dataset, "data_en/images", 
        features_path, blip_features_path, 
        beit3_features_path, stella_features_path
    )
    return test_dataset


class Flickr8k(Dataset):
    """Flickr8k dataset"""
    
    def __init__(self, json_file, root='data_en/flickr8k/', transform=None, load_images=False):
        self.im_folder = os.path.join(root, 'images')
        self.transform = transform
        self.load_images = load_images

        with open(os.path.join(root, json_file)) as fp:
            data = json.load(fp)

        self.data = list()
        for i in data:
            for human_judgement in data[i]['human_judgement']:
                if np.isnan(human_judgement['rating']):
                    print('NaN')
                    continue
                d = {
                    'image': data[i]['image_path'].split('/')[-1],
                    'references': [' '.join(gt.split()) for gt in data[i]['ground_truth']],
                    'candidate': ' '.join(human_judgement['caption'].split()),
                    'human_score': human_judgement['rating']
                }
                self.data.append(d)

    def get_image(self, filename):
        img = Image.open(os.path.join(self.im_folder, filename)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im_idx = self.data[idx]['image']
        candidate = self.data[idx]['candidate']
        references = self.data[idx]['references']
        score = self.data[idx]['human_score']

        if self.load_images:
            im = self.get_image(im_idx)
        else:
            im = os.path.join(self.im_folder, im_idx)

        return im, candidate, references, score


# ============================================================================
# Evaluation functions
# ============================================================================
def collect_coef(memory, dataset_name, method, coef_tensor):
    """Record correlation coefficients"""
    memory.setdefault(dataset_name, {})
    coef = {k: round(float(v.numpy() if not isinstance(v, float) else v), 4) 
            for k, v in coef_tensor.items()}
    memory[dataset_name].update({method: coef})
    gprint(f"[{dataset_name}]", method, coef)


def compute_pearl_coef(args, test_dataset, dataset_name, kendall_type):
    """Compute correlation coefficients with Pearl model
    
    This is the central function that computes Kendall Tau.
    It uses scipy.stats.kendalltau within RegressionReport to compute correlations.
    """
    yprint("Compute Pearl ...")
    rep = RegressionReport(kendall_type)
    
    # Load model
    if args.model:
        model = load_checkpoint(args.model)
    else:
        raise ValueError("Please specify model path (--model)")
        
    data = []
    gt_scores = []
    imgids = []
    mts = []
    refs = []
    
    # Prepare data
    for data_ in (pbar := tqdm(test_dataset)):
        pbar.set_description("Prepare dataset ...")
        data.append(data_)
        gt_scores.append(data_["score"])
        imgids.append(data_["imgid"])
        mts.append(data_["mt"])
        refs.append(data_["refs"])
    
    # Model prediction
    start_time = time.time()
    _, sys_score, img_score, ref_score = model.predict(data, cuda=True, batch_size=32)
    end_time = time.time()
    
    execution_time = end_time - start_time

    # Compute Kendall correlation (this is where Kendall Tau is actually computed)
    coef = rep.compute(sys_score, gt_scores)
    img_coef = rep.compute(img_score, gt_scores)
    ref_coef = rep.compute(ref_score, gt_scores)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'image_id': imgids,
        'mt': mts,
        'system_score': sys_score,
        'img_score': img_score,
        'ref_score': ref_score,
        'ground_truth': gt_scores,
        'ref': refs
    })
        
    return coef, img_coef, ref_coef


def compute_correlation_scores_flickr(memory, dataloader, pacscore, pearl, preprocess, args):
    """Compute correlation scores for Flickr8k dataset
    
    This also computes Kendall Tau (using scipy.stats.kendalltau directly)
    """
    gen = {}
    gts = {}

    human_scores = list()
    ims_cs = list()
    gen_cs = list()
    gts_cs = list()
    all_scores = dict()
    
    pacscore.eval()
    pearl.eval()

    for it, (images, candidates, references, scores) in enumerate(iter(dataloader)):
        for i, (im_i, gts_i, gen_i, score_i) in enumerate(zip(images, references, candidates, scores)):
            gen['%d_%d' % (it, i)] = [gen_i, ]
            gts['%d_%d' % (it, i)] = gts_i

            ims_cs.append(im_i)
            gen_cs.append(gen_i)
            gts_cs.append(gts_i)
            human_scores.append(score_i)

    # Compute traditional metrics
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    all_scores_metrics = evaluation.get_all_metrics(gts, gen, return_per_cap=True)
    
    for k, v in all_scores_metrics.items():
        if k == 'BLEU':
            all_scores['BLEU-1'] = v[0]
            all_scores['BLEU-4'] = v[-1]
        else:
            all_scores[k] = v
                
    # PAC-S
    _, pac_scores, candidate_feats, len_candidates = PACScore(
        pacscore, preprocess, ims_cs, gen_cs, args.device, w=2.0
    )
    all_scores['PAC-S'] = pac_scores
    
    # RefPAC-S
    _, per_instance_text_text = RefPACScore(
        pacscore, gts_cs, candidate_feats, args.device, torch.tensor(len_candidates)
    )
    refpac_scores = 2 * pac_scores * per_instance_text_text / (pac_scores + per_instance_text_text)
    all_scores['RefPAC-S'] = refpac_scores
    
    # Set feature paths
    if args.dataset_name == "flickr8k_expert":
        clip_features_path = "features/clip++_b_features/flickr8k_ex"
        blip2_features_path = "features/blip2_features/flickr8k_ex"
        beit3_features_path = "features/beit3_features/flickr8k_ex"
        stella_features_path = "features/stella_features/flickr8k_ex"
    elif args.dataset_name == "flickr8k_cf":
        clip_features_path = "features/clip++_b_features/flickr8k_cf"
        blip2_features_path = "features/blip2_features/flickr8k_cf"
        beit3_features_path = "features/beit3_features/flickr8k_cf"
        stella_features_path = "features/stella_features/flickr8k_cf"
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")
    
    # Load features
    clip_features = load_clip_features(clip_features_path)
    mt_clip_features = clip_features["mt_features"]
    ref_clip_features = clip_features["ref_features"]
    img_clip_features = clip_features["img_features"]

    blip2_features = load_clip_features(blip2_features_path)
    mt_blip2_features = blip2_features["mt_features"]
    ref_blip2_features = blip2_features["ref_features"]
    img_blip2_features = blip2_features["img_features"]

    beit3_features = load_clip_features(beit3_features_path)
    mt_beit3_features = beit3_features["mt_features"]
    ref_beit3_features = beit3_features["ref_features"]
    img_beit3_features = beit3_features["img_features"]

    stella_features = load_clip_features(stella_features_path, img=False)
    mt_stella_features = stella_features["mt_features"]
    ref_stella_features = stella_features["ref_features"]

    # Pearl prediction
    data = [{
        "mt": gen,
        "refs": refs,
        "img": Image.open(image).convert("RGB"),
        "mt_features": mt_clip_features["A photo depicts " + gen],
        "refs_features": [ref_clip_features["A photo depicts " + ref] for ref in refs],
        "img_features": img_clip_features[os.path.basename(image)],
        "blip_mt_features": mt_blip2_features[gen],
        "blip_refs_features": [ref_blip2_features[ref] for ref in refs],
        "blip_img_features": img_blip2_features[os.path.basename(image)],
        "beit3_mt_features": mt_beit3_features[gen],
        "beit3_refs_features": [ref_beit3_features[ref] for ref in refs],
        "beit3_img_features": img_beit3_features[os.path.basename(image)],
        "stella_mt_features": mt_stella_features[gen],
        "stella_refs_features": [ref_stella_features[ref] for ref in refs],
    } for image, refs, gen in zip(ims_cs, gts_cs, gen_cs)]
    
    _, sys_score, img_score, ref_score = pearl.predict(data, cuda=True, batch_size=32)
    all_scores['Pearl'] = sys_score
    all_scores['Pearl-img'] = img_score
    all_scores['Pearl-ref'] = ref_score
    del data

    # Compute Kendall Tau (using scipy.stats.kendalltau directly)
    for k, v in all_scores.items():
        kendalltau_b = 100 * scipy.stats.kendalltau(v, human_scores, variant='b')[0]
        kendalltau_c = 100 * scipy.stats.kendalltau(v, human_scores, variant='c')[0]
        print('%s \t Kendall Tau-b: %.3f \t  Kendall Tau-c: %.3f'
              % (k, kendalltau_b, kendalltau_c))
        collect_coef(
            memory,
            args.dataset_name,
            k,
            {"Kendall": kendalltau_c if args.kendall_type == "c" else kendalltau_b}
        )
    
    return memory


def compute_flickr(args, checkpoint, memory, tops):
    """Evaluate Flickr dataset"""
    # Pearl
    pearl = load_checkpoint(checkpoint)

    # PAC-S
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pacscore, preprocess = clip.load("ViT-B/32", device=device)
    pacscore = pacscore.to(device)
    pacscore = pacscore.float()

    checkpoint_path = "pacscore/checkpoints/clip_ViT-B-32.pth"
    checkpoint_data = torch.load(checkpoint_path)
    pacscore.load_state_dict(checkpoint_data['state_dict'])
    pacscore.eval()

    args.device = device
    args.datasets = ['flickr8k_expert', 'flickr8k_cf']
    args.batch_size_compute_score = 10
    
    for d in args.datasets:
        print("Computing correlation scores on dataset: " + d)
        if d == 'flickr8k_expert':
            dataset = Flickr8k(root='data_en/flickr8k/', json_file='flickr8k.json')
            dataloader = DataLoader(
                dataset, batch_size=args.batch_size_compute_score, 
                shuffle=False, collate_fn=collate_fn
            )
            args.kendall_type = "c"
        elif d == 'flickr8k_cf':
            dataset = Flickr8k(root='data_en/flickr8k/', json_file='crowdflower_flickr8k.json')
            dataloader = DataLoader(
                dataset, batch_size=args.batch_size_compute_score, 
                shuffle=False, collate_fn=collate_fn
            )
            args.kendall_type = "b"
        
        args.dataset_name = d
        memory = compute_correlation_scores_flickr(memory, dataloader, pacscore, pearl, preprocess, args)
    
    return memory, tops


def compute_composite(args, memory, tops):
    """Evaluate Composite dataset"""
    dataset_name = "composite"
    path = f"data_en/composite/en_test_{dataset_name}_da2.csv"
    features_path = f"features/clip++_features/composite"
    blip_features_path = f"features/blip2_features/composite"
    beit3_features_path = f"features/beit3_features/composite"
    stella_features_path = f"features/stella_features/composite"
    
    yprint(f"Processing {dataset_name} ... (path: {path})")
    test_dataset = get_dataset(
        path, features_path, blip_features_path, 
        beit3_features_path, stella_features_path
    )

    if args.pearl:
        pearl_coef, pearl_img_coef, pearl_ref_coef = compute_pearl_coef(
            args, test_dataset, dataset_name, kendall_type='c'
        )
        collect_coef(memory, dataset_name, "Pearl", pearl_coef)
        collect_coef(memory, dataset_name, "Pearl_img", pearl_img_coef)
        collect_coef(memory, dataset_name, "Pearl_ref", pearl_ref_coef)

    return memory, tops


def compute_nebula(args, memory, tops):
    """Evaluate Nebula dataset"""
    nebula_datasets = ["nebula_test", "nebula_val"]
    
    for dataset_name in nebula_datasets:
        path = f"data_en/nebula/{dataset_name}.csv"
        features_path = f"features/clip++_features/nebula"
        blip_features_path = f"features/blip2_features/nebula"
        beit3_features_path = f"features/beit3_features/nebula"
        stella_features_path = f"features/stella_features/nebula"
        
        yprint(f"Processing {dataset_name} ... (path: {path})")
        test_dataset = get_dataset(
            path, features_path, blip_features_path, 
            beit3_features_path, stella_features_path
        )

        if args.pearl:
            pearl_coef, pearl_img_coef, pearl_ref_coef = compute_pearl_coef(
                args, test_dataset, dataset_name, kendall_type='c'
            )
            collect_coef(memory, dataset_name, "Pearl", pearl_coef)
            collect_coef(memory, dataset_name, "Pearl_img", pearl_img_coef)
            collect_coef(memory, dataset_name, "Pearl_ref", pearl_ref_coef)

    return memory, tops


# ============================================================================
# Main function
# ============================================================================
def main(args):
    memory, tops = {}, {}
    
    # Composite evaluation
    if args.composite:
        yprint("\n========== Evaluating Composite Dataset ==========")
        memory, tops = compute_composite(args, memory, tops)
    
    # Flickr evaluation
    if args.flickr:
        yprint("\n========== Evaluating Flickr Dataset ==========")
        memory, tops = compute_flickr(args, args.model, memory, tops)
    
    # Nebula evaluation
    if args.nebula:
        yprint("\n========== Evaluating Nebula Dataset ==========")
        memory, tops = compute_nebula(args, memory, tops)

    # Save results
    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(memory, f, indent=4)
    
    yprint(f"\n[RESULTS] Saved to {output_file}")
    gprint(json.dumps(memory, indent=4))

    rprint("\n[TOP]")
    for dataset_name, values in tops.items():
        rprint(f"> {dataset_name}")
        if isinstance(values, dict):  # coef
            for kind, coef in values.items():
                rprint(f"{kind}: {coef[0]} ({coef[1]})")
        else:  # acc
            method, acc = values
            rprint(f"{method} ({acc})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-contained script to evaluate models on Composite, Flickr, and Nebula datasets"
    )
    
    # Model configuration
    parser.add_argument('--model', default=None, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--pearl', action='store_true', 
                        help='Use Pearl (Pearl) model')

    # Dataset selection
    parser.add_argument('--composite', action='store_true', 
                        help='Evaluate on Composite dataset')
    parser.add_argument('--flickr', action='store_true', 
                        help='Evaluate on Flickr dataset')
    parser.add_argument('--nebula', action='store_true', 
                        help='Evaluate on Nebula dataset')
    parser.add_argument('--all', action='store_true', 
                        help='Evaluate on all datasets')
    
    args = parser.parse_args()
    
    # If --all is specified, enable all datasets
    if args.all:
        args.composite = True
        args.flickr = True
        args.nebula = True
    
    # Check if at least one dataset is selected
    if not (args.composite or args.flickr or args.nebula):
        parser.error("Please specify at least one dataset (--composite, --flickr, --nebula, or --all)")
    
    main(args)
