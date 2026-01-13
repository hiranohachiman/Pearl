import sys
sys.path.append("./pacscore")
from pearl.models.utils import load_clip_features

import argparse
import torch
import pacscore.evaluation as evaluation
import scipy.stats

from pacscore.models.clip import clip
from pacscore.utils import collate_fn
from pacscore.evaluation import PACScore, RefPACScore
from pacscore.models import open_clip
from data import Flickr8k
from torch.utils.data import DataLoader
from pearl.metrics.regression_metrics import RegressionReport
from pearl.models import load_checkpoint
import argparse
from pearl.models import load_checkpoint
from PIL import Image
from utils import *
import os

def collect_coef(memory, dataset_name, method, coef_tensor):
    memory.setdefault(dataset_name, {})
    coef = {k : round(float(v.numpy() if not isinstance(v,float) else v),4) for k, v in coef_tensor.items()}
    memory[dataset_name].update({method : coef})
    gprint(f"[{dataset_name}]",method,coef)

def compute_correlation_scores(memory, dataloader, pacscore, pearl, preprocess, args):
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
    _, pac_scores, candidate_feats, len_candidates = PACScore(pacscore, preprocess, ims_cs, gen_cs, args.device, w=2.0)
    all_scores['PAC-S'] = pac_scores
    
    # RefPAC-S
    _, per_instance_text_text = RefPACScore(pacscore, gts_cs, candidate_feats, args.device, torch.tensor(len_candidates))
    refpac_scores = 2 * pac_scores * per_instance_text_text / (pac_scores + per_instance_text_text)
    all_scores['RefPAC-S'] = refpac_scores
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

    # Pearl
    data = [{
        "mt" : gen,
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
        } for image, refs, gen in zip(ims_cs, gts_cs, gen_cs)
    ]
    _, sys_score, img_score, ref_score = pearl.predict(data,cuda=True,batch_size=32)
    all_scores['Pearl'] = sys_score
    all_scores['Pearl-img'] = img_score
    all_scores['Pearl-ref'] = ref_score
    del data

    for k, v in all_scores.items():
        kendalltau_b = 100 * scipy.stats.kendalltau(v, human_scores, variant='b')[0]
        kendalltau_c = 100 * scipy.stats.kendalltau(v, human_scores, variant='c')[0]
        print('%s \t Kendall Tau-b: %.3f \t  Kendall Tau-c: %.3f'
              % (k, kendalltau_b, kendalltau_c))
        collect_coef(memory,
                    args.dataset_name,
                    k,
                    {"Kendall" : kendalltau_c if args.kendall_type == "c" else kendalltau_b}
        )
    return memory

def compute_scores(memory, pacscore, pearl, preprocess, args):
    args.datasets = ['flickr8k_expert', 'flickr8k_cf'] 

    args.batch_size_compute_score = 10
    for d in args.datasets:
        print("Computing correlation scores on dataset: " + d)
        if d == 'flickr8k_expert':
            dataset = Flickr8k(root='data_en/flickr8k/',json_file='flickr8k.json')
            dataloader = DataLoader(dataset, batch_size=args.batch_size_compute_score, shuffle=False, collate_fn=collate_fn)
            args.kendall_type = "c"
        elif d == 'flickr8k_cf':
            dataset = Flickr8k(root='data_en/flickr8k/',json_file='crowdflower_flickr8k.json')
            dataloader = DataLoader(dataset, batch_size=args.batch_size_compute_score, shuffle=False, collate_fn=collate_fn)
            args.kendall_type = "b"
        
        args.dataset_name = d
        memory = compute_correlation_scores(memory, dataloader, pacscore, pearl, preprocess, args)
    return memory


def compute_flickr(args,checkpoint,memory,tops):
    # Pearl
    pearl = load_checkpoint(checkpoint)

    # PAC-S
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pacscore, preprocess = clip.load("ViT-B/32", device=device)
    pacscore = pacscore.to(device)
    pacscore = pacscore.float()

    checkpoint = torch.load("pacscore/checkpoints/clip_ViT-B-32.pth") # Use checkpoints trained with PACScore
    pacscore.load_state_dict(checkpoint['state_dict'])
    pacscore.eval()

    args.device = device
    memory = compute_scores(memory, pacscore, pearl, preprocess, args)
    return memory, tops
