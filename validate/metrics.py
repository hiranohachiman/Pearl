import json
import shutil
import os
from pearl.metrics.regression_metrics import RegressionReport
from pearl.models import load_checkpoint
from tqdm import tqdm
import json
from pearl.models import download_model, load_checkpoint, model2download, str2model
from pearl.trainer import TrainerConfig, build_trainer
import yaml
from utils import *
from dataset import *
import pandas as pd
import numpy as np
import time

def compute_pearl_coef(args,test_dataset,dataset_name,kendall_type):
    yprint("Compute Pearl ...")
    rep = RegressionReport(kendall_type)
    if args.model:
        model = load_checkpoint(args.model)
    elif args.hparams:
        yaml_file = yaml.load(open(args.hparams).read(), Loader=yaml.FullLoader)
        train_configs = TrainerConfig(yaml_file)
        model_config = str2model[train_configs.model].ModelConfig(yaml_file)
        print(str2model[train_configs.model].ModelConfig)
        print(model_config.namespace()) 
        model = str2model[train_configs.model](model_config.namespace())
        model.eval()
        model.freeze()
        
    data = []
    gt_scores = []
    imgids = []
    mts = []
    refs = []
    for data_ in (pbar := tqdm(test_dataset)):
        pbar.set_description("Prepare dataset ...")
        data.append(data_)
        gt_scores.append(data_["score"])
        imgids.append(data_["imgid"])
        mts.append(data_["mt"])
        refs.append(data_["refs"])
    start_time = time.time()  
    _, sys_score, img_score, ref_score = model.predict(data, cuda=True, batch_size=32)
    end_time = time.time()  

    execution_time = end_time - start_time  

    coef = rep.compute(sys_score, gt_scores)
    img_coef = rep.compute(img_score, gt_scores)
    ref_coef = rep.compute(ref_score, gt_scores)
    
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
