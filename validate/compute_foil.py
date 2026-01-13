import json
from pearl.metrics.regression_metrics import RegressionReport
from pearl.models import load_checkpoint
from tqdm import tqdm
import json
from pearl.models import download_model, load_checkpoint, model2download, str2model
from pearl.trainer import TrainerConfig, build_trainer
import yaml
from utils import *
from dataset import *
from pascal50s import Pascal50sDataset
from PIL import Image
from pathlib import Path
from copy import deepcopy
from pearl.models.utils import load_clip_features

class FoilDatset:
    def __init__(self, coco_root_path="data_en/coco", foil_path="data_en/foil/foilv1.0_test_2017.json"):
        coco_root_path = Path(coco_root_path)
        coco_path = coco_root_path / Path("captions_val2014.json")
        coco_refs = self._read_coco(coco_path)
        self.data = self._build_foil(foil_path, coco_refs) # data[anno_id][foil or orig] = [anno1, anno2, ...]
        self.coco_root_path = coco_root_path
        self.dataset = {"one_ref" : None, "four_ref" : None}

    def _read_coco(self, coco_annos):
        refs = {}
        with open(coco_annos) as f:
            coco = json.load(f)
        for ann in coco["annotations"]:
            refs.setdefault(ann['image_id'],[]).append(ann['caption'])
        return refs
    
    def _build_foil(self, path, coco_refs):
        with open(path) as f:
            self.data = json.load(f)
        images = self.data["images"]
        annos = self.data["annotations"]

        data = {}
        imgid_to_img = {img["id"] : img for img in images}
        for anno in annos:
            anno_id = anno["id"]
            data.setdefault(anno_id, {"foil" : [], "orig" : []})
            key = "foil" if anno["foil"] else "orig"
            anno["image"] = imgid_to_img[anno["image_id"]]
            anno["refs"] = coco_refs[anno["image_id"]]
            data[anno_id][key].append(anno)
        
        return data

    def get_data(self,one_ref):
        key = "one_ref" if one_ref else "four_ref"
        if self.dataset[key] is not None:
            return self.dataset[key]
        
        clip_features_path = "features/clip_features/foil"
        clip_features = load_clip_features(clip_features_path)
        mt_clip_features = clip_features["mt_features"]
        img_clip_features = clip_features["img_features"]
        ref_clip_features = clip_features["ref_features"]

        blip2_features_path = "features/blip2_features/foil"
        blip2_features = load_clip_features(blip2_features_path)
        mt_blip2_features = blip2_features["mt_features"]
        img_blip2_features = blip2_features["img_features"]
        ref_blip2_features = blip2_features["ref_features"]

        beit3_features_path = "features/beit3_features/foil"
        beit3_features = load_clip_features(beit3_features_path)
        mt_beit3_features = beit3_features["mt_features"]
        img_beit3_features = beit3_features["img_features"]
        ref_beit3_features = beit3_features["ref_features"]

        stella_features_path = "features/stella_features/foil"
        stella_features = load_clip_features(stella_features_path, img=False)
        mt_stella_features = stella_features["mt_features"]
        img_stella_features = stella_features["img_features"]
        ref_stella_features = stella_features["ref_features"]

        dataset = []
        for _, data in (pbar := tqdm(self.data.items())):  # data[anno_id][foil or orig] = [anno1, anno2, ...]
            pbar.set_description("Prepare dataset ...")
            foiles, origs = data["foil"], data["orig"]

            assert len(origs) == 1
            N = len(foiles)
            for foil, orig in zip(foiles, [origs[0]]*N):
                refs = foil["refs"]
                refs = [r for r in refs if r != orig["caption"]]
                if one_ref:
                    refs = [refs[0]]
                
                filename = Path(foil["image"]["file_name"])
                img_path = Path("data_en/images") / filename

                dataset.append({
                    "imgid" : img_path,
                    "refs": refs,
                    "mt": foil["caption"],
                    "type": "foil",
                    "mt_features": mt_clip_features["A photo depicts " + foil["caption"]],
                    "refs_features": [ref_clip_features["A photo depicts " + r] for r in refs],
                    "img_features": img_clip_features[foil["image"]["file_name"]],
                    "blip_mt_features": mt_blip2_features[foil["caption"]],
                    "blip_refs_features": [ref_blip2_features[r] for r in refs],
                    "blip_img_features": img_blip2_features[foil["image"]["file_name"]],
                    "beit3_mt_features": mt_beit3_features[foil["caption"]],
                    "beit3_refs_features": [ref_beit3_features[r] for r in refs],
                    "beit3_img_features": img_beit3_features[foil["image"]["file_name"]],
                    "stella_mt_features": mt_stella_features[foil["caption"]],
                    "stella_refs_features": [ref_stella_features[r] for r in refs],
                })
                dataset.append({
                    "imgid" : img_path,
                    "refs": refs,
                    "mt": orig["caption"],
                    "type": "orig",
                    "mt_features": mt_clip_features["A photo depicts " + orig["caption"]],
                    "refs_features": [ref_clip_features["A photo depicts " + r] for r in refs],
                    "img_features": img_clip_features[foil["image"]["file_name"]],
                    "blip_mt_features": mt_blip2_features[orig["caption"]],
                    "blip_refs_features": [ref_blip2_features[r] for r in refs],
                    "blip_img_features": img_blip2_features[foil["image"]["file_name"]],
                    "beit3_mt_features": mt_beit3_features[orig["caption"]],
                    "beit3_refs_features": [ref_beit3_features[r] for r in refs],
                    "beit3_img_features": img_beit3_features[foil["image"]["file_name"]],
                    "stella_mt_features": mt_stella_features[orig["caption"]],
                    "stella_refs_features": [ref_stella_features[r] for r in refs],
                })
        
        self.dataset[key] = dataset
        return self.dataset[key]

def collect_acc(memory, dataset_name, method, acc):
    memory.setdefault(dataset_name, {})
    memory[dataset_name].update({method : acc})
    gprint(f"[{dataset_name}]",method,acc)

def pearl(dataset,args):
    yprint("Compute Pearl ...")
    rep = RegressionReport()
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
    for data_ in (pbar := tqdm(dataset)):
        pbar.set_description("Prepare dataset ...")
        data.append(data_)
    
    _, sys_score, img_score, ref_score = model.predict(data,cuda=True,batch_size=32)
    return sys_score, img_score, ref_score

def compute_acc(model_fn,dataset,one_ref,**kwargs):
    # Split by buckets because images do not fit in RAM.
    bucket_count = 10
    data = dataset.get_data(one_ref)
    
    print("Compute ...")
    sys_score = []
    img_score = []
    ref_score = []
    for i in range(bucket_count):
        bucket_size = len(data) // bucket_count
        subset = deepcopy(data[i*bucket_size:(i+1)*bucket_size])
        for j, sub in enumerate(pbar := tqdm(subset)):
            pbar.set_description(f"Processing {i+1}/{bucket_count}")
            subset[j].update({"img" : Image.open(sub["imgid"]).convert("RGB")})
        sub_sys_score, sub_img_score, sub_ref_score = model_fn(subset,**kwargs)
        sys_score.extend(sub_sys_score)
        img_score.extend(sub_img_score)
        ref_score.extend(sub_ref_score)
        del subset
    
    assert len(sys_score) == len(data)
    assert len(sys_score) % 2 == 0
    assert len(img_score) == len(data)
    assert len(ref_score) == len(data)

    acc = 0.
    img_acc = 0.
    ref_acc = 0.
    N = len(sys_score) // 2
    for i in range(0,2*N,2):
        s1 = sys_score[i] # foil
        s2 = sys_score[i+1] # orig
        
        img_s1 = img_score[i]
        img_s2 = img_score[i+1]
        
        ref_s1 = ref_score[i]
        ref_s2 = ref_score[i+1]
        
        # sanity check
        assert data[i]["type"] == "foil" and data[i+1]["type"] == "orig"

        if s2 > s1:
            acc += 1.
        
        if img_s2 > img_s1:
            img_acc += 1.
        
        if ref_s2 > ref_s1:
            ref_acc += 1.
        
    acc /= N
    img_acc /= N
    ref_acc /= N
    rprint(f"acc: {acc}")
    rprint(f"img_acc: {img_acc}")
    rprint(f"ref_acc: {ref_acc}")
    
    return acc, img_acc, ref_acc


def compute_foil(args, memory, tops):
    dataset = FoilDatset()
    dataset_name = "foil"
    for one_ref in [True, False]:
        suffix = "(one_ref)" if one_ref else "(four-ref)"
        dataset_name += suffix
        if args.pearl:
            pearl_acc, pearl_img_acc, pearl_ref_acc = compute_acc(pearl, dataset, one_ref, args=args)
            collect_acc(memory, dataset_name, f"Pearl{suffix}", pearl_acc)
            collect_acc(memory, dataset_name, f"Pearl_img{suffix}", pearl_img_acc)
            collect_acc(memory, dataset_name, f"Pearl_ref{suffix}", pearl_ref_acc)

    # aggregate
    max_acc = ("", 0.)
    max_img_acc = ("", 0.)
    max_ref_acc = ("", 0.)
    for method, acc in memory[dataset_name].items():
        if max_acc[1] < acc:
            max_acc = (method, acc)
        if max_img_acc[1] < acc:
            max_img_acc = (method, acc)
        if max_ref_acc[1] < acc:
            max_ref_acc = (method, acc)

    rprint("[TOP]")
    rprint(max_acc)
    rprint(max_img_acc)
    rprint(max_ref_acc)
    tops[dataset_name] = {
        "acc" : max_acc,
        "img_acc" : max_img_acc,
        "ref_acc" : max_ref_acc
    }

    return memory, tops
