import pandas as pd
import itertools
from pearl.models.model_base import SpicaDataset

def get_dataset(path, features_path, blip_features_path, beit3_features_path, stella_features_path):
    df = pd.read_csv(path)
    df = df[["mt","refs","score", "imgid"]]
    refs_list = []
    for refs in df["refs"]:
        refs = eval(refs)
        refs_list.append(refs)

    df["refs"] = refs_list
    df["mt"] = df["mt"].astype(str)
    df["score"] = df["score"].astype(float)
    df["imgid"] = df["imgid"].astype(str)
    test_dataset = df.to_dict("records")
    test_dataset = SpicaDataset(test_dataset, "data/images", features_path, blip_features_path, beit3_features_path, stella_features_path)
    return test_dataset



# def get_dataset(path, permute=False):
#     dataset_list = []
#     if permute:
#         df = pd.read_csv(path)
#         df = df[["mt","refs","score", "imgid"]]
#         min_len = 1e18
#         for refs in df["refs"]:
#             refs = eval(refs)
#             min_len = min(len(refs), min_len)
#         idxs = list(range(min_len))
#         idx_list = itertools.permutations(idxs, 2)
#     else:
#         idx_list = [(0, 1)]

#     for idx1, idx2 in idx_list:
#         df = pd.read_csv(path)
#         df = df[["mt","refs","score", "imgid"]]
#         src, ref = [], []
#         for refs in df["refs"]:
#             refs = eval(refs)
#             src.append(refs[idx1])
#             ref.append(refs[idx2])

#         df["src"] = src
#         df["ref"] = ref
#         df["mt"] = df["mt"].astype(str)
#         df["score"] = df["score"].astype(float)
#         df["imgid"] = df["imgid"].astype(str)
#         test_dataset = df.to_dict("records")
#         test_dataset = SpicaDataset(test_dataset, "data_en/images")
#         dataset_list.append(test_dataset)

#     return dataset_list
