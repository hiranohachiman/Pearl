# -*- coding: utf-8 -*-
import random
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import torch

from pearl.models.estimators.estimator_base import Estimator
from pearl.modules.feedforward import FeedForward
from pearl.modules.scalar_mix import ScalarMixWithDropout
from torchnlp.utils import collate_tensors
import torch
import math
from pearl.models.hadamard.hadamard import HadamardNet

from typing import List, Union

try:
    import warnings
    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class IndependentFeatureMLP(torch.nn.Module):
    def __init__(self, feature_size, random_init=False, operation="diff"):
        super().__init__()
        if operation == "diff":
            if random_init:
                self.weight = torch.nn.Parameter(torch.randn(feature_size, 2))
            else:
                self.weight = torch.nn.Parameter(
                    torch.tensor([[1.0, -1.0]] * feature_size)
                )
        elif operation == "sum":
            if random_init:
                self.weight = torch.nn.Parameter(torch.randn(feature_size, 2))
            else:
                self.weight = torch.nn.Parameter(
                    torch.tensor([[1.0, 1.0]] * feature_size)
                )

        else:
            raise ValueError(f"Unknown operation: {operation}")

        self.operation = operation
        if random_init:
            self.bias = torch.nn.Parameter(torch.randn(feature_size))
        else:
            self.bias = torch.nn.Parameter(torch.zeros(feature_size))

    def forward(self, tensor1, tensor2):
        if self.operation == "hadamard":
            combined_input = torch.stack([tensor1, tensor1 * tensor2], dim=-1)
            combined_tensor = torch.einsum("bfi,fij->bf", combined_input, self.weight)
            return combined_tensor + self.bias
        else:
            combined_input = torch.stack((tensor1, tensor2), dim=-1)
            combined_tensor = (
                torch.einsum("bfi,fi->bf", combined_input, self.weight) + self.bias
            )
            return combined_tensor


class PearlEstimator(Estimator):
    """
    Estimator class that uses a pretrained encoder to extract features from
    the sequences and then passes those features to a feed forward estimator.

    :param hparams: Namespace containing the hyperparameters.
    """

    class ModelConfig(Estimator.ModelConfig):
        switch_prob: float = 0.0

    def __init__(
        self,
        hparams: Namespace,
    ) -> None:
        super().__init__(hparams)

    def _build_model(self) -> Estimator:
        """
        Initializes the estimator architecture.
        """
        super()._build_model()

        self.emb_dim = 768

        input_emb_sz = self.emb_dim * 2 * 8

        self.ff = torch.nn.Sequential(
            *[
                FeedForward(
                    in_dim=input_emb_sz,
                    # out_dim=input_emb_sz,
                    hidden_sizes=self.hparams.hidden_sizes,
                    activations=self.hparams.activations,
                    dropout=self.hparams.dropout,
                    final_activation=(
                        self.hparams.final_activation
                        if hasattr(
                            self.hparams, "final_activation"
                        )  # compatability with older checkpoints!
                        else "Sigmoid"
                    ),
                ),
                torch.nn.Sigmoid(),
            ]
        )
        input_emb_sz = self.emb_dim * 2 * 8
        self.ff2 = torch.nn.Sequential(
            *[
                FeedForward(
                    in_dim=input_emb_sz,
                    # out_dim=input_emb_sz,
                    hidden_sizes=self.hparams.hidden_sizes,
                    activations=self.hparams.activations,
                    dropout=self.hparams.dropout,
                    final_activation=(
                        self.hparams.final_activation
                        if hasattr(
                            self.hparams, "final_activation"
                        )  # compatability with older checkpoints!
                        else "Sigmoid"
                    ),
                ),
                torch.nn.Sigmoid(),
            ]
        )
        device = torch.device('cuda')

        emb_dim = self.emb_dim
        blip_dim = 256
        trm_layer1 = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim * 2, nhead=8, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(trm_layer1, num_layers=2)
        trm_layer2 = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim * 2, nhead=8, batch_first=True
        )
        self.transformer2 = torch.nn.TransformerEncoder(trm_layer2, num_layers=1)
        trm_layer3 = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim * 2, nhead=8, batch_first=True
        )
        self.transformer3 = torch.nn.TransformerEncoder(trm_layer3, num_layers=1)
        trm_layer4 = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim * 2, nhead=8, batch_first=True
        )
        self.transformer4 = torch.nn.TransformerEncoder(trm_layer4, num_layers=1)

        trm_layer5 = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim * 2, nhead=8, batch_first=True
        )
        self.transformer5 = torch.nn.TransformerEncoder(trm_layer5, num_layers=1)
        trm_layer6 = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim * 2, nhead=8, batch_first=True
        )
        self.transformer6 = torch.nn.TransformerEncoder(trm_layer6, num_layers=1)

        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim * 2, dtype=torch.float32),
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim, dtype=torch.float32),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim * 2, dtype=torch.float32),
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim, dtype=torch.float32),
        )
        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim * 2, dtype=torch.float32),
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim, dtype=torch.float32),
        )

        self.mlp4 = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim * 2, dtype=torch.float32),
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim, dtype=torch.float32),
        )

        self.mlp5 = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim * 2, dtype=torch.float32),
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim, dtype=torch.float32),
        )
        self.mlp6 = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim * 2, dtype=torch.float32),
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim, dtype=torch.float32),
        )

        self.upsampler1 = torch.nn.Sequential(
            torch.nn.Linear(blip_dim, blip_dim * 2, dtype=torch.float32),
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(blip_dim * 2, emb_dim, dtype=torch.float32),
        )
        self.upsampler2 = torch.nn.Sequential(
            torch.nn.Linear(blip_dim, blip_dim * 2, dtype=torch.float32),
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(blip_dim * 2, emb_dim, dtype=torch.float32),
        )

        self.upsampler3 = torch.nn.Sequential(
            torch.nn.Linear(blip_dim, blip_dim * 2, dtype=torch.float32),
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(blip_dim * 2, emb_dim, dtype=torch.float32),
        )

        self.diff_feature_mlp1 = IndependentFeatureMLP(
            feature_size=emb_dim, operation="diff", random_init=False
        )
        self.diff_feature_mlp2 = IndependentFeatureMLP(
            feature_size=emb_dim, operation="diff", random_init=False
        )
        self.diff_feature_mlp3 = IndependentFeatureMLP(
            feature_size=emb_dim, operation="diff", random_init=False
        )
        self.diff_feature_mlp4 = IndependentFeatureMLP(
            feature_size=emb_dim, operation="diff", random_init=False
        )
        self.diff_feature_mlp5 = IndependentFeatureMLP(
            feature_size=emb_dim, operation="diff", random_init=False
        )
        self.diff_feature_mlp6 = IndependentFeatureMLP(
            feature_size=emb_dim, operation="diff", random_init=False
        )

        self.pos_encoder = PositionalEncoding(
            d_model=emb_dim * 2, dropout=0.1, max_len=1000
        )
        self.pos_encoder2 = PositionalEncoding(
            d_model=emb_dim * 2, dropout=0.1, max_len=1000
        )
        self.pos_encoder3 = PositionalEncoding(
            d_model=emb_dim * 2, dropout=0.1, max_len=1000
        )
        self.pos_encoder4 = PositionalEncoding(
            d_model=emb_dim * 2, dropout=0.1, max_len=1000
        )
        self.pos_encoder5 = PositionalEncoding(
            d_model=emb_dim * 2, dropout=0.1, max_len=1000
        )
        self.pos_encoder6 = PositionalEncoding(
            d_model=emb_dim * 2, dropout=0.1, max_len=1000
        )


        self.cls_token1 = torch.nn.Parameter(torch.zeros(1, 1, emb_dim * 2))
        torch.nn.init.normal_(self.cls_token1, std=1e-6)
        self.cls_token2 = torch.nn.Parameter(torch.zeros(1, 1, emb_dim * 2))
        torch.nn.init.normal_(self.cls_token2, std=1e-6)
        self.cls_token3 = torch.nn.Parameter(torch.zeros(1, 1, emb_dim * 2))
        torch.nn.init.normal_(self.cls_token3, std=1e-6)
        self.cls_token4 = torch.nn.Parameter(torch.zeros(1, 1, emb_dim * 2))
        torch.nn.init.normal_(self.cls_token4, std=1e-6)
        self.cls_token5 = torch.nn.Parameter(torch.zeros(1, 1, emb_dim * 2))
        torch.nn.init.normal_(self.cls_token5, std=1e-6)
        self.cls_token6 = torch.nn.Parameter(torch.zeros(1, 1, emb_dim * 2))
        torch.nn.init.normal_(self.cls_token6, std=1e-6)


        self.learnable_query1 = torch.nn.Parameter(torch.zeros(1, 8, emb_dim * 2))
        torch.nn.init.xavier_uniform_(self.learnable_query1)

        self.learnable_query2 = torch.nn.Parameter(torch.zeros(1, 8, emb_dim * 2))
        torch.nn.init.xavier_uniform_(self.learnable_query2)

        self.learnable_query3 = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        torch.nn.init.xavier_uniform_(self.learnable_query3)

        self.hadamard_net1 = HadamardNet()
        self.hadamard_net1.load_state_dict(torch.load("experiments/hadamard.pth"))
        self.hadamard_net2 = HadamardNet()
        self.hadamard_net2.load_state_dict(torch.load("experiments/hadamard.pth"))
        self.hadamard_net3 = HadamardNet()
        self.hadamard_net3.load_state_dict(torch.load("experiments/hadamard.pth"))
        self.hadamard_net4 = HadamardNet()
        self.hadamard_net4.load_state_dict(torch.load("experiments/hadamard.pth"))
        self.hadamard_net5 = HadamardNet()
        self.hadamard_net5.load_state_dict(torch.load("experiments/hadamard.pth"))
        self.hadamard_net6 = HadamardNet()
        self.hadamard_net6.load_state_dict(torch.load("experiments/hadamard.pth"))


        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=emb_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.cross_attention2 = torch.nn.MultiheadAttention(
            embed_dim=emb_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.query_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2, emb_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 4, emb_dim * 2)
        )

        self.ref_img_ratio = torch.nn.Parameter(torch.tensor(0.5)) 
    
    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Sets different Learning rates for different parameter groups."""
        optimizer = self._build_optimizer(self.parameters())
        scheduler = self._build_scheduler(optimizer)
        return [optimizer], [scheduler]

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        import numpy as np
        temp_refs_features = []
        temp_blip_refs_features = []
        temp_beit3_refs_features = []
        temp_stella_refs_features = []
        for s in sample:
            temp_refs_features.append(s["refs_features"])
            temp_blip_refs_features.append(s["blip_refs_features"])
            temp_beit3_refs_features.append(s["beit3_refs_features"])
            temp_stella_refs_features.append(s["stella_refs_features"])
        sample = collate_tensors(sample)
        sample["refs_features"] = temp_refs_features
        sample["blip_refs_features"] = temp_blip_refs_features
        sample["beit3_refs_features"] = temp_beit3_refs_features
        sample["stella_refs_features"] = temp_stella_refs_features

        inputs = {"img": sample["img"]}
        refs_features = []
        
        for refs_feature in sample["refs_features"]:
            features = []
            for ref_feature in refs_feature:
                if ref_feature is not None:
                    features.append(torch.tensor(np.array(ref_feature)))
                else:
                    features.append(None)
            refs_features.append(features)
        imgs_features = torch.tensor(np.array(sample["img_features"]))
        mt_features = torch.tensor(np.array(sample["mt_features"]))

        inputs.update({
            "imgs_features": imgs_features,
            "mt_features": mt_features,
            "refs_features": refs_features,
        })
        
        blip_img_features = torch.from_numpy(
            np.array(sample["blip_img_features"])
        )
        

        blip_mt_features = torch.from_numpy(
            np.array(sample["blip_mt_features"])
        )
        
        blip_refs_features = []
        for ref in sample["blip_refs_features"]:
            features = []
            for ref_feature in ref:
                if ref_feature is not None:
                    features.append(torch.from_numpy(np.array(ref_feature)))
                else:
                    features.append(None)
            blip_refs_features.append(features)

        inputs.update({
            "blip_img_features": blip_img_features,
            "blip_mt_features": blip_mt_features,
            "blip_refs_features": blip_refs_features,
        })

        beit3_img_features = torch.from_numpy(
            np.array(sample["beit3_img_features"])
        )

        beit3_mt_features = torch.from_numpy(
            np.array(sample["beit3_mt_features"])
        )

        beit3_refs_features = []
        for ref in sample["beit3_refs_features"]:
            features = []
            for ref_feature in ref:
                if ref_feature is not None:
                    features.append(torch.from_numpy(np.array(ref_feature)))
                else:
                    features.append(None)
            beit3_refs_features.append(features)

        inputs.update({
            "beit3_img_features": beit3_img_features,
            "beit3_mt_features": beit3_mt_features,
            "beit3_refs_features": beit3_refs_features,
        })

        stella_mt_features = torch.from_numpy(
            np.array(sample["stella_mt_features"])
        )

        stella_refs_features = []
        for ref in sample["stella_refs_features"]:
            features = []
            for ref_feature in ref:
                if ref_feature is not None:
                    features.append(torch.from_numpy(np.array(ref_feature)))
                else:
                    features.append(None)
            stella_refs_features.append(features)

        inputs.update({
            "stella_mt_features": stella_mt_features,
            "stella_refs_features": stella_refs_features,
        })

        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets

    def masked_global_average_pooling(self, input_tensor, mask):
        mask = mask.logical_not()  # mask[x] = input[x] is not pad
        mask_expanded = mask.unsqueeze(-1).expand_as(input_tensor).float()
        input_tensor_masked = input_tensor * mask_expanded
        num_elements = mask.sum(dim=1, keepdim=True).float() 
        output_tensor = input_tensor_masked.sum(dim=1) / num_elements
        return output_tensor

    def forward(
        self,
        refs_features: list = None,
        mt_features: torch.tensor = None,
        imgs_features: torch.tensor = None,
        blip_img_features: torch.tensor = None,
        blip_mt_features: torch.tensor = None,
        blip_refs_features: list = None,
        beit3_mt_features: torch.tensor = None,
        beit3_refs_features: list = None,
        beit3_img_features: torch.tensor = None,
        stella_mt_features: torch.tensor = None,
        stella_refs_features: list = None,
        img: list = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Function that encodes both Source, MT and Reference and returns a quality score.

        :param src_tokens: SRC sequences [batch_size x src_seq_len]
        :param mt_tokens: MT sequences [batch_size x mt_seq_len]
        :param ref_tokens: REF sequences [batch_size x ref_seq_len]
        :param src_lengths: SRC lengths [batch_size]
        :param mt_lengths: MT lengths [batch_size]
        :param ref_lengths: REF lengths [batch_size]

        :param alt_tokens: Alternative REF sequences [batch_size x alt_seq_len]
        :param alt_lengths: Alternative REF lengths [batch_size]

        :return: Dictionary with model outputs to be passed to the loss function.
        """

    
        max_refs = 5

        def process_refs_features(batch_refs_features, mt_features, max_refs):
            batch_size = len(batch_refs_features)
            feature_dim = mt_features.shape[-1]
            
            refs_features_tensor = torch.zeros(batch_size, max_refs, feature_dim, 
                                            device=mt_features.device)
            refs_mask = torch.zeros(batch_size, max_refs, 
                                  device=mt_features.device, dtype=torch.bool)
            
            for batch_idx, refs in enumerate(batch_refs_features):
                valid_refs = [ref for ref in refs if ref is not None]
                for ref_idx, ref in enumerate(valid_refs[:max_refs]):
                    refs_features_tensor[batch_idx, ref_idx] = ref
                    refs_mask[batch_idx, ref_idx] = True
            
            return refs_features_tensor, refs_mask

        curr_dim = imgs_features.shape[-1] 
        target_dim = 768
        pad_size = target_dim - curr_dim


        imgs_features = torch.nn.functional.pad(
            imgs_features, (0, pad_size), "constant", 0
        )


        mt_features = torch.nn.functional.pad(
            mt_features, (0, pad_size), "constant", 0
        )

        padded_refs_features = []
        for ref_feature in refs_features:
            features = []
            for ref in ref_feature:
                if ref is not None:  
                    padded_ref = torch.nn.functional.pad(
                        ref, (0, pad_size), "constant", 0
                    )
                    features.append(padded_ref)
                else:
                    features.append(None)
            padded_refs_features.append(features)
        refs_features = padded_refs_features       

        clip_refs_features, clip_refs_mask = process_refs_features(
            refs_features, mt_features, max_refs)
        blip_refs_features, blip_refs_mask = process_refs_features(
            blip_refs_features, blip_mt_features, max_refs)
        beit3_refs_features, beit3_refs_mask = process_refs_features(
            beit3_refs_features, beit3_mt_features, max_refs)
        stella_refs_features, stella_refs_mask = process_refs_features(
            stella_refs_features, stella_mt_features, max_refs)

        diff_clip = self.mlp1(self.diff_feature_mlp1(imgs_features, mt_features))
        mul_clip = self.mlp1(self.hadamard_net1(imgs_features, mt_features))
        diff_beit3 = self.mlp3(self.diff_feature_mlp3(beit3_img_features, beit3_mt_features))
        mul_beit3 = self.mlp3(self.hadamard_net3(beit3_img_features, beit3_mt_features))
        
        x1 = [torch.cat([diff_clip, mul_clip], dim=1)]
        x2 = [torch.cat([diff_beit3, mul_beit3], dim=1)]
        x3 = []

        emb_dim = self.emb_dim 
        for i in range(blip_img_features.shape[1]):
            curr_dim = blip_img_features.shape[-1]  
            
           
            pad_size = emb_dim - curr_dim
            
            
            padded_img_features = torch.nn.functional.pad(
                blip_img_features[:, i, :], (0, pad_size), "constant", 0
            )
            padded_mt_features = torch.nn.functional.pad(
                blip_mt_features, (0, pad_size), "constant", 0
            )
            
            
            diff_blip = self.mlp2(self.diff_feature_mlp2(padded_img_features, padded_mt_features))
            mul_blip = self.mlp2(self.hadamard_net2(padded_img_features, padded_mt_features))
            x3.extend([torch.cat([diff_blip, mul_blip], dim=1)])
        
        assert torch.equal(clip_refs_mask, blip_refs_mask) and \
               torch.equal(blip_refs_mask, beit3_refs_mask) and \
               torch.equal(beit3_refs_mask, stella_refs_mask)

        x1 = torch.stack(x1, dim=1)
        x2 = torch.stack(x2, dim=1)
        x3 = torch.stack(x3, dim=1)

        combined_features = torch.cat([x1, x2, x3], dim=1)
        combined_features = self.pos_encoder(combined_features.permute(1, 0, 2))
        combined_features = combined_features.permute(1, 0, 2)
        combined_features = self.transformer(combined_features)

        query = self.learnable_query1.expand(combined_features.shape[0], -1, -1)
        attn_output, _ = self.cross_attention(
            query=query,
            key=combined_features,
            value=combined_features
        )
        img_score = self.ff(attn_output.flatten(1)).squeeze() 

        ref_scores = []
        blip_mt_features = torch.nn.functional.pad(
            blip_mt_features, (0, emb_dim - blip_mt_features.shape[-1]), "constant", 0
        )
        for ref_idx in range(5): 
            ref_diff_beit3 = self.mlp5(self.diff_feature_mlp5(
                beit3_refs_features[:, ref_idx], beit3_mt_features))
            ref_mul_beit3 = self.mlp5(self.hadamard_net5(
                beit3_refs_features[:, ref_idx], beit3_mt_features))

            
            current_blip_refs = torch.nn.functional.pad(
                blip_refs_features[:, ref_idx], 
                (0, emb_dim - blip_refs_features.shape[-1]), 
                "constant", 0
            )
            ref_diff_blip = self.mlp4(self.diff_feature_mlp4(
                current_blip_refs, blip_mt_features))
            ref_mul_blip = self.mlp4(self.hadamard_net4(
                current_blip_refs, blip_mt_features))

           
            ref_diff_stella = self.mlp6(self.diff_feature_mlp6(
                stella_refs_features[:, ref_idx], stella_mt_features))
            ref_mul_stella = self.mlp6(self.hadamard_net6(
                stella_refs_features[:, ref_idx], stella_mt_features))

            
            ref_x1 = torch.cat([ref_diff_blip, ref_mul_blip], dim=1).unsqueeze(1)
            ref_x2 = torch.cat([ref_diff_beit3, ref_mul_beit3], dim=1).unsqueeze(1)
            ref_x3 = torch.cat([ref_diff_stella, ref_mul_stella], dim=1).unsqueeze(1)

            ref_combined_features = torch.cat([ref_x1, ref_x2, ref_x3], dim=1)
            ref_combined_features = self.pos_encoder(ref_combined_features.permute(1, 0, 2))
            ref_combined_features = ref_combined_features.permute(1, 0, 2)
            ref_combined_features = self.transformer(ref_combined_features)

            query = self.learnable_query1.expand(ref_combined_features.shape[0], -1, -1)
            ref_attn_output, _ = self.cross_attention(
                query=query,
                key=ref_combined_features,
                value=ref_combined_features
            )
            
            current_ref_score = self.ff2(ref_attn_output.flatten(1)).squeeze()  
            
            ref_mask = (clip_refs_mask[:, ref_idx] & 
                       blip_refs_mask[:, ref_idx] & 
                       beit3_refs_mask[:, ref_idx] & 
                       stella_refs_mask[:, ref_idx])
            current_ref_score = torch.where(ref_mask, current_ref_score, 
                                          torch.zeros_like(current_ref_score))
            
            ref_scores.append(current_ref_score)
        ref_scores = torch.stack(ref_scores, dim=1)
        
        ref_score = torch.max(ref_scores, dim=1).values

        score = torch.where(
            torch.any(clip_refs_mask, dim=1), 
            0.5 * img_score + 0.5 * ref_score,  
            img_score  
        )

        return {
            "score": score, 
            "img_score": img_score, 
            "ref_score": ref_score,
            "ref_scores": ref_scores  
        }
