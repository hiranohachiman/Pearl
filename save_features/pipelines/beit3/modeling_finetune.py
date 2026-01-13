import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import numpy as np

from pipelines.beit3 import utils
from pipelines.beit3.beit3_wrapper import BEiT3Wrapper, _get_base_config, _get_large_config
from torchscale.architecture.config import EncoderConfig
from transformers import XLMRobertaTokenizer
from PIL import Image
import torchvision.transforms as transforms
from torchscale.model.BEiT3 import BEiT3
import open_clip


class BEiT3ForRetrieval(BEiT3Wrapper):
    def __init__(
            self, 
            args,
            **kwargs
    ):
        super(BEiT3ForRetrieval, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.language_head.apply(self._init_weights)
        self.vision_head.apply(self._init_weights)
        self.criterion = utils.ClipLoss(
            rank=utils.get_rank(), 
            world_size=utils.get_world_size(), 
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image=None, text_description=None, padding_mask=None, only_infer=True, **kwargs):
        if image is not None:
            outputs = self.beit3(
                textual_tokens=None, 
                visual_tokens=image, 
                text_padding_position=None, 
            )
            x = outputs["encoder_out"]
            vision_cls = self.vision_head(x[:, 0, :])
            vision_cls = F.normalize(vision_cls, dim=-1)
        else:
            vision_cls = None

        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description, 
                visual_tokens=None, 
                text_padding_position=padding_mask, 
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
        else:
            language_cls = None
        
        if only_infer:
            return vision_cls, language_cls
        else:
            loss, logits_per_image, logits_per_text = self.criterion(
                vision_cls, language_cls, self.logit_scale.exp())
            return loss, vision_cls, language_cls


if __name__ == "__main__":
    img_size = 224  # Image size
    patch_size = 16  # Patch size
    vocab_size = 64010  # Vocabulary size
    drop_path_rate = 0  # Dropout rate
    mlp_ratio = 4  # Multiplier for the MLP dimension
    checkpoint_activations = False  # Whether to enable checkpoint activations
    args = EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12,
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12,
        checkpoint_activations=checkpoint_activations,
    )

    model = BEiT3ForRetrieval(args=args, only_infer=True)

    texts = ["a man"]

    tokenizer = XLMRobertaTokenizer("/home/initial/workspace/smilab24/m1_project/DENEB/experiments/beit3.spm")
    text_tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    print(text_tokens)
    print(model)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by the model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization
                            std=[0.229, 0.224, 0.225]),
    ])
    state_dict = '/home/initial/workspace/smilab24/m1_project/operation_check/beit3/beit3_base_itc_patch16_224.pth'
    # Remove keys from the state_dict that do not include 'beit3.'
    # model.load_state_dict(state_dict, strict=False)
    utils.load_model_and_may_interpolate(state_dict, model, 'model|module', "")
    # Store multiple image paths inside a list
    image_paths = [
        '/home/initial/workspace/smilab24/m1_project/operation_check/caption_evaluation/example/im1.jpg',
    ]

    # Build a list of image tensors
    input_tensors = []

    for img_path in image_paths:
        # Load and preprocess each image
        image = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(image)
        input_tensors.append(input_tensor)

    # Create a batch tensor
    batch_input = torch.stack(input_tensors, dim=0)  # Add the batch dimension up front

    # Extract features with the model
    with torch.no_grad():
        # Extract input_ids from text_tokens
        text_input_ids = text_tokens['input_ids']
        image_features, text_features = model(image=batch_input, text_description=text_input_ids)
    
    print(text_features.shape)
    print(image_features.shape)

    # Cosine similarity
    similarity = F.cosine_similarity(text_features, image_features, dim=-1)
    print(similarity)
    # Check whether the result matches CLIP
