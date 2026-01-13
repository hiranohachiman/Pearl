import open_clip
import torch
from PIL import Image
import torch.nn.functional as F

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="laion2b_s32b_b82k"
)
tokenizer = open_clip.get_tokenizer("ViT-L-14")
pacscheckpoint = torch.load("../../DENEB/experiments/openClip_ViT-L-14.pth")["state_dict"]

model.load_state_dict(pacscheckpoint, strict=True)

model = model.cuda()
model.eval()

texts = ["tennis"]
text_tokens = tokenizer(texts).cuda()
image_path = "/home/initial/workspace/smilab24/m1_project/operation_check/caption_evaluation/example/im1.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).cuda()

text_features = model.encode_text(text_tokens)
image_features = model.encode_image(image)
print(image_features.shape)
print(text_features.shape)
# cosine similarity
similarity = F.cosine_similarity(text_features, image_features, dim=-1)
print(similarity)
