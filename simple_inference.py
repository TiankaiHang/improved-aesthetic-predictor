# import webdataset as wds
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import cv2
import glob

from warnings import filterwarnings

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm

from os.path import join
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip
from tqdm.auto import tqdm

from PIL import Image, ImageFile


#####  This script will predict the aesthetic score for this image file:

# img_path = "test.jpg"


# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

# ----------------------------------
#        load model
# ----------------------------------
model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

s = torch.load("sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo

model.load_state_dict(s)

model.to("cuda")
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   


image_files = glob.glob(f"/data/home/tiankai/datasets/guided-diffusion-new/exp/guided_diffusion/dec21_ldm32_beit_base_layer12_lr1e-4_099_099_img64_pred_x0__min_snr_5__fp16_bs8x32/1000000_samples1000_edm_20_scale2.0/*.png")

predicted_scores = []
for img_path in tqdm(image_files):
    pil_image = Image.open(img_path)

    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model2.encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy() )

    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

    # print( "Aesthetic score predicted by the model:")
    # print( prediction )
    
    predicted_scores.append(prediction.item())

predicted_scores = torch.tensor(predicted_scores)
# import pdb; pdb.set_trace()
sorted, inds = torch.sort(predicted_scores, descending=True)

for idx in inds[:100]:
    print(image_files[idx])

    base_file_name = os.path.basename(image_files[idx])

    # combine the images to a single one
    # iters = [200000, 400000, 600000, 800000, 1000000]
    iters = [50000, 200000, 400000, 1000000]
    exps = [
        # "jan20_abl_ldm32_beit_base_layer12_lr1e-4_099_099_img64_pred_x0__snr__fp16_bs8x32",
        # "jan20_abl_ldm32_beit_base_layer12_lr1e-4_099_099_img64_pred_x0__trunc_snr__fp16_bs8x32",
        "jan20_abl_ldm32_beit_base_layer12_lr1e-4_099_099_img64_pred_noise__const__fp16_bs8x32",
        "jan20_abl_ldm32_beit_base_layer12_lr1e-4_099_099_img64_pred_x0__const__fp16_bs8x32",
        "dec21_ldm32_beit_base_layer12_lr1e-4_099_099_img64_pred_x0__min_snr_5__fp16_bs8x32",
    ]
    img_size = 256
    margin = 10
    combined_image = np.ones(
        (img_size*len(exps)+margin*(len(exps)-1), img_size*len(iters)+margin*(len(iters)-1), 3),
        dtype=np.uint8,
    ) * 255
    for i in range(len(exps)):
        for j in range(len(iters)):
            exp = exps[i]
            iter = iters[j]
            img = cv2.imread(
                f"/data/home/tiankai/datasets/guided-diffusion-new/exp/guided_diffusion/"
                f"{exp}/{iter:06d}_samples1000_edm_20_scale2.0/{base_file_name}",
            )
            combined_image[
                i*margin+i*img_size : i*margin+(i+1)*img_size,
                j*margin+j*img_size : j*margin+(j+1)*img_size,
            ] = img

    cv2.imwrite(f"data/{base_file_name}", combined_image)
