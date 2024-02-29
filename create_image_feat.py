import os
import clip
import h5py
import numpy as np
from PIL import Image
import pdb

import torch
import torch.nn as nn
from torchvision import transforms


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("RN50", device=device)
    clip_visual_modules = list(clip_model.visual.children())[:-1]
    clip_visual = nn.Sequential(*clip_visual_modules)
    for p in clip_visual.parameters():
        p.requires_grad = False

    for root, dirs, files in os.walk("/media/xunuo/dataset/AI2THOR/Scene_Data/"):
        for name in dirs:
            scene_data_path = os.path.join(root, name)
            f1 = h5py.File(os.path.join(scene_data_path, "images.hdf5"), 'r')
            f2 = h5py.File(os.path.join(scene_data_path, "clip_featuremap.hdf5"), 'w')

            print(scene_data_path)

            for fk in f1.keys():
                state = np.array(f1[fk])
                state = Image.fromarray(state)
                state = preprocess(state).unsqueeze(0).to(device)
                state = clip_visual(state.to(torch.float16))
                state = state.to(torch.float32)

                f2.create_dataset(fk, data=state.cpu().numpy())

            f1.close()
            f2.close()