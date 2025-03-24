import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from MedSAM.segment_anything import sam_model_registry
from MedSAM.segment_anything.modeling import OnePrompt
from MedSAM.segment_anything.modeling import OnePromptFormer
from skimage import io, transform
import torch.nn.functional as F
import os
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function 


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))
    

### Load the data

transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])

transform_train_seg = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((args.image_size,args.image_size)),
])

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((args.image_size, args.image_size)),
        
])
isic_train_dataset = ISIC2016(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
isic_test_dataset = ISIC2016(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
# pack = next(iter(nice_test_loader))
# print(pack['image'].shape)
 
### Load the model
MedSAM_CKPT_PATH = "MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH).to(device)
depth = 12
prompt_embed_dim = 256
embed_dim = 768
out_chans = 256
token_num = 4096
patch_size = 16
mlp_dim: int = 1024
features_former = OnePromptFormer(
                    depth=depth,
                    embed_dim = embed_dim, 
                    prompt_embed_dim = prompt_embed_dim,
                    out_chans = out_chans,
                    token_num = token_num, 
                    patch_size=16,
                    mlp_dim = mlp_dim
    ).to(device)
model = OnePrompt(
    image_encoder=medsam_model.image_encoder,
    onepropmt_former=features_former,
    mask_decoder=medsam_model.mask_decoder,
    prompt_encoder=medsam_model.prompt_encoder,
).to(device)
# print(model)
### Train the model
for data in nice_train_loader:
    prediction = model(data, data, multimask_output=True)
    # print(model.device)
    print(prediction.shape)
    