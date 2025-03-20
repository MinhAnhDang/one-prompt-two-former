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
import torch
from MedSAM.segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

# print(net)

# print("----------------------------------------------------------------")
# MedSAM_CKPT_PATH = "MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
# medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
# print(medsam_model)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

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

# print(isic_test_dataset[0]['image'].shape)
# print(next(iter(nice_test_loader)).keys())
# print(next(iter(nice_test_loader))['image'].shape)
net.eval()
pack = next(iter(nice_test_loader))
tmp_img = pack['image'].to(dtype = torch.float32, device = GPUdevice)[0,:,:,:].unsqueeze(0).repeat(args.b, 1, 1, 1)
tmp_mask = pack['label'].to(dtype = torch.float32, device = GPUdevice)[0,:,:,:].unsqueeze(0).repeat(args.b, 1, 1, 1)
if 'pt' not in pack:
    tmp_img, pt, tmp_mask = generate_click_prompt(tmp_img, tmp_mask)
else:
    pt = pack['pt']
    point_labels = pack['p_label']

if point_labels[0] != -1:
    # point_coords = onetrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
    point_coords = pt
    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    pt = (coords_torch, labels_torch)


imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)

name = pack['image_meta_dict']['filename_or_obj']
b_size,c,w,h = imgs.size()
mask_type = torch.float32
imgs = imgs.to(dtype = mask_type,device = GPUdevice)
with torch.no_grad():
    imge, skips= net.image_encoder(imgs)
    timge, tskips = net.image_encoder(tmp_img)
    print("Image embedding:", imge.shape)
    print("Skip connections:", skips[-1].shape)
    # print("Template image embedding:", timge.shape)
    # print("Template skip connections:", tskips.shape)
    p1, p2, se, de = net.prompt_encoder(
            points=pt,
            boxes=None,
            doodles= None,
            masks=None,
        )
    print("Prompt 1:", p1.shape)
    print("Prompt 2:", p2.shape)
    print("Sparse embeddings:", se.shape)
    print("Dense embeddings:", de.shape)
    pred, _ = net.mask_decoder(
        skips_raw = skips,
        skips_tmp = skips,
        raw_emb = imge,
        tmp_emb = imge,
        pt1 = p1,
        pt2 = p2,
        image_pe=net.prompt_encoder.get_dense_pe(), 
        sparse_prompt_embeddings=se,
        dense_prompt_embeddings=de, 
        multimask_output=False,
    )

