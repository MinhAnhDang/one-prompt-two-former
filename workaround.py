# import os
# from datetime import datetime
# from collections import OrderedDict
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
# import torchvision
# import torchvision.transforms as transforms
# from skimage import io
# from torch.utils.data import DataLoader
# #from dataset import *
# from torch.autograd import Variable
# from PIL import Image
# from tensorboardX import SummaryWriter
# #from models.discriminatorlayer import discriminator
# from dataset import *
# from conf import settings
# import time
# import cfg
# from tqdm import tqdm
# from torch.utils.data import DataLoader, random_split
# from utils import *
# import function 
# import torch
# from MedSAM.segment_anything import sam_model_registry
# from skimage import io, transform
# import torch.nn.functional as F

# args = cfg.parse_args()

# GPUdevice = torch.device('cuda', args.gpu_device)
# net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
# image_embedding = torch.randn(1, 64, 64, 768).to(GPUdevice)
# skips = [torch.randn(1, 64, 64, 768).to(GPUdevice) for _ in range(12)]

# box_np = np.array([[100,100, 300, 300]])
# # transfer box_np t0 1024x1024 scale
# box_1024 = box_np / np.array([1024, 1024, 1024, 1024]) * 1024
# box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=image_embedding.device)
# if len(box_torch.shape) == 2:
#     box_torch = box_torch[:, None, :] # (B, 1, 4)

# p1, p2, sparse_embeddings, dense_embeddings = net.prompt_encoder(
#     points=None,
#     boxes=box_torch,
#     masks=None,
#     doodles=None,
# )
# print("Prompt 1:", p1.shape)
# print("Prompt 2:", p2.shape)
# print("Sparse embeddings:", sparse_embeddings.shape)
# print("Dense embeddings:", dense_embeddings.shape)

# pred, _ = net.mask_decoder(
#         skips_raw = skips,
#         skips_tmp = skips,
#         raw_emb = image_embedding,
#         tmp_emb = image_embedding,
#         pt1 = p1,
#         pt2 = p2,
#         image_pe=net.prompt_encoder.get_dense_pe(), 
#         sparse_prompt_embeddings=sparse_embeddings,
#         dense_prompt_embeddings=dense_embeddings, 
#         multimask_output=False,
#     )
# print(net)

# # print("----------------------------------------------------------------")
# # MedSAM_CKPT_PATH = "MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
# # medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
# # print(medsam_model)

# optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

# transform_train = transforms.Compose([
#         transforms.Resize((args.image_size,args.image_size)),
#         transforms.ToTensor(),
#     ])

# transform_train_seg = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((args.image_size,args.image_size)),
# ])

# transform_test = transforms.Compose([
#     transforms.Resize((args.image_size, args.image_size)),
#     transforms.ToTensor(),
# ])

# transform_test_seg = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((args.image_size, args.image_size)),
    
# ])
# isic_train_dataset = ISIC2016(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
# isic_test_dataset = ISIC2016(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

# nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
# nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)

# # print(isic_test_dataset[0]['image'].shape)
# # print(next(iter(nice_test_loader)).keys())
# # print(next(iter(nice_test_loader))['image'].shape)
# net.eval()
# pack = next(iter(nice_test_loader))
# tmp_img = pack['image'].to(dtype = torch.float32, device = GPUdevice)[0,:,:,:].unsqueeze(0).repeat(args.b, 1, 1, 1)
# tmp_mask = pack['label'].to(dtype = torch.float32, device = GPUdevice)[0,:,:,:].unsqueeze(0).repeat(args.b, 1, 1, 1)
# if 'pt' not in pack:
#     tmp_img, pt, tmp_mask = generate_click_prompt(tmp_img, tmp_mask)
# else:
#     pt = pack['pt']
#     point_labels = pack['p_label']

# if point_labels[0] != -1:
#     # point_coords = onetrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
#     point_coords = pt
#     coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
#     labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
#     coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
#     pt = (coords_torch, labels_torch)


# imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
# masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)

# name = pack['image_meta_dict']['filename_or_obj']
# b_size,c,w,h = imgs.size()
# mask_type = torch.float32
# imgs = imgs.to(dtype = mask_type,device = GPUdevice)
# with torch.no_grad():
#     imge, skips= net.image_encoder(imgs)
#     timge, tskips = net.image_encoder(tmp_img)
#     print("Image embedding:", imge.shape)
#     print("Skip connections:", skips[-1].shape)
#     # print("Template image embedding:", timge.shape)
#     # print("Template skip connections:", tskips.shape)
#     p1, p2, se, de = net.prompt_encoder(
#             points=pt,
#             boxes=None,
#             doodles= None,
#             masks=None,
#         )
#     print("Prompt 1:", p1.shape)
#     print("Prompt 2:", p2.shape)
#     print("Sparse embeddings:", se.shape)
#     print("Dense embeddings:", de.shape)
#     pred, _ = net.mask_decoder(
#         skips_raw = skips,
#         skips_tmp = skips,
#         raw_emb = imge,
#         tmp_emb = imge,
#         pt1 = p1,
#         pt2 = p2,
#         image_pe=net.prompt_encoder.get_dense_pe(), 
#         sparse_prompt_embeddings=se,
#         dense_prompt_embeddings=de, 
#         multimask_output=False,
#     )

# %% environment and functions
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from segment_anything import sam_model_registry
from segment_anything.modeling import OnePromptFormer
from skimage import io, transform
import torch.nn.functional as F
# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
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

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

MedSAM_CKPT_PATH = "MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH).to(device)
medsam_model.eval()
torch.cuda.empty_cache()
# torch.cuda.set_per_process_memory_fraction(1.0, device=0)
img_np = io.imread('data/ytma10_010704_benign1_ccd.tif')
if len(img_np.shape) == 2:
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
else:
    img_3c = img_np
H, W, _ = img_3c.shape

img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
img_1024 = (img_1024 - img_1024.min()) / np.clip(
    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
)  # normalize to [0, 1], (H, W, 3)
# convert the shape to (3, H, W)
img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

box_np = np.array([[100,100, 300, 300]])
# transfer box_np t0 1024x1024 scale
box_1024 = box_np / np.array([W, H, W, H]) * 1024
with torch.no_grad():
    print("Image shape", img_1024_tensor.shape)
    image_embedding, skips = medsam_model.image_encoder(img_1024_tensor) # (1, 256, 64, 64)
    print("Image embedding", image_embedding.shape)
    # print("Skips embedding", skips[-1].shape)
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=image_embedding.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)

    p1, p2, sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    # print("Prompt 1:", p1.shape)
    # print("Prompt 2:", p2.shape)
    # print("Sparse embeddings:", sparse_embeddings.shape)
    # print("Dense embeddings:", dense_embeddings.shape)
    depth = len(skips)
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
    ).to(image_embedding.device)
    
    r_emb, x = features_former(
                skips_raw = skips,
                skips_tmp = skips,
                raw_emb = image_embedding,
                tmp_emb = image_embedding,
                pt1 = p1,
                pt2 = p2,
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False)
    print("x after all former", x.shape)
    # prompt_former = OnePromptFormer(
    #                 depth=depth,
    #                 embedding_dim = prompt_embed_dim, 
    #                 prompt_embed_dim = prompt_embed_dim,
    #                 token_num = token_num, 
    #                 num_heads = 2, 
    #                 mlp_dim = mlp_dim
    # )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=r_emb, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=x, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    print("Predicted mask:", medsam_seg.shape)
    
    
# medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(img_3c)
# show_box(box_np[0], ax[0])
# ax[0].set_title("Input Image and Bounding Box")
# ax[1].imshow(img_3c)
# show_mask(medsam_seg, ax[1])
# show_box(box_np[0], ax[1])
# ax[1].set_title("MedSAM Segmentation")
# plt.show()
