import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from MedSAM.segment_anything import oneprompt_model_registry
# from skimage import io, transform
import torch.nn.functional as F
import os
from datetime import datetime
import monai
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
# from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import wandb
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityd,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    Resized,
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)

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
    
args = cfg.parse_args()
# if args.use_wandb:
#     import wandb

#     wandb.login()
#     wandb.init(
#         project=args.task_name,
#         config={
#             "lr": args.lr,
#             "batch_size": args.batch_size,
#             "data_path": args.tr_npy_path,
#             "model_type": args.model_type,
#         },
#     )
    
# %% set up model for training
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(model_save_path, exist_ok=True)
shutil.copyfile(
    __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
)
# %% set up model
 
### Load the model
MedSAM_CKPT_PATH = "MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
medSAM_weights = torch.load(MedSAM_CKPT_PATH, map_location=device)
model = oneprompt_model_registry["vit_b"](checkpoint=None).to(device)
missing_keys, unexpected_keys = model.load_state_dict(medSAM_weights, strict=False)
# Log missing and unexpected keys for debugging purposes
# print("Missing keys:", missing_keys)
# print("Unexpected keys:", unexpected_keys)

prompt_mask_dec_params = list(model.mask_decoder.parameters()) + list(model.prompt_encoder.parameters())+list(model.oneprompt_former.parameters())
# Opimizer
optimizer = torch.optim.AdamW(
    prompt_mask_dec_params, lr=args.lr, weight_decay=args.weight_decay
)
# Dice loss
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
# cross entropy loss
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

# %% Load the dataset
if args.dataset == 'isic':
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
    # print("Number of training samples: ", len(isic_train_dataset))
    # nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    # nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # pack = next(iter(nice_test_loader))
    # print(pack['image'].shape)
    
    train_transforms = Compose(
        [   
            LoadImaged(keys=["image", "label"], reader="PILReader", ensure_channel_first=True),
            ScaleIntensityd(keys=["image","label"], minv=0.0, maxv=1.0),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            Resized(keys=["image", "label"],spatial_size=(args.image_size, args.image_size)),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityd(keys=["image","label"], minv=0.0, maxv=1.0),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )
    data_dir = args.data_path
    split_JSON = "dataset_0.json"

    datasets = os.path.join(data_dir, split_JSON)
    datalist = load_decathlon_datalist(datasets, True, "training")
    # val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=0,
    )
    nice_train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=args.b, shuffle=True)
    # val_ds = CacheDataset(
    #     data=val_files, transform=val_transforms, cache_num=2, cache_rate=1.0, num_workers=0
    # )
    # nice_val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
### Train the model
# %% train
num_epochs = args.num_epochs
iter_num = 0
losses = []
best_loss = 1e10
start_epoch = 0

# Resume training
if args.resume is not None:
    if os.path.isfile(args.resume):
        ## Map model to be loaded to specified single GPU
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        
model.train()
print("Number of total parameters: ", sum(p.numel() for p in model.parameters()),)  # 93735472
print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))  # 93729252
print("Number of prompt encoder, mask decoder and one-prompt-former parameters: ", sum(p.numel() for p in prompt_mask_dec_params if p.requires_grad))
# print(model)
if args.use_amp:
    scaler = torch.amp.GradScaler('cuda')
    
for epoch in range(start_epoch, num_epochs):
    epoch_loss = 0
    for step, data in enumerate(tqdm(nice_train_loader)):
        optimizer.zero_grad()
        if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(data, data, multimask_output=False)[0]
                    loss = seg_loss(outputs['masks'], data['label'].to(device)) + ce_loss(
                        outputs['masks'], data['label'].float().to(device)
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
                outputs = model(data, data, multimask_output=False)[0]
                loss = seg_loss(outputs['masks'], data['label'].to(device)) + ce_loss(outputs['masks'], data['label'].float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        epoch_loss += loss.item()
        iter_num += 1
    epoch_loss /= step
    losses.append(epoch_loss)
    if args.use_wandb:
        wandb.log({"epoch_loss": epoch_loss})
    print(
        f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
    )
    ## save the latest model
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
    ## save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

