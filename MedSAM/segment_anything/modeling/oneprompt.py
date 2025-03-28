import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
import argparse
import random
from datetime import datetime
from .oneprompt_former import OnePromptFormer
import shutil
import glob
from utils import *

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

class OnePrompt(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    
    def __init__(
        self,
        image_encoder,
        oneprompt_former,
        mask_decoder,
        prompt_encoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.oneprompt_former = oneprompt_former
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        # freeze image encoder encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    
    def forward(self,
                batched_input: List[Dict[str, Any]], 
                template_input: List[Dict[str, Any]],
                multimask_output: bool,
                ) -> List[Dict[str, torch.Tensor]]:
        device = self.device
        imgs = batched_input['image'].to(dtype = torch.float32, device = device)
        masks = batched_input['label'].to(dtype = torch.float32, device = device)
        b, c, h, w = imgs.shape
        tmp_img = template_input['image'].to(dtype = torch.float32, device = device)[0,:,:,:].unsqueeze(0).repeat(b, 1, 1, 1)
        tmp_mask = template_input['label'].to(dtype = torch.float32, device = device)[0,:,:,:].unsqueeze(0).repeat(b, 1, 1, 1)
        # do not compute gradients for prompt encoder
        if 'pt' not in template_input:
            # tmp_img, pt, tmp_mask = generate_click_prompt3D(tmp_img, tmp_mask)
            pt, point_labels = generate_click_prompt2D(tmp_img, tmp_mask)
            
        else:
            pt = template_input['pt']
            point_labels = template_input['p_label']

        if point_labels[0] != -1:
            point_coords = pt
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
            coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
            pt = (coords_torch, labels_torch)
        # print("pt", pt)
        
        with torch.no_grad():
            r_emb, r_list = self.image_encoder(imgs)  # (B, 256, 64, 64)
            t_emb, t_list= self.image_encoder(tmp_img)  # (B, 256, 64, 64)
        outputs = []
        
        p1, p2, sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=pt,
            boxes=None,
            masks=None,
        )
       
        r_emb, mixed_features = self.oneprompt_former(
            skips_raw = r_list,
            skips_tmp = t_list,
            raw_emb = r_emb,
            tmp_emb = t_emb,
            pt1 = p1,
            pt2 = p2,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False)
        

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=r_emb,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=mixed_features,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
       
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=batched_input["image"].shape[-2:],
            # original_size=batched_input["original_size"],
        )
        # print("masks", masks.shape)
        # masks = masks > self.mask_threshold
        outputs.append(
            {
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            }
        )
        return outputs
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        # original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        # masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
