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

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

class OnePrompt(nn.Module):
    def __init__(
        self,
        image_encoder,
        onepropmt_former,
        mask_decoder,
        prompt_encoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        super().__init__()
        # self.args = args
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.oneprompt_former = onepropmt_former
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        # freeze image encoder encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    
    @torch.no_grad()
    def forward(self,
                batched_input: List[Dict[str, Any]], 
                template_input: List[Dict[str, Any]],
                multimask_output: bool,
                ) -> List[Dict[str, torch.Tensor]]:
        device = self.device
        input_images = torch.stack([self.preprocess(x) for x in batched_input["image"].to(device)], dim=0)
        template_images = torch.stack([self.preprocess(x) for x in template_input["image"].to(device)], dim=0)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            r_emb, r_list = self.image_encoder(input_images)  # (B, 256, 64, 64)
            t_emb, t_list= self.image_encoder(template_images)  # (B, 256, 64, 64)
        outputs = []
        for image_record, r_list, t_list, r_emb, t_emb in zip(batched_input, r_list, t_list, r_emb, t_emb):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
                
            p1, p2, sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
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
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )

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
        original_size: Tuple[int, ...],
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
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
