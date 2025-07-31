# Copyright 2024 The MonoGS Authors.

# Licensed under the License issued by the MonoGS Authors
# available here: https://github.com/muskie82/MonoGS/blob/main/LICENSE.md

import torch
from thirdparty.gaussian_splatting.utils.loss_utils import ssim

import cv2
import numpy as np
from torchvision.transforms import Resize


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


# Not used, but kept for reference
def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


# Not used, but kept for reference
def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)

    mask = viewpoint.mask
    if mask is not None:
        mask = mask.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    loss = 0
    if config["Training"]["ssim_loss"]:
        ssim_loss = 1.0 - ssim(image, gt_image)
        
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    if mask is not None:
        rgb_pixel_mask = torch.logical_and(rgb_pixel_mask, mask > 0.5)
    
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    if config["Training"]["ssim_loss"]:
        hyperparameter = config["opt_params"]["lambda_dssim"]
        loss += (1.0 - hyperparameter) * l1_rgb + hyperparameter * ssim_loss
    else:
        loss += l1_rgb

    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    if mask is not None:
        depth_pixel_mask = torch.logical_and(depth_pixel_mask, mask > 0.5)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * loss.mean() + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()


def get_dimension(image, down_scale=8):
    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    H, W, _ = cv2.resize(image, (w1, h1))[:h1-h1%down_scale, :w1-w1%down_scale].shape
    return H, W

def get_dimension_from_file(imgfile, down_scale=8):
    """ Get proper image dimension for DROID """
    image = cv2.imread(imgfile)
    return get_dimension(image, down_scale)

# def preprocess_masks(image, mask):
#     """ Resize masks for masked droid """
#     H, W = get_dimention(image)
#     resize_1 = Resize((H, W), antialias=True)
#     resize_2 = Resize((H//8, W//8), antialias=True)
    
#     img_msks = []
#     for i in range(0, len(masks), 500):
#         m = resize_1(masks[i:i+500])
#         img_msks.append(m)
#     img_msks = torch.cat(img_msks)

#     conf_msks = []
#     for i in range(0, len(masks), 500):
#         m = resize_2(masks[i:i+500])
#         conf_msks.append(m)
#     conf_msks = torch.cat(conf_msks)

#     return img_msks, conf_msks
# def preprocess_mask(image, mask, down_scale=8):
#     """ Resize masks for masked droid """
#     print(image.shape)
#     H, W = get_dimension(image)
#     resize_1 = Resize((H, W), antialias=True)
#     resize_2 = Resize((H//down_scale, W//down_scale), antialias=True)
    
#     img_msk = resize_1(mask.unsqueeze(0))
#     conf_msk = resize_2(mask.unsqueeze(0))

#     return img_msk, conf_msk
def preprocess_mask(mask, H, W, down_scale=8):
    """ Resize masks for masked droid """
    resize_1 = Resize((H, W), antialias=True)
    resize_2 = Resize((H//down_scale, W//down_scale), antialias=True)
    
    img_msk = resize_1(mask.unsqueeze(0))
    conf_msk = resize_2(mask.unsqueeze(0))

#     img_msk = 1.0 - img_msk
#     conf_msk = 1.0 - conf_msk # to mask out droid weight
# #     img_msk = img_msk > 0.5 # binary
# #     conf_msk = conf_msk > 0.5
    
    img_msk = img_msk < 0.5 # binary: float -> bool, produce human=1, bg=0
    conf_msk = conf_msk < 0.5
    
    return img_msk, conf_msk


