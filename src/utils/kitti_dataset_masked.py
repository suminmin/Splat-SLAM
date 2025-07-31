import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov

import joblib
import einops
from glob import glob
from scipy.spatial.transform import Rotation

from .datasets import BaseDataset

import sys
sys.path.append("/root/sumi/sumi_repository/gaussian_splatting/_AD/drivestudio")
import kornia
from omegaconf import OmegaConf
# from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str

KITTI_DATASET_CONFIG = "/root/sumi/sumi_repository/gaussian_splatting/_AD/drivestudio/configs/datasets/kitti/1cams.yaml"

KITTI_DATA_DIR = "/root/sumi/sumi_repository/gaussian_splatting/_AD/drivestudio/data/kitti"


def load_and_preprocess_mask(mask_path):
#     mask_data_fullsize = np.load(mask_path).astype(np.uint8) * 255
    mask_data_fullsize = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8) 
    mask_data_fullsize = cv2.erode(mask_data_fullsize, kernel, iterations=1) 
    mask_data_fullsize = cv2.dilate(mask_data_fullsize, kernel, iterations=5) 
    mask_data_fullsize = cv2.bitwise_not(mask_data_fullsize)
    return mask_data_fullsize

class MaskedKITTI(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(MaskedKITTI, self).__init__(cfg, device)
        
        self.fps = 30
        
        self.has_depth_gt = False
        
        # self.color_paths, _, self.poses, self.mask_paths = self.load_emdb(self.input_folder, frame_rate=self.fps)
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        # if max_frames < 0:
        #     max_frames = int(1e5)

        #
        dataset_cfg = OmegaConf.load(KITTI_DATASET_CONFIG)
        
        dataset_cfg.data.data_root = cfg["data"]["dataset_root"] # f"{KITTI_DATA_DIR}/processed/"
        dataset_cfg.data.scene_idx = cfg["scene"] # "2011_09_26_drive_0091_sync"

        # dataset_cfg.data.start_timestep = 0
        # dataset_cfg.data.end_timestep = max_frames
        
        # self.dataset = DrivingDataset(data_cfg=dataset_cfg.data)

        data_cfg = dataset_cfg.data
        data_path = os.path.join(data_cfg.data_root, data_cfg.scene_idx)

        if os.path.exists(os.path.join(data_path, "ego_pose")):
            total_frames = len(os.listdir(os.path.join(data_path, "ego_pose")))

        # if data_cfg.end_timestep == -1:
        if max_frames == -1:
            end_timestep = total_frames - 1
        else:
            end_timestep = data_cfg.end_timestep
        # to make sure the last timestep is included
        end_timestep = end_timestep + 1

        start_timestep = data_cfg.start_timestep

        self.dataset = import_str(data_cfg.pixel_source.type)(
            data_cfg.dataset,
            data_cfg.pixel_source,
            data_path,
            start_timestep,
            end_timestep,
            device="cpu",
        )

        self.poses = self.load_kitti()
        #

        # self.color_paths = self.color_paths[:max_frames][::stride]
        # self.depth_paths = None
        # self.poses = self.poses[:max_frames][::stride]
        max_frames = end_timestep
        self.color_paths = self.dataset.camera_data[0].img_filepaths[:max_frames][::stride]
        self.depth_paths = None
        self.poses = self.poses[:max_frames][::stride]

        self.mask_paths = self.dataset.camera_data[0].dynamic_mask_filepaths[:max_frames][::stride]

        self.w2c_first_pose = np.linalg.inv(self.poses[0])

        # self.n_img = len(self.color_paths)
        # self.n_img = self.dataset.frame_num
        self.n_img = self.dataset.num_frames

    def get_c2w_pose_gt(self):
        poses = self.dataset.camera_data[0].cam_to_worlds.cpu()
        return poses.numpy() # 4x4

    def load_kitti(self):
        poses = self.get_c2w_pose_gt().astype(np.float64) # tx ty tz qx qy qz qw

        # tstamp_image = np.arange(len(poses)).astype(np.float64) / self.fps

        return poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose
    

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)

        mask_path = self.mask_paths[index]
        mask_data_fullsize = load_and_preprocess_mask(mask_path)

        # image_info, _ = self.dataset.full_image_set[index]
        # color_data_fullsize = (image_info["pixels"].cpu().numpy() * 255).astype(np.uint8)
        # mask_data_fullsize = (image_info["dynamic_masks"].cpu().numpy()  * 255).astype(np.uint8)
        
        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx_orig, self.cx_orig, self.fy_orig, self.cy_orig
            # undistortion is only applied on color image, not depth!
            color_data_fullsize = cv2.undistort(color_data_fullsize, K, self.distortion)
            mask_data_fullsize = cv2.undistort(mask_data_fullsize, K, self.distortion)

        outsize = (self.H_out_with_edge, self.W_out_with_edge)

        color_data = cv2.resize(color_data_fullsize, (self.W_out_with_edge, self.H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        # color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)
        
        mask_data = cv2.resize(mask_data_fullsize, (self.W_out_with_edge, self.H_out_with_edge), interpolation=cv2.INTER_NEAREST)
        mask_data = torch.from_numpy(mask_data).float() / 255.0 # (h,w)
        
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]
        
        mask_data = mask_data.unsqueeze(dim=0)  # [1, h, w]

#         depth_data_fullsize = self.depthloader(index,self.depth_paths,self.png_depth_scale)
#         if depth_data_fullsize is not None:
#             depth_data_fullsize = torch.from_numpy(depth_data_fullsize).float()
#             depth_data = F.interpolate(
#                 depth_data_fullsize[None, None], outsize, mode='nearest')[0, 0]


        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]
#             depth_data = depth_data[:, edge:-edge]
            mask_data = mask_data[:, :, edge:-edge]

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
#             depth_data = depth_data[edge:-edge, :]
            mask_data = mask_data[:, edge:-edge, :]

        if self.poses is not None:
            pose = torch.from_numpy(self.poses[index]).float() #torch.from_numpy(np.linalg.inv(self.poses[0]) @ self.poses[index]).float()
        else:
            pose = None

        color_data_fullsize = cv2.cvtColor(color_data_fullsize,cv2.COLOR_BGR2RGB)
        color_data_fullsize = color_data_fullsize / 255.
        color_data_fullsize = torch.from_numpy(color_data_fullsize)

        depth_data = None
        
        mask_data_fullsize = mask_data_fullsize / 255.
        mask_data_fullsize = torch.from_numpy(mask_data_fullsize)
        
        return index, color_data, depth_data, pose, mask_data


dataset_dict = {
    "kitti":  MaskedKITTI,
}


def get_dataset(cfg, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, device=device)
