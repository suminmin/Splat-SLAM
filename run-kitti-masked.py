# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import argparse
import os

from thirdparty.glorie_slam import config
# from src.slam import SLAM
from src.slam_masked import SLAM
# from src.utils.datasets import get_dataset
# from src.utils.emdb_dataset import get_dataset
# from src.utils.emdb_dataset_masked import get_dataset
# from src.utils.kitti_dataset_masked import get_dataset
from src.utils.get_dataset import get_dataset
from time import gmtime, strftime
from colorama import Fore,Style

# from src.utils import emdb_dataset
from src.utils import emdb_dataset_masked as emdb_dataset
from thirdparty.monogs.utils.slam_utils import get_dimension_from_file, get_dimension

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("--only_tracking", action="store_true", help="Only tracking is triggered")
    args = parser.parse_args()
    # args = parser.parse_args(["./configs/KITTI/2011_09_26_drive_0001_sync.yaml"])

    torch.multiprocessing.set_start_method('spawn')

    cfg = config.load_config(
        args.config, './configs/splat_slam.yaml'
    )
    setup_seed(cfg['setup_seed'])
    
    ##
    tmp_dataset = get_dataset(cfg)
    intrinsics = tmp_dataset.dataset.camera_data[0].intrinsics
    intrinsic = intrinsics[0].cpu().numpy()
    # image_info, camera_info = tmp_dataset.dataset.full_image_set[0]

    # height = camera_info["height"].item()
    # width = camera_info["width"].item()
    # fx = camera_info["intrinsics"][0,0]
    # fy = camera_info["intrinsics"][1,1]
    # cx = camera_info["intrinsics"][0,2]
    # cy = camera_info["intrinsics"][1,2]
    height = tmp_dataset.dataset.camera_data[0].HEIGHT
    width = tmp_dataset.dataset.camera_data[0].WIDTH
    fx = intrinsic[0,0]
    fy = intrinsic[1,1]
    cx = intrinsic[0,2]
    cy = intrinsic[1,2]
    
    cfg["cam"]["H"] = height
    cfg["cam"]["W"] = width
    cfg["cam"]["fx"] = fx
    cfg["cam"]["fy"] = fy
    cfg["cam"]["cx"] = cx
    cfg["cam"]["cy"] = cy

    def get_dim():
        # tmp_dataset = get_dataset(cfg)
        # H_out, W_out = get_dimension_from_file( tmp_dataset.color_paths[0] )

        # image_info, _ = tmp_dataset.dataset.full_image_set[0]
        # color_data_fullsize = image_info["pixels"].cpu().numpy()
        # H_out, W_out = get_dimension( color_data_fullsize )
        tmp_dataset = get_dataset(cfg)
        H_out, W_out = get_dimension_from_file( tmp_dataset.color_paths[0] )
        return H_out, W_out
    
    H_out, W_out = get_dim()
    cfg["cam"]["H_out"] = H_out
    cfg["cam"]["W_out"] = W_out
    ##

    if args.only_tracking:
        cfg['only_tracking'] = True
        cfg['mono_prior']['predict_online'] = True

    output_dir = cfg['data']['output']
    output_dir = output_dir+f"/{cfg['scene']}"

    start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    start_info = "-"*30+Fore.LIGHTRED_EX+\
                 f"\nStart Splat-SLAM at {start_time},\n"+Style.RESET_ALL+ \
                 f"   scene: {cfg['dataset']}-{cfg['scene']},\n" \
                 f"   only_tracking: {cfg['only_tracking']},\n" \
                 f"   output: {output_dir}\n"+ \
                 "-"*30
    print(start_info)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config.save_config(cfg, f'{output_dir}/cfg.yaml')
    
    dataset = get_dataset(cfg)

    slam = SLAM(cfg, dataset)
    slam.run()

    end_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print("-"*30+Fore.LIGHTRED_EX+f"\nSplat-SLAM finishes!\n"+Style.RESET_ALL+f"{end_time}\n"+"-"*30)

