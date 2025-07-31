from src.utils.datasets import Replica, ScanNet, TUM_RGBD

from src.utils.emdb_dataset_masked import MaskedEMDB
from src.utils.kitti_dataset_masked import MaskedKITTI


dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "tumrgbd": TUM_RGBD,
    "emdb": MaskedEMDB,
    "kitti": MaskedKITTI,
}

def get_dataset(cfg, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, device=device)
