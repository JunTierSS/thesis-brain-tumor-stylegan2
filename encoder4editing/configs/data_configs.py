# configs/data_configs.py

"""
Dataset configuration for encoder4editing.

Each entry in DATASETS defines:
- which transform configuration to use
- which root folders to use for train/test (source/target)
"""

from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
 
    'brain_mri_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['brain_mri_train'],
        'train_target_root': dataset_paths['brain_mri_train'],
        'test_source_root': dataset_paths['brain_mri_test'],
        'test_target_root': dataset_paths['brain_mri_test'],
    },
}
