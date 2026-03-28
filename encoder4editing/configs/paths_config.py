# configs/paths_config.py

"""
Paths configuration for encoder4editing.

Here we define:
- dataset_paths: root folders for each dataset
- model_paths: paths to pretrained models (StyleGAN, etc.)
"""

# configs/paths_config.py

dataset_paths = {
    'ffhq': '',
    'celeba_test': '',
    'cars_train': '',
    'cars_test': '',
    'horse_train': '',
    'horse_test': '',
    'church_train': '',
    'church_test': '',
    'cats_train': '',
    'cats_test': '',

    # Brain MRI dataset
    'brain_mri_train': '/content/localdata/train_unzipped',
    'brain_mri_test': '/content/localdata/test_unzipped',
}

model_paths = {
    # IMPORTANT: keep this key so TrainOptions does not crash
    'stylegan_ffhq': '/content/drive/MyDrive/encoder4editing/encoder4editing/stylegan2-ffhq-config-f.pt',

    # Your StyleGAN brain model (converted to .pt)
    'stylegan_brain': '/content/drive/MyDrive/TESIS/StyleGan2/Modelsinge/stylegan_brain_rosinality.pt',

    # Not used for MRI, leave them empty
    'ir_se50': '/content/drive/MyDrive/encoder4editing/encoder4editing/model_ir_se50.pth',
    'shape_predictor': '',
    'moco': '/content/drive/MyDrive/encoder4editing/encoder4editing/moco_v2_800ep_pretrain.pt',
}
