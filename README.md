# Brain Tumor Classification with StyleGAN2-ADA Synthetic Augmentation

This repository contains the code, models, and results for the thesis:

**"Brain Tumor Classification Using Conditional StyleGAN2-ADA Synthetic Data Augmentation with Explainable AI Evaluation"**

by Junwei He

## Abstract

This thesis investigates the use of conditional StyleGAN2-ADA (StyleGAN-C) for generating synthetic brain MRI images to augment training data for a CNN-based tumor classifier. The framework addresses the challenge of limited labeled medical imaging data by generating class-conditional synthetic MRI slices for three tumor types: **glioma**, **meningioma**, and **pituitary**. The study evaluates the impact of synthetic augmentation on classification performance using ResNet-50 with K-Fold cross-validation, and includes explainability analysis via Grad-CAM and human-in-the-loop evaluation with neuroradiologists.

## Experiments

The thesis is organized around 5 main experiments:

| Experiment | Description | Thesis Section |
|---|---|---|
| **Exp 1** | Image generation with conditional StyleGAN2-ADA | Ch. 4.3, 5.1 |
| **Exp 2A** | Inference by generation: counterfactual trajectories + LPIPS | Ch. 4.5, 5.2 |
| **Exp 2B** | Latent inversion with E4E encoder, supervised encoder, PCA, SVM | Ch. 4.6, 5.3 |
| **Exp 2C** | Synthetic augmentation CNN (baseline, augmented, ablations, statistical tests) | Ch. 4.7, 5.4 |
| **Exp 3** | Classifier evaluation: HITL neuroradiology review + Grad-CAM XAI | Ch. 4.8, 5.5-5.6 |

## Notebooks

All experiment notebooks are in `notebooks/`, numbered by execution order and mapped to thesis experiments:

| # | Notebook | Experiment | Description |
|---|---|---|---|
| 00 | `00_data_preparation.ipynb` | Data Prep | Dataset splitting by patient ID to prevent data leakage |
| 01 | `01_exp1_generation.ipynb` | Exp 1 | Conditional image generation with StyleGAN-C (Figures 5.1-5.5) |
| 02 | `02_exp1_generation_metrics.ipynb` | Exp 1 | FID, PSNR, SSIM, LPIPS metrics for generated images (Tables 5.1-5.3) |
| 03 | `03_exp2a_counterfactual_inference.ipynb` | Exp 2A | LPIPS-based inference by generation + counterfactual videos (Tables 5.4-5.5) |
| 04 | `04_exp2b_latent_inversion_pca.ipynb` | Exp 2B | SVM on CNN/encoder/e4e features + PCA analysis (Tables 5.6-5.8) |
| 05 | `05_exp2c_baseline_cnn.ipynb` | Exp 2C | Baseline ResNet-50 CNN with K-Fold cross-validation (Table 5.15) |
| 06 | `06_exp2c_augmented_cnn.ipynb` | Exp 2C | Full pipeline: real + synthetic augmentation with K-Fold (Tables 5.10-5.18) |
| 07 | `07_exp2c_ablation_cas_synthetic_only.ipynb` | Exp 2C | Ablation: Classification Accuracy Score on synthetic-only data |
| 08 | `08_exp2c_ablation_cas_scoring.ipynb` | Exp 2C | Ablation: CAS grid search across N and truncation psi (Table 5.19) |
| 09 | `09_exp2c_ablation_oversample_augstrength.ipynb` | Exp 2C | Ablation 2-3: oversample baseline + weak vs strong augmentation (Tables 5.20-5.21) |
| 10 | `10_exp2c_leakage_analysis.ipynb` | Exp 2C | Data leakage investigation via generative augmentation (Sec 6.1.5) |
| 11 | `11_exp2c_statistical_tests.ipynb` | Exp 2C | McNemar test + NLL t-tests (Tables 5.22-5.24) |
| 12 | `12_exp2c_calibration_ece.ipynb` | Exp 2C | Expected Calibration Error analysis |
| 13 | `13_exp3_xai_gradcam.ipynb` | Exp 3 | Grad-CAM explainability visualizations (Table 5.25) |
| 14 | `14_exp3_hitl_neuroradiology.ipynb` | Exp 3 | HITL evaluation forms + neuroradiologist review (Tables 5.26-5.30) |

## Repository Structure

```
thesis-brain-tumor-stylegan2/
|-- Thesis_Junwei.pdf                  # Full thesis document (131 pages)
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- .gitattributes                     # Git LFS tracking
|
|-- notebooks/                         # 15 experiment notebooks (see table above)
|
|-- src/                               # Standalone Python modules
|   |-- dataloader.py                  # Custom DataLoader for K-Fold training
|   |-- dataset.py                     # Dataset class definitions
|   |-- preprocesamiento.py            # Image preprocessing pipeline
|
|-- stylegan2-ada-pytorch/             # Modified StyleGAN2-ADA library (NVIDIA)
|   |-- SG2_ADA_PyTorch.ipynb          # Training notebook
|   |-- train.py                       # Training script
|   |-- generate.py                    # Generation script
|   |-- training/                      # Training loop, dataset, augmentation
|   |-- metrics/                       # FID, IS, KID, PPL metrics
|   +-- ...
|
|-- encoder4editing/                   # Modified E4E encoder library
|   |-- inference_playground.ipynb     # Inference demo
|   |-- scripts/train.py              # Encoder training
|   +-- ...
|
|-- models/                            # Trained models (Git LFS)
|   |-- stylegan2/
|   |   |-- network-snapshot-000900.pkl    # StyleGAN-C complete dataset (352MB)
|   |   +-- network-snapshot-000600.pkl    # StyleGAN-C reduced dataset (352MB)
|   |-- stylegan_brain_rosinality.pt       # Converted model (117MB)
|   |-- encoder_e4e_best_model.pt          # E4E encoder (915MB)
|   |-- cnn_resnet_best.pt                 # CNN ResNet-50 features (91MB)
|   |-- encoder_sup_best.pt                # Supervised encoder (44MB)
|   +-- cnn_checkpoints/                   # K-Fold best checkpoints
|       |-- ensemble_realonly_final.ckpt
|       |-- inge1.3_final.ckpt
|       |-- inge1.1_final.ckpt
|       |-- ingereal_final.ckpt
|       +-- ensemble_1.45fold/fold_01-05_best.ckpt
|
|-- data/
|   |-- samples/                       # Example images (5 per class)
|   |   |-- real/{glioma,meningioma,pituitary}/
|   |   +-- synthetic/{glioma,meningioma,pituitary}/
|   |-- splits/                        # Dataset splits (Git LFS)
|   |   |-- by_patient_30test1/        # Patient-stratified 70/30 split
|   |   |-- random_split_clean_test/   # Random split (for leak comparison)
|   |   |-- datasetclean.zip           # Clean dataset (99MB)
|   |   +-- datasetcleanmask.zip       # Dataset with masks (101MB)
|   |-- synth_batches/                 # Generated synthetic images batch 1 (303MB)
|   |-- synth_batches1/                # Generated synthetic images batch 2 (264MB)
|   |-- features/                      # Extracted feature embeddings
|   |   |-- features_cnn_resnet50_*.npz
|   |   +-- w_features/               # W-space features (88MB)
|   |-- counterfactual_pairs/          # 60 counterfactual .npz pairs
|   |-- prototypes/                    # Class prototype means
|   |-- projections_batch_all/         # Batch projections for inference
|   +-- Testingset/                    # Held-out test images
|
+-- results/
    |-- exp1_generation/               # StyleGAN training logs + sample images
    |-- exp2a_counterfactuals/         # Counterfactual MP4 videos + LPIPS metrics
    |-- exp2b_latent_inversion/        # XAI results (Grad-CAM, occlusion)
    |-- exp2c_augmentation/            # CNN prediction CSVs, ECE metrics, precision plots
    +-- exp3_hitl/                     # Neuroradiology forms, Grad-CAM overlays
```

## Full Repository (Hugging Face)

> **Note:** The Hugging Face dataset is currently being uploaded and is not yet complete. This section will be updated once the upload finishes.

The complete repository including trained models, dataset splits, synthetic images, and feature embeddings will be available on Hugging Face:

**[huggingface.co/datasets/JunTierSS/thesis-brain-tumor-stylegan2](https://huggingface.co/datasets/JunTierSS/thesis-brain-tumor-stylegan2)**

The Hugging Face version (~7.4GB) will include everything in this GitHub repo plus:
- StyleGAN2-ADA trained models (`.pkl`, 352MB each)
- E4E encoder, CNN ResNet-50, and supervised encoder checkpoints (`.pt`)
- K-Fold CNN checkpoints (`.ckpt`, 9 files)
- Dataset splits (`.zip`), synthetic image batches, feature embeddings (`.npz`)
- Training run checkpoints from ablation experiments

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (11GB+ VRAM recommended)

### Quick Start (GitHub only)

This GitHub repository is self-contained with all code, notebooks, experiment results, and sample data. You can explore the thesis work without downloading the full dataset.

```bash
git clone https://github.com/JunTierSS/thesis-brain-tumor-stylegan2.git
cd thesis-brain-tumor-stylegan2
pip install -r requirements.txt
```

### Full Setup (with models and data)

To reproduce experiments or run the notebooks, download the complete repository from Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli download JunTierSS/thesis-brain-tumor-stylegan2 --repo-type dataset --local-dir .
```

### Running on Google Colab

Most notebooks were developed on Google Colab. To run them:

1. Upload the notebook to Colab
2. Mount your Google Drive and update paths accordingly
3. The notebooks reference paths like `/content/drive/MyDrive/TESIS/...` which correspond to this repository's directory structure

## Dataset

The original brain MRI dataset is based on a public brain tumor dataset with images organized by patient ID into three classes:
- **Glioma** - Glial cell tumors
- **Meningioma** - Meningeal tumors
- **Pituitary** - Pituitary gland tumors

Data is split by patient ID (not by image) to prevent data leakage across train/test sets.

## Methodology

The framework consists of three main stages:

1. **Synthetic Data Generation (Exp 1):** A conditional StyleGAN2-ADA (StyleGAN-C) model is trained on brain MRI images to generate class-conditional synthetic slices for glioma, meningioma, and pituitary tumor types. Image quality is evaluated using FID, PSNR, SSIM, and LPIPS metrics.

2. **Generative Model Evaluation (Exp 2A-2B):**
   - **Inference by Generation (Exp 2A):** Counterfactual trajectories are generated by interpolating between tumor class prototypes in the W latent space, and patient-level classification is performed using LPIPS distances.
   - **Latent Space Analysis (Exp 2B):** An E4E encoder and a supervised encoder are trained for MRI-to-latent inversion. SVM classifiers are trained on CNN features, encoder features, and W-space embeddings. PCA is used to visualize class separability.

3. **Classification and Evaluation (Exp 2C, Exp 3):**
   - **Synthetic Augmentation (Exp 2C):** A ResNet-50 classifier is trained with K-Fold cross-validation on real data (baseline) and on real + synthetic data (augmented). Three ablation studies are conducted: Classification Accuracy Score (CAS) on synthetic-only data, oversampling baseline, and augmentation strength comparison. Statistical significance is assessed with McNemar and NLL t-tests, and model calibration is measured with Expected Calibration Error (ECE).
   - **Explainability and Clinical Validation (Exp 3):** Grad-CAM heatmaps provide visual explanations of classifier decisions. A Human-in-the-Loop (HITL) study with neuroradiologists evaluates the clinical plausibility of synthetic images and the usefulness of XAI overlays.

## Key Results

- StyleGAN-C generates visually convincing class-conditional MRI images (FID ~46 at psi=1.0)
- Synthetic augmentation improves CNN ensemble calibration (lower NLL) particularly for meningioma
- Supervised encoder features achieve 0.947 accuracy with SVM classification
- Neuroradiologist HITL evaluation validates clinical plausibility of generated images
- Grad-CAM XAI provides interpretable decision explanations

## Citation

*Citation details coming soon.*

## Acknowledgments

- **Brain Tumor MRI Dataset** by [Jun Cheng (Figshare)](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) - Original brain tumor MRI dataset with glioma, meningioma, and pituitary classes
- **[StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch)** by Karras et al. (NVIDIA, NeurIPS 2020) - Training Generative Adversarial Networks with Limited Data
- **[Encoder4Editing](https://github.com/omertov/encoder4editing)** by Tov et al. (SIGGRAPH 2021) - Designing an Encoder for StyleGAN Image Manipulation
