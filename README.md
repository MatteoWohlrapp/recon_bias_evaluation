# Medical Image Reconstruction Bias Evaluation

This repository contains code for evaluating bias in medical image reconstruction models, specifically focusing on the UCSF-PDGM and CheXpert datasets. The project implements a complete pipeline for processing medical images, running predictions with various models, and evaluating potential biases in the reconstruction results.

## Repository Structure

The repository is organized into four main components:

### 1. Processing (`processing/`)
- Image preprocessing and preparation for model input
- Data loading and transformation utilities
- Configuration options for different processing pipelines

### 2. Prediction (`prediction/`)
- Model prediction implementations for different architectures:
  - U-Net
  - GAN
  - SDE (Stochastic Differential Equations)
- Dataset specific prediction depending on the input configuration

### 3. Visualization (`visualization/`)
- Tools for visualizing reconstruction results with segmentation masks and GradCAM
- Dataset visualization utilities

### 4. Evaluation (`evaluation/`)
- Comprehensive evaluation metrics and analysis
- Dataset-specific evaluation scripts:
  - `evaluate_ucsf.py` for UCSF-PDGM
  - `evaluate_chex.py` for CheXpert
  - `evaluate_mitigation.py` for bias mitigation analysis
  - `evaluate_combined.py` for combining results from multiple datasets
- Plotting utilities for results visualization
- LaTeX table generation for results presentation

## Pipeline Overview

The repository implements a complete pipeline for bias evaluation in medical image reconstruction:

1. **Processing**: Batch process medical images through reconstruction models (UNet, GAN, or SDE) and save the results to disk. This step is necessary because some models (particularly SDE) are computationally expensive and slow to run repeatedly.

2. **Prediction**: Load processed (reconstructed) images and run them through downstream task models (classification, segmentation) to evaluate performance.

3. **Evaluation**: Analyze prediction results to identify potential biases across demographic groups or other factors.

4. **Visualization**: Generate visual comparisons, heatmaps, and other visualizations to help interpret the results.

## Usage

### Processing Images

The processing step runs images through reconstruction models and saves the results for later use:

```bash
python processing/process_images.py --opt=[path to options YAML]
```

#### Key Parameters in `process_images.py` Options YAML:

```yaml
name: [name_of_processing_job]
results_path: [path_to_save_processed_images]
batch_size: 32
num_workers: 8

paths:
  recon_bias: [path_to_recon_bias_repo]
  pix2pix: [path_to_pix2pix_repo]
  sde: [path_to_sde_repo]

datasets:
  name: [chex|ucsf]  # Dataset name
  # UCSF specific parameters:
  dataroot: [path_to_ucsf_dataset]
  sampling_mask: [radial|linear]
  seed: [random_seed]
  type: [FLAIR|other_MRI_type]
  pathology: []  # List of pathologies to include
  lower_slice: 60
  upper_slice: 130
  split: [train|test|val]
  num_rays: [number_of_rays]
  
  # CheXpert specific parameters:
  csv_path_A: [path_to_noisy_metadata]
  csv_path_B: [path_to_clean_metadata]
  dataroot_A: [path_to_noisy_images]
  dataroot_B: [path_to_clean_images]
  split: [train|test|val]

models:
  network: [UNet|GAN|Diffusion]
  model_path: [path_to_pretrained_model]
```

### Running Predictions

After processing, evaluate the reconstruction quality by running downstream tasks:

```bash
python prediction/prediction.py --opt=[path to options YAML]
```

#### Key Parameters in `prediction.py` Options YAML:

```yaml
name: [name_of_prediction_job]
results_path: [path_to_save_prediction_results]
batch_size: 16
num_workers: 4

datasets:
  name: [chex|ucsf]
  # Dataset-specific paths
  # For preprocessed images:
  processed_image_root: [path_to_processed_images]
  
  # Other dataset parameters similar to processing

models:
  network: [UNet|GAN|Diffusion]
  model_path: [path_to_model] # Only for UNet/GAN

task_models:
  name: [ucsf|chexpert-classifier]
  # For UCSF:
  models:
    - type: [classifier|segmentation]
      name: [model_name]
      path: [path_to_model]
  # For CheXpert:
  path: [path_to_classifier_model]
```

### Visualizing Results

Generate visualizations to better understand the reconstruction results:

```bash
python visualization/visualize.py --opt=[path to options YAML]
```

#### Key Parameters in `visualize.py` Options YAML:

```yaml
name: [name_of_visualization_job]
results_path: [path_to_save_visualizations]

target_column: [metric_to_visualize]  # e.g., "UNet_recon_dice" or "Lung Lesion_recon"

csv_files:
  - [path_to_prediction_results_1]
  - [path_to_prediction_results_2]
  - [path_to_prediction_results_3]

datasets:
  name: [chex|ucsf]
  # Dataset-specific parameters as in processing/prediction

models:
  - network: [UNet|GAN|Diffusion]
    model_path: [path_to_model]  # For UNet/GAN
    processed_image_root: [path_to_processed_images]  # For Diffusion

task_model:
  name: [segmentation|chexpert-classifier]
  path: [path_to_model]
```

### Evaluating Results

Run comprehensive evaluations and generate analysis reports using the evaluation module:

```bash
python evaluation/evaluation.py --opt=[path to options YAML]
```

#### Key Parameters in Evaluation Options YAML:

```yaml
name: [name_of_evaluation_job]
results_path: [path_to_save_evaluation_results]

# The mode determines which type of evaluation to run
mode: [ucsf|chex|mitigation|combined]

predictions:
  - acceleration: [acceleration_factor]  # e.g., 4, 8, 16
    model: [unet|pix2pix|sde]
    predictions: [path_to_prediction_csv]
    fairness: [True|False]  # Whether to do fairness evaluation
```

The `mode` parameter determines which type of evaluation to run:

1. **`mode: ucsf`** - Evaluates UCSF-PDGM dataset results
2. **`mode: chex`** - Evaluates CheXpert dataset results
3. **`mode: mitigation`** - Evaluates bias mitigation techniques
4. **`mode: combined`** - Combines and compares results across datasets and techniques

For batch processing of multiple evaluation configurations, you can use:

```bash
python evaluation/evaluation.py --opt=[path_to_template]
```

The combined evaluation capability allows for comprehensive analysis that:
- Compares performance across different datasets
- Evaluates the effectiveness of various bias mitigation strategies
- Generates reports that show trade-offs between performance and fairness
- Identifies common patterns of bias across different medical imaging domains

## Datasets

### UCSF-PDGM Dataset
The UCSF-PDGM (University of California San Francisco - Preoperative Determination of Glioma Molecular) dataset consists of MRI brain scans with annotations for tumor segmentation and molecular classification. The dataset includes demographic information, which allows for analysis of algorithmic bias.

### CheXpert Dataset
CheXpert is a large public dataset for chest X-ray interpretation, consisting of 224,316 chest radiographs of 65,240 patients. The dataset includes labels for 14 observations (e.g., Pleural Effusion, Pneumonia) and can be used to evaluate how reconstruction affects downstream classification tasks.

## Models

### Reconstruction Models
- **UNet**: A convolutional neural network with skip connections for image-to-image translation
- **GAN (Pix2Pix)**: A conditional adversarial network for image-to-image translation
- **SDE (Stochastic Differential Equations)**: A diffusion-based model for image restoration

### Downstream Task Models
- **Classification**: Models for detecting pathologies in CheXpert and tumor classification in UCSF-PDGM
- **Segmentation**: Models for tumor segmentation in UCSF-PDGM

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- NumPy
- scikit-image
- pandas
- polars
- matplotlib
- segmentation-models-pytorch
- tqdm
- pyyaml