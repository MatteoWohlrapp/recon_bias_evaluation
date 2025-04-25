import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import os
import torch
from PIL import Image
from datasets import create_dataset
from unet.reconstruction_model import ReconstructionModel
from unet.unet_chex import UNetChex
from unet.unet_ucsf import UNetUCSF
from gan.GAN import UnetGenerator
from segmentation.segmentation_model import SegmentationModel
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Reuse these utility functions from original script
from visualize import (
    setup_logger, save_image, save_image_with_gradcam,
    compute_gradcam, min_max_slice_normalization,
    create_segmentation_overlay, load_task_model
)

def get_processed_image(x, model_config, dataset_name, device):
    """Process image for diffusion models"""
    if dataset_name == "chex":
        # Initialize a list to store processed images
        processed_images = []
        
        # Ensure we're working with the full path
        image_paths = x["image_path"] if isinstance(x["image_path"], list) else [x["image_path"]]
        
        # Process each image in the batch
        for image_path in image_paths:
            full_path = os.path.join(
                model_config["processed_image_root"],
                str(image_path).replace(".jpg", ".npy"),  # Ensure path is string
            )                        
            # load and process image
            image = np.load(full_path)
            image = min_max_slice_normalization(image)
            processed_images.append(image)

        # Stack all processed images into a batch and convert to torch tensor
        output = torch.from_numpy(np.stack(processed_images, axis=0)).float()
        output = output.unsqueeze(1)
        output = output.to(device)
        return output
    elif dataset_name == "ucsf":
        processed_images = []
        patient_ids, slice_ids = (
            x["patient_id"],
            x["slice_id"],
        )
        if isinstance(patient_ids, int) or isinstance(slice_ids, int):
            patient_ids = [patient_ids]
            slice_ids = [slice_ids]

        for patient_id, slice_id in zip(patient_ids, slice_ids):
            slice_id = (
                slice_id.item()
                if isinstance(slice_id, torch.Tensor)
                else slice_id
            )
            # make slice id 3 digits
            slice_id = f"{slice_id:03d}"
            full_path = os.path.join(
                model_config["processed_image_root"],
                f"{patient_id}_{slice_id}.npy",
            )
            # load and process image
            image = np.load(full_path)
            image = min_max_slice_normalization(image)
            processed_images.append(image)
        output = torch.from_numpy(np.stack(processed_images, axis=0)).float()
        output = output.unsqueeze(1)
        output = output.to(device)
        return output
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def load_single_model(model_config, dataset_name, device):
    """Helper function to load a single model"""
    if model_config['network'] == "UNet":
        model = ReconstructionModel()
        if dataset_name == "chex":
            network = UNetChex()
        elif dataset_name == "ucsf":
            network = UNetUCSF()
        model.set_network(network)
        model.load_state_dict(torch.load(model_config['model_path'], map_location=device))
        model.network.eval()
        model.to(device)
        return lambda x: model(x["image_A"].unsqueeze(0).to(device))
    elif model_config['network'] == "GAN":
        model = UnetGenerator(input_nc=1, output_nc=1, num_downs=7)
        model.load_state_dict(torch.load(model_config['model_path'], map_location=device))
        model.eval()
        model.to(device)
        return lambda x: model(x["image_A"].unsqueeze(0).to(device))
    elif model_config['network'] == "Diffusion":
        return lambda x: get_processed_image(x, model_config, dataset_name, device)

def load_reconstruction_models(config, device):
    """Load all reconstruction models (standard + mitigations)"""
    models = {
        "standard": {},
        "mitigations": {
            "UNet": [],
            "GAN": [],
            "Diffusion": []
        }
    }
    
    dataset_name = config['datasets']['name']
    
    # Load standard models
    for model_config in config['models']['standard']:
        model_type = model_config['network']
        models["standard"][model_type] = load_single_model(model_config, dataset_name, device)
    
    # Load mitigation models
    for model_type, mitigation_configs in config['models']['mitigations'].items():
        for mitigation_config in mitigation_configs:
            model = load_single_model(mitigation_config, dataset_name, device)
            models["mitigations"][model_type].append(model)
    
    return models

def find_interesting_mitigation_cases(config, top_k=5):
    """Find interesting cases based on dataset type:
    - UCSF: First sort by UNet_recon_sum, then find highest variations
    - CheXpert: Directly find highest variations"""
    target_column = config['target_column']
    all_differences = []
    model_results_by_row = {}
    dataset_name = config['datasets']['name']
    
    # For UCSF, first get indices sorted by UNet_recon_sum
    if dataset_name == "ucsf":
        unet_standard_csv = config['standard_csvs']['UNet']
        if not os.path.exists(unet_standard_csv):
            raise FileNotFoundError(f"Standard CSV file not found: {unet_standard_csv}")
        
        unet_standard_df = pd.read_csv(unet_standard_csv)
        if 'UNet_recon_sum' not in unet_standard_df.columns:
            raise ValueError("UNet_recon_sum column not found in standard UNet CSV")
        
        # Sort by UNet_recon_sum in descending order and take top 100
        sorted_indices = np.argsort(unet_standard_df['UNet_recon_sum'].values)[::-1]
        indices_to_check = sorted_indices[:100]  # Take top 100 indices
    else:
        # For CheXpert, use all indices
        standard_csv = config['standard_csvs']['UNet']  # Use any CSV to get total length
        standard_df = pd.read_csv(standard_csv)
        indices_to_check = np.arange(len(standard_df))
    
    # Process each model type
    for model_type in ['UNet', 'GAN', 'Diffusion']:
        # Load standard CSV
        standard_csv = config['standard_csvs'][model_type]
        if not os.path.exists(standard_csv):
            raise FileNotFoundError(f"Standard CSV file not found: {standard_csv}")
        
        standard_df = pd.read_csv(standard_csv)
        
        # Load mitigation CSVs
        mitigation_csvs = config['mitigation_csvs'][model_type]
        mitigation_dfs = []
        for csv_path in mitigation_csvs:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Mitigation CSV file not found: {csv_path}")
            mitigation_dfs.append(pd.read_csv(csv_path))
        
        # Get target column
        target_col = config.get('target_columns', {}).get(model_type, target_column)
        
        # Check if target column exists in all dataframes
        if target_col not in standard_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in standard CSV for {model_type}")
        
        for i, df in enumerate(mitigation_dfs):
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in mitigation {i+1} CSV for {model_type}")
        
        # Calculate mitigation values and differences for selected indices
        standard_values = standard_df[target_col].values[indices_to_check]
        mitigation_values = np.stack([df[target_col].values[indices_to_check] for df in mitigation_dfs])
        avg_mitigation = np.nanmean(mitigation_values, axis=0)
        differences = np.abs(standard_values - avg_mitigation)
        
        # Store differences for these indices
        all_differences.append(differences)
        
        # Store detailed results for selected indices
        for i, idx in enumerate(indices_to_check):
            if idx not in model_results_by_row:
                model_results_by_row[idx] = {}
            
            # Get standard value and PSNR
            standard_value = standard_df[target_col].iloc[idx]
            standard_psnr = standard_df["psnr"].iloc[idx] if "psnr" in standard_df.columns else np.nan
            
            # Get mitigation values and PSNRs
            mitigation_values = []
            mitigation_psnrs = []
            for df in mitigation_dfs:
                mitigation_values.append(df[target_col].iloc[idx])
                if "psnr" in df.columns:
                    mitigation_psnrs.append(df["psnr"].iloc[idx])
                else:
                    mitigation_psnrs.append(np.nan)
            
            model_results_by_row[idx][model_type] = {
                "standard_value": standard_value,
                "standard_psnr": standard_psnr,
                "mitigation_values": mitigation_values,
                "mitigation_psnrs": mitigation_psnrs,
                "difference": differences[i]
            }
    
    # Calculate average difference across models
    avg_differences = np.nanmean(np.stack(all_differences), axis=0)
    
    # Get top k indices with highest variations
    valid_indices = ~np.isnan(avg_differences)
    valid_differences = avg_differences[valid_indices]
    top_k_positions = np.argsort(valid_differences)[-top_k:][::-1]
    
    # Map back to original indices
    final_indices = indices_to_check[valid_indices][top_k_positions]
    
    # Create results
    results = []
    for idx in final_indices:
        result = {
            "row_idx": idx,
            "avg_difference": avg_differences[np.where(indices_to_check == idx)[0][0]],
            "unet_recon_sum": float(unet_standard_df['UNet_recon_sum'].iloc[idx]) if dataset_name == "ucsf" else 0.0,
            "model_results": model_results_by_row[idx]
        }
        results.append(result)
    
    return results

def save_mitigation_analysis(case_dir, result, case_num):
    """Save analysis details to text file"""
    with open(case_dir / "analysis.txt", "w") as f:
        f.write(f"Analysis for Case {case_num}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Row Index: {result['row_idx']}\n")
        f.write(f"Average Difference Across Methods: {result['avg_difference']:.4f}\n\n")
        
        for model_type in ['UNet', 'GAN', 'Diffusion']:
            f.write(f"\n{model_type} Analysis:\n")
            f.write("-" * 20 + "\n")
            method_data = result['model_results'][model_type]
            f.write(f"Standard {model_type} Value: {method_data['standard_value']:.4f}\n")
            f.write(f"Standard {model_type} PSNR: {method_data['standard_psnr']:.4f}\n\n")
            
            for i, (val, psnr) in enumerate(zip(method_data['mitigation_values'], 
                                              method_data['mitigation_psnrs']), 1):
                f.write(f"Mitigation {i} {model_type} Value: {val:.4f}\n")
                f.write(f"Mitigation {i} {model_type} PSNR: {psnr:.4f}\n\n")
            
            f.write(f"Average Difference: {method_data['difference']:.4f}\n")

def save_mitigation_visualizations(dataset_name, row_idx, image_data, models, task_model, case_dir, result):
    """Save visualizations for all reconstructions"""
    # Save original and noisy images at the top level
    save_image(image_data["image_B"], case_dir / f"{row_idx}_original.png")
    save_image(image_data["image_A"], case_dir / f"{row_idx}_noisy.png")
    
    # Save task model's prediction on original image (for both datasets)
    logits, attention = compute_gradcam(task_model, image_data["image_B"].unsqueeze(0))
    if dataset_name == "ucsf":
        # For UCSF: Save segmentation mask and overlay
        save_image(attention, case_dir / "original_segmentation.png")
        orig_overlay = create_segmentation_overlay(image_data["image_B"], attention)
        save_image(orig_overlay, case_dir / "original_segmentation_overlay.png")
        
        # Also save ground truth if available
        if "segmentation_slice" in image_data:
            gt_mask = image_data["segmentation_slice"]
            save_image(gt_mask, case_dir / "ground_truth.png")
            gt_overlay = create_segmentation_overlay(image_data["image_B"], gt_mask)
            save_image(gt_overlay, case_dir / "ground_truth_overlay.png")
    else:
        # For CheXpert: Save GradCAM visualization
        save_image_with_gradcam(image_data["image_B"], attention, case_dir / "original_gradcam.png")
    
    # For each reconstruction method
    for model_type in ['UNet', 'GAN', 'Diffusion']:
        method_dir = case_dir / model_type
        method_dir.mkdir(exist_ok=True)
        
        # Get stored prediction and PSNR values
        method_data = result['model_results'][model_type]
        standard_value = method_data['standard_value']
        standard_psnr = method_data['standard_psnr']
        mitigation_values = method_data['mitigation_values']
        mitigation_psnrs = method_data['mitigation_psnrs']
        
        # Save standard reconstruction
        standard_model = models["standard"][model_type]
        standard_recon = standard_model(image_data)
        save_image(standard_recon, method_dir / "standard_reconstruction.png")
        
        # Save metrics to file for standard model
        with open(method_dir / "standard_metrics.txt", "w") as f:
            f.write(f"Prediction Value: {standard_value:.4f}\n")
            f.write(f"PSNR: {standard_psnr:.4f}\n")
        
        # Process standard with task model
        if dataset_name == "ucsf":  # Segmentation for UCSF
            _, prediction = compute_gradcam(task_model, standard_recon)
            save_image(prediction, method_dir / "standard_segmentation.png")
            overlay = create_segmentation_overlay(standard_recon, prediction)
            save_image(overlay, method_dir / "standard_segmentation_overlay.png")
        else:  # GradCAM for CheXpert
            _, gradcam = compute_gradcam(task_model, standard_recon)
            save_image_with_gradcam(standard_recon, gradcam, method_dir / "standard_gradcam.png")
        
        # Save mitigation reconstructions
        for i, mitigation_model in enumerate(models["mitigations"][model_type], 1):
            mitigation_recon = mitigation_model(image_data)
            mitigation_dir = method_dir / f"mitigation{i}"
            mitigation_dir.mkdir(exist_ok=True)
            
            save_image(mitigation_recon, mitigation_dir / "reconstruction.png")
            
            # Save metrics to file for mitigation model
            with open(mitigation_dir / "metrics.txt", "w") as f:
                f.write(f"Prediction Value: {mitigation_values[i-1]:.4f}\n")
                f.write(f"PSNR: {mitigation_psnrs[i-1]:.4f}\n")
            
            # Process with task model
            if dataset_name == "ucsf":  # Segmentation for UCSF
                _, prediction = compute_gradcam(task_model, mitigation_recon)
                save_image(prediction, mitigation_dir / "segmentation.png")
                overlay = create_segmentation_overlay(mitigation_recon, prediction)
                save_image(overlay, mitigation_dir / "segmentation_overlay.png")
            else:  # GradCAM for CheXpert
                _, gradcam = compute_gradcam(task_model, mitigation_recon)
                save_image_with_gradcam(mitigation_recon, gradcam, mitigation_dir / "gradcam.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, required=True, help="Path to options YAML file.")
    args = parser.parse_args()

    # Load config
    with open(args.opt, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["results_path"]) / f"{config['name']}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("mitigation_visualization", results_dir)
    logger.info(f"Config:\n{yaml.dump(config, default_flow_style=False)}")
    
    # Load models
    models = load_reconstruction_models(config, device)
    task_model = load_task_model(config["task_model"], device)
    dataset = create_dataset(config["datasets"])
    
    # Find interesting cases
    cases = find_interesting_mitigation_cases(config, top_k=5)
    
    # Process each case
    for i, case in enumerate(cases, 1):
        case_dir = results_dir / f"case_{i}"
        case_dir.mkdir(exist_ok=True)
        
        # Save analysis text
        save_mitigation_analysis(case_dir, case, i)
        
        # Load image data
        row_idx = case['row_idx']
        image_data = dataset[row_idx]
        
        # Save visualizations
        save_mitigation_visualizations(
            config['datasets']['name'],
            row_idx, 
            image_data, 
            models, 
            task_model, 
            case_dir,
            case  # Pass the case data that contains PSNR and prediction values
        )
        
        # Log results
        logger.info(f"\nProcessed case {i}:")
        logger.info(f"Row index: {row_idx}")
        logger.info(f"Average difference: {case['avg_difference']:.4f}")
        for model_type in ['UNet', 'GAN', 'Diffusion']:
            model_results = case["model_results"][model_type]
            logger.info(f"{model_type} standard value: {model_results['standard_value']:.4f}")
            logger.info(f"{model_type} standard PSNR: {model_results['standard_psnr']:.4f}")
            
            mitigation_values = [f'{v:.4f}' for v in model_results['mitigation_values']]
            mitigation_psnrs = [f'{p:.4f}' for p in model_results['mitigation_psnrs']]
            
            logger.info(f"{model_type} mitigation values: {mitigation_values}")
            logger.info(f"{model_type} mitigation PSNRs: {mitigation_psnrs}")
            logger.info(f"{model_type} difference: {model_results['difference']:.4f}")

if __name__ == "__main__":
    main()
