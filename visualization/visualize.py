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

def setup_logger(name, save_dir, filename="log.txt"):
    # ... same logger setup as in evaluation.py ...
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(os.path.join(save_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def load_task_model(config, device):
    if config["name"] == "chexpert-classifier":
        model = torch.load(config["path"], map_location=device)
        model.to(device)
        return model
    elif config["name"] == "segmentation":

        model_path = config["path"]
        model = SegmentationModel()
        model = model.to(device)

        network = smp.Unet(
            in_channels=1,
            classes=1,
            encoder_name="resnet34",
            encoder_depth=5,
            encoder_weights=None,
            decoder_channels=(256, 128, 64, 32, 16),
            activation="sigmoid",
        )

        network = network.to(device)
        model.set_network(network)
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        model.network.eval()
        return model
    else:
        raise ValueError(f"Task model {config['name']} not found")

def load_reconstruction_models(config, device):
    reconstruction_models = {}
    model_configs = config["models"]
    for model_config in model_configs:
        if model_config["network"] == "UNet":

            model = ReconstructionModel()
            if config["datasets"]["name"] == "chex":
                print("Loading UNetChex")
                network = UNetChex()
            elif config["datasets"]["name"] == "ucsf":
                print("Loading UNetUCSF")
                network = UNetUCSF()
            model.set_network(network)
            model.load_state_dict(
                torch.load(model_config["model_path"], map_location=device)
            )
            model.network.eval()
            model.to(device)
            reconstruction_models[model_config["network"]] = lambda x: model(x["image_A"].unsqueeze(0).to(device))

        elif model_config["network"] == "GAN":

            model = UnetGenerator(input_nc=1, output_nc=1, num_downs=7)
            model.load_state_dict(
                torch.load(model_config["model_path"], map_location=device)
            )
            model.eval()
            model.to(device)
            reconstruction_models[model_config["network"]] = lambda x: model(x["image_A"].unsqueeze(0).to(device))

        elif model_config["network"] == "Diffusion":

            def get_processed_image(x):
                if config["datasets"]["name"] == "chex":
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
                elif config["datasets"]["name"] == "ucsf":
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
                    raise ValueError(f"Dataset {config['datasets']['name']} not supported")

            reconstruction_models[model_config["network"]] = lambda x: get_processed_image(x)

    return reconstruction_models

def find_interesting_patterns(csv_files, target_column, top_k=5):
    """
    Find patterns in the CSV files based on specified criteria
    
    Args:
        csv_files (list): List of paths to CSV files
        target_column (str): Column name to analyze for discrepancy
        top_k (int): Number of top cases to return
    
    Returns:
        tuple: (min_target_max_psnr_rows, min_psnr_max_target_rows)
    """
    # Read all CSVs into a list of dataframes
    dfs = [pd.read_csv(f) for f in csv_files]
    
    # Verify that all CSVs have the same number of rows and are sorted the same way
    base_len = len(dfs[0])
    for df in dfs[1:]:
        if len(df) != base_len:
            raise ValueError("All CSV files must have the same number of rows")
    
    results = []
    for row_idx in range(base_len):
        target_values = [df.loc[row_idx, target_column] for df in dfs]
        
        # Skip rows where any target value is NaN
        if any(pd.isna(val) or val == 0 for val in target_values):
            continue
            
        row_values = {
            'target_values': target_values,
            'psnr_values': [df.loc[row_idx, 'psnr'] for df in dfs],
            'row_idx': row_idx
        }
        results.append(row_values)
    
    # Calculate metrics for each row
    for r in results:
        r['target_std'] = np.std(r['target_values'])
        r['psnr_std'] = np.std(r['psnr_values'])
        r['target_range'] = max(r['target_values']) - min(r['target_values'])
        r['psnr_range'] = max(r['psnr_values']) - min(r['psnr_values'])
    
    # Find top-k rows with min target discrepancy but max PSNR discrepancy
    min_target_max_psnr = sorted(
        results,
        key=lambda x: (x['target_std'], -x['psnr_std'])
    )[:top_k]
    
    # Find top-k rows with min PSNR discrepancy but max target discrepancy
    min_psnr_max_target = sorted(
        results,
        key=lambda x: (x['psnr_std'], -x['target_std'])
    )[:top_k]
    
    return min_target_max_psnr, min_psnr_max_target

def save_colorbar(save_path, title="Intensity"):
    """
    Save a standalone colorbar using the jet colormap
    
    Args:
        save_path (Path): Path to save the colorbar
        title (str): Title for the colorbar
    """
    plt.figure(figsize=(8, 2))
    
    # Create axes for the colorbar
    ax = plt.gca()
    
    # Create a dummy scalar mappable using the jet colormap
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    sm.set_array([])
    
    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label(title, fontsize=12)
    
    # Hide the main axes
    ax.set_visible(False)
    
    # Save and close
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def save_difference_with_colormap(image1, image2, save_path, alpha=1.0):
    """
    Save difference between two images using a colored heatmap
    
    Args:
        image1 (torch.Tensor): First image tensor
        image2 (torch.Tensor): Second image tensor
        save_path (Path): Path to save the image
        alpha (float): Transparency of the colormap (0-1)
    """
    # Convert tensors to numpy arrays and ensure proper dimensions
    if image1.dim() == 4:
        image1 = image1.squeeze(0)
    if image1.dim() == 3:
        image1 = image1.squeeze(0)
    
    if image2.dim() == 4:
        image2 = image2.squeeze(0)
    if image2.dim() == 3:
        image2 = image2.squeeze(0)
    
    image1 = image1.cpu().detach().numpy()
    image2 = image2.cpu().detach().numpy()
    
    # Compute absolute difference
    difference = np.abs(image1 - image2)
    
    # Normalize difference to [0, 1]
    difference = (difference - difference.min()) / (difference.max() - difference.min() + 1e-8)
    
    # Create figure and axis
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    
    # Plot the difference using jet colormap
    plt.imshow(difference, cmap='jet', alpha=alpha)
    
    # Save and close
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_segmentation_overlay(image, mask, color=[1, 0, 0], alpha=0.3):
    """
    Create an overlay of the segmentation mask on the image
    
    Args:
        image (torch.Tensor): Original image (C, H, W)
        mask (torch.Tensor): Segmentation mask (1, H, W)
        color (list): RGB color for the overlay [R, G, B]
        alpha (float): Transparency of the overlay (0-1)
    """
    # Convert tensors to numpy arrays
    image = image.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    
    # Ensure proper dimensions
    if image.ndim == 4:
        image = image.squeeze(0)
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    
    # Normalize image to 0-1 if needed
    if image.max() > 1:
        image = image / 255.0
    
    # Create RGB image if grayscale
    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    
    # Ensure mask is 2D
    if mask.shape[0] == 1:
        mask = mask.squeeze(0)
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    for i, c in enumerate(color):
        colored_mask[i][mask > 0.5] = c
    
    # Blend images
    overlay = image * (1 - alpha) + colored_mask * alpha
    return torch.from_numpy(overlay)

def save_analysis_images(row_idx, image_A, image_B, reconstructed_imgs, classifier, case_dir, segmentation_slice=None):
    """Save analysis images including original, reconstructed, and segmentation/GradCAM"""
    device = next(classifier.parameters()).device
    
    # Save colorbars once
    save_colorbar(case_dir / "difference_colorbar.png", "Absolute Difference")
    if not isinstance(classifier, SegmentationModel):
        save_colorbar(case_dir / "gradcam_colorbar.png", "GradCAM Intensity")
    
    # Get original images
    original_img = image_B.to(device)
    reconstructed_img = image_A.to(device)
    
    # Save the original images
    save_image(original_img, case_dir / f"{row_idx}_original.png")
    save_image(reconstructed_img, case_dir / f"{row_idx}_reconstructed.png")
    
    # Handle ground truth segmentation if available
    if segmentation_slice is not None:
        gt_segmentation = segmentation_slice.to(device)
        save_image(gt_segmentation, case_dir / f"{row_idx}_gt_segmentation.png")
        # Create and save overlay for ground truth
        gt_overlay = create_segmentation_overlay(original_img, gt_segmentation)
        save_image(gt_overlay, case_dir / f"{row_idx}_gt_overlay.png")
    
    # Get prediction from classifier (either segmentation mask or GradCAM)
    if isinstance(classifier, SegmentationModel):
        _, prediction_mask = compute_gradcam(classifier, original_img)
        save_image(prediction_mask, case_dir / f"{row_idx}_prediction.png")
        # Create and save overlay for prediction
        pred_overlay = create_segmentation_overlay(original_img, prediction_mask)
        save_image(pred_overlay, case_dir / f"{row_idx}_prediction_overlay.png")
    else:
        _, gradcam = compute_gradcam(classifier, original_img)
        save_image_with_gradcam(original_img, gradcam, case_dir / f"{row_idx}_gradcam.png")
    
    # For each reconstruction
    for model_name, recon_img in reconstructed_imgs.items():
        model_dir = case_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save reconstructed image
        save_image(recon_img, model_dir / "reconstruction.png")
        
        # Save difference image with colormap
        save_difference_with_colormap(
            original_img.cpu(), 
            recon_img.cpu(), 
            model_dir / "difference.png"
        )
        
        # Get and save prediction for reconstructed image
        recon_img = recon_img.to(device)
        if isinstance(classifier, SegmentationModel):
            _, prediction = compute_gradcam(classifier, recon_img)
            save_image(prediction, model_dir / "reconstruction_prediction.png")
            # Create and save overlay for reconstruction prediction
            recon_overlay = create_segmentation_overlay(recon_img, prediction)
            save_image(recon_overlay, model_dir / "reconstruction_prediction_overlay.png")
        else:
            _, gradcam = compute_gradcam(classifier, recon_img)
            save_image_with_gradcam(recon_img, gradcam, model_dir / "reconstruction_gradcam.png")

def load_original_image(row_idx, dataset):
    """
    Load original image based on row index from dataset
    
    Args:
        row_idx (int): Index of the row in CSV
        config (dict): Configuration dictionary containing dataset info
    
    Returns:
        torch.Tensor: Loaded and preprocessed image
    """
    data = dataset[row_idx]
    return data

def compute_gradcam(model, input_image, target_layer=None):
    """
    Compute GradCAM visualization for classification models,
    or return segmentation mask for segmentation models
    
    Args:
        model: The model (classifier or segmentation)
        input_image (torch.Tensor): Input image tensor
        target_layer: Optional specific layer to use for GradCAM
    
    Returns:
        tuple: (original image tensor, gradcam/segmentation tensor)
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Add batch dimension if not present
    if input_image.dim() == 3:
        input_image = input_image.unsqueeze(0)
    
    original_image = input_image.clone()
    input_image = input_image.to(device)
    
    # For segmentation models, return the segmentation mask
    if isinstance(model, SegmentationModel):
        with torch.no_grad():
            output = model(input_image)
            #mask = torch.sigmoid(output)
            mask = torch.where(output > 0.5, 1, 0)
            return original_image.cpu(), mask.cpu()
    
    # For classification models, compute GradCAM
    if target_layer is None:
        for module in reversed(list(model.modules())):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                target_layer = module
                break
    
    activations = []
    gradients = []
    
    def save_activation(module, input, output):
        activations.append(output)
    
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    handle1 = target_layer.register_forward_hook(save_activation)
    handle2 = target_layer.register_full_backward_hook(save_gradient)
    
    output = model(input_image)
    if output.dim() > 1:
        score = output[:, 0]
    else:
        score = output
    
    model.zero_grad()
    score.backward()
    
    handle1.remove()
    handle2.remove()
    
    activation = activations[0]
    gradient = gradients[0]
    
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
    gradcam = torch.sum(weights * activation, dim=1, keepdim=True)
    gradcam = torch.relu(gradcam)
    
    gradcam = torch.nn.functional.interpolate(
        gradcam,
        size=input_image.shape[2:],
        mode='bilinear',
        align_corners=False
    )
    
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
    
    return original_image.cpu(), gradcam.cpu()

def save_image_with_gradcam(image, gradcam, save_path, alpha=0.5):
    """
    Save image with colored GradCAM overlay
    
    Args:
        image (torch.Tensor): Original image tensor
        gradcam (torch.Tensor): GradCAM tensor
        save_path (Path): Path to save the image
        alpha (float): Transparency of the overlay (0-1)
    """
    # Convert tensors to numpy arrays
    if image.dim() == 4:
        image = image.squeeze(0)
    if image.dim() == 3:
        image = image.squeeze(0)
    
    if gradcam.dim() == 4:
        gradcam = gradcam.squeeze(0)
    if gradcam.dim() == 3:
        gradcam = gradcam.squeeze(0)
    
    image = image.cpu().detach().numpy()
    gradcam = gradcam.cpu().detach().numpy()
    
    # Normalize image to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Create figure and axis
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    
    # Plot original image in grayscale
    plt.imshow(image, cmap='gray')
    
    # Apply jet colormap to GradCAM
    colored_gradcam = cm.jet(gradcam)[..., :3]
    
    # Overlay GradCAM on original image
    plt.imshow(colored_gradcam, alpha=alpha)
    
    # Save and close
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_image(image, save_path):
    """
    Save image tensor to disk
    
    Args:
        image (torch.Tensor): Image tensor to save
        save_path (Path): Path to save the image
    """
    # Ensure image is on CPU and detached from computation graph
    image = image.cpu().detach()
    
    # Remove batch dimension if present
    if image.dim() == 4:
        image = image.squeeze(0)
    
    # Handle different channel configurations
    if image.dim() == 3:
        # If it's an RGB image (3 channels)
        if image.size(0) == 3:
            # Move channels to the end for PIL
            image = image.permute(1, 2, 0)
        # If it's a single channel image
        elif image.size(0) == 1:
            image = image.squeeze(0)
    
    # Normalize to [0, 1] if not already
    if image.min() < 0 or image.max() > 1:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Convert to numpy and then to uint8
    image_np = (image.numpy() * 255).astype(np.uint8)
    
    # Save using PIL
    if len(image_np.shape) == 3 and image_np.shape[-1] == 3:  # RGB image
        Image.fromarray(image_np, mode='RGB').save(save_path)
    else:  # Grayscale image
        Image.fromarray(image_np, mode='L').save(save_path)

def min_max_slice_normalization(image):
    """
    Normalize image to [0, 1] range
    
    Args:
        image (numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Normalized image
    """
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val + 1e-8)

def save_analysis_text(case_dir, result, case_num, pattern_type):
    """
    Save analysis details in a text file
    
    Args:
        case_dir (Path): Directory for the current case
        result (dict): Dictionary containing analysis results
        case_num (int): Case number
        pattern_type (str): Type of pattern ('min_target_max_psnr' or 'min_psnr_max_target')
    """
    with open(case_dir / "analysis.txt", "w") as f:
        f.write(f"Analysis for Case {case_num}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Pattern Type: {pattern_type}\n")
        f.write(f"Row Index: {result['row_idx']}\n\n")
        
        f.write("Target Values:\n")
        f.write("-" * 20 + "\n")
        for i, val in enumerate(result['target_values'], 1):
            f.write(f"Model {i}: {val:.4f}\n")
        f.write(f"Target Std Dev: {result['target_std']:.4f}\n\n")
        
        f.write("PSNR Values:\n")
        f.write("-" * 20 + "\n")
        for i, val in enumerate(result['psnr_values'], 1):
            f.write(f"Model {i}: {val:.4f}\n")
        f.write(f"PSNR Std Dev: {result['psnr_std']:.4f}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt", type=str, required=True, help="Path to options YAML file."
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.opt, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["results_path"]) / f"{config['name']}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("visualization", results_dir)
    logger.info(f"Config:\n{yaml.dump(config, default_flow_style=False)}")

    # Load reconstruction models
    reconstruction_models = load_reconstruction_models(config, device)
    
    # Load classifier
    classifier = load_task_model(config["task_model"], device)

    dataset = create_dataset(config["datasets"])

    # Get list of CSV files from config
    csv_files = [Path(p) for p in config["csv_files"]]
    target_column = config["target_column"]

    # Find interesting patterns
    min_target_max_psnr, min_psnr_max_target = find_interesting_patterns(
        csv_files, target_column, top_k=5
    )

    # Create directories for each type of pattern
    min_target_dir = results_dir / "min_target_max_psnr"
    min_psnr_dir = results_dir / "min_psnr_max_target"
    min_target_dir.mkdir(exist_ok=True)
    min_psnr_dir.mkdir(exist_ok=True)

    # Process min_target_max_psnr cases
    for i, result in enumerate(min_target_max_psnr, 1):
        case_dir = min_target_dir / f"case_{i}"
        case_dir.mkdir(exist_ok=True)
        
        # Save analysis text file
        save_analysis_text(case_dir, result, i, "Minimum Target Discrepancy, Maximum PSNR Discrepancy")
        
        row_idx = result['row_idx']
        image = load_original_image(row_idx, dataset)
        reconstructed_imgs = {
            name: model(image) 
            for name, model in reconstruction_models.items()
        }
        
        save_analysis_images(row_idx, image["image_A"], image["image_B"], reconstructed_imgs, classifier, case_dir, image.get("segmentation_slice", None))
        
        # Log results
        logger.info(f"\nCase {i}:")
        logger.info(f"Row index: {row_idx}")
        logger.info(f"Target values: {result['target_values']}")
        logger.info(f"Target std: {result['target_std']:.4f}")
        logger.info(f"PSNR values: {result['psnr_values']}")
        logger.info(f"PSNR std: {result['psnr_std']:.4f}")

    # Process min_psnr_max_target cases
    for i, result in enumerate(min_psnr_max_target, 1):
        case_dir = min_psnr_dir / f"case_{i}"
        case_dir.mkdir(exist_ok=True)
        
        # Save analysis text file
        save_analysis_text(case_dir, result, i, "Minimum PSNR Discrepancy, Maximum Target Discrepancy")
        
        row_idx = result['row_idx']
        image = load_original_image(row_idx, dataset)
        reconstructed_imgs = {
            name: model(image) 
            for name, model in reconstruction_models.items()
        }
        
        save_analysis_images(row_idx, image["image_A"], image["image_B"], reconstructed_imgs, classifier, case_dir, image.get("segmentation_slice", None))
        
        # Log results
        logger.info(f"\nCase {i}:")
        logger.info(f"Row index: {row_idx}")
        logger.info(f"Target values: {result['target_values']}")
        logger.info(f"Target std: {result['target_std']:.4f}")
        logger.info(f"PSNR values: {result['psnr_values']}")
        logger.info(f"PSNR std: {result['psnr_std']:.4f}")

if __name__ == "__main__":
    main()
