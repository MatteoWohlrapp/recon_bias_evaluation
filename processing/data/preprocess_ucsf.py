import os
import pathlib
from pathlib import Path
import nibabel as nib
import numpy as np
import polars as pl
from skimage.io import imsave
import torchvision.transforms as transforms
import torch
from tqdm import tqdm

def min_max_slice_normalization(scan: np.ndarray) -> np.ndarray:
    scan_min = scan.min()
    scan_max = scan.max()
    if scan_max == scan_min:
        return scan
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan

def preprocess_ucsf_data(
    input_root: str,
    output_root: str,
    metadata_path: str,
    type: str = "FLAIR",
    lower_slice: int = 60,
    upper_slice: int = 130
):
    """
    Preprocess UCSF NIFTI files into PNG images.
    
    Args:
        input_root: Path to input NIFTI files
        output_root: Path to output PNG files
        metadata_path: Path to original metadata file
        type: Type of scan to process
        lower_slice: Lower slice bound
        upper_slice: Upper slice bound
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Create images directory
    images_dir = output_root / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Read metadata
    df = pl.read_csv(metadata_path)
    
    # Filter metadata
    if type:
        df = df.filter(pl.col("type") == type)
    
    # Initialize new metadata list
    new_metadata = []
    
    # Process each unique patient
    unique_patients = df.select("patient_id", "file_path").unique()
    
    for row in tqdm(unique_patients.iter_rows(), total=len(unique_patients)):
        patient_id, file_path = row
        
        # Load NIFTI file
        nifti_path = input_root / file_path
        if not nifti_path.exists():
            print(f"Warning: File not found: {nifti_path}")
            continue
            
        try:
            nifti_img = nib.load(str(nifti_path))
            scan = nifti_img.get_fdata()
            
            # Process each slice
            for slice_id in range(max(0, lower_slice), min(scan.shape[2], upper_slice + 1)):
                slice_data = scan[:, :, slice_id]
                
                # Normalize slice
                slice_data = min_max_slice_normalization(slice_data)
                
                # Create filename: patient_id_slice_id.png
                output_filename = f"{patient_id}_{slice_id:03d}.png"
                output_path = images_dir / output_filename
                
                # Save image
                imsave(
                    str(output_path),
                    (slice_data * 255).astype(np.uint8),
                    check_contrast=False
                )
                
                # Add to new metadata
                new_metadata.append({
                    "patient_id": patient_id,
                    "slice_id": slice_id,
                    "path": f"images/{output_filename}",
                    "split": "test"  # You might want to preserve the original split if available
                })
                
        except Exception as e:
            print(f"Error processing {nifti_path}: {e}")
    
    # Create new metadata file
    new_df = pl.DataFrame(new_metadata)
    new_df.write_csv(output_root / "metadata.csv")
    
    print(f"Preprocessing complete. Processed {len(new_metadata)} slices.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--type", type=str, default="FLAIR")
    parser.add_argument("--lower_slice", type=int, default=60)
    parser.add_argument("--upper_slice", type=int, default=130)
    args = parser.parse_args()
    
    preprocess_ucsf_data(
        args.input_root,
        args.output_root,
        args.metadata_path,
        args.type,
        args.lower_slice,
        args.upper_slice
    ) 