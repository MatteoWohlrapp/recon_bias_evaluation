import os
import pathlib
from typing import Optional
from skimage.io import imread
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import polars as pl
import nibabel as nib
from fastmri import fft2c, ifft2c
from fastmri.data.subsample import RandomMaskFunc


def min_max_slice_normalization(scan: torch.Tensor) -> torch.Tensor:
    scan_min = scan.min()
    scan_max = scan.max()
    if scan_max == scan_min:
        return scan
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan


class ChexDataset(Dataset):
    """Dataset for X-ray reconstruction."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (dict) -- stores all the experiment flags
        """
        self.data_root_A = pathlib.Path(opt["dataroot_A"])
        self.data_root_B = pathlib.Path(opt["dataroot_B"])
        self.csv_path_A = pathlib.Path(opt["csv_path_A"])
        self.csv_path_B = pathlib.Path(opt["csv_path_B"])
        self.split = "test"

        # Load metadata
        self.metadata_A, self.metadata_B = self._load_metadata()

        # Set up transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256), antialias=True),
                min_max_slice_normalization,
            ]
        )

        # Create path to index mapping
        self.path_to_idx = {
            row["Path"]: idx for idx, row in enumerate(self.metadata_A.iter_rows(named=True))
        }

    def _load_metadata(self):
        """Load metadata for the dataset from CSV files."""
        if not self.csv_path_A.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path_A}")
        if not self.csv_path_B.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path_B}")

        df_A = pl.read_csv(self.csv_path_A)
        df_B = pl.read_csv(self.csv_path_B)

        # Filter by split
        df_A = df_A.filter(pl.col("split") == self.split)
        df_B = df_B.filter(pl.col("split") == self.split)

        return df_A, df_B

    def get_item_by_path(self, image_path):
        """Get item using image path instead of index."""
        if image_path not in self.path_to_idx:
            raise ValueError(f"Image path not found in dataset: {image_path}")
        return self[self.path_to_idx[image_path]]

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        row_A = self.metadata_A.row(index, named=True)
        row_B = self.metadata_B.row(index, named=True)

        # Load images
        image_path_A = os.path.join(self.data_root_A, row_A["Path"])
        image_path_B = os.path.join(self.data_root_B, row_B["Path"])

        image_A = imread(image_path_A, as_gray=True).astype(np.float32)
        image_B = imread(image_path_B, as_gray=True).astype(np.float32)

        # Apply transforms
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return {
            "image_A": image_A,  # degraded image
            "image_B": image_B,  # original image
            "image_path": row_A["Path"],
            "row_idx": index
        }

    def __len__(self):
        """Return the total number of images."""
        return len(self.metadata_A)


class UCSFDataset(Dataset):
    """Dataset for MRI reconstruction."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (dict) -- stores all the experiment flags
        """
        self.data_root = pathlib.Path(opt["dataroot"])
        self.sampling_mask = opt.get("sampling_mask", "radial")
        self.type = opt.get("type", "FLAIR")
        self.pathology = opt.get("pathology", [])
        self.lower_slice = opt.get("lower_slice", 60)
        self.upper_slice = opt.get("upper_slice", 130)
        self.split = "test"
        self.num_rays = opt.get("num_rays", 60)
        print(f"num_rays: {self.num_rays}")

        # Load metadata
        self.metadata = self._load_metadata()

        # Set up transforms
        self.transform = transforms.Compose(
            [
                min_max_slice_normalization,
                transforms.Resize((256, 256), antialias=True),
            ]
        )

        # Create path-slice to index mapping
        self.path_to_idx = {
            (row["file_path"], row["slice_id"]): idx 
            for idx, row in enumerate(self.metadata.iter_rows(named=True))
        }

    def _load_metadata(self):
        """Load metadata for the dataset from CSV file."""
        metadata_file = self.data_root / "metadata.csv"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        df = pl.read_csv(metadata_file)

        # Apply filters based on parameters
        if self.type:
            df = df.filter(pl.col("type") == self.type)
        if self.pathology:
            df = df.filter(pl.col("pathology").is_in(self.pathology))
        if self.lower_slice is not None:
            df = df.filter(pl.col("slice_id") >= self.lower_slice)
        if self.upper_slice is not None:
            df = df.filter(pl.col("slice_id") <= self.upper_slice)

        # Filter by split
        df = df.filter(pl.col("split") == self.split)

        return df

    def get_item_by_path(self, file_path, slice_id):
        """Get item using file path and slice ID instead of index."""
        key = (file_path, slice_id)
        if key not in self.path_to_idx:
            raise ValueError(f"Image path and slice ID not found in dataset: {key}")
        return self[self.path_to_idx[key]]

    def convert_to_complex(self, image_slice):
        """Convert a real-valued 2D image slice to complex format."""
        complex_tensor = torch.stack(
            (image_slice, torch.zeros_like(image_slice)), dim=-1
        )
        return complex_tensor

    def create_radial_mask(self, shape):
        """Create a radial mask for undersampling k-space."""
        H, W = shape
        center = (H // 2, W // 2)
        Y, X = np.ogrid[:H, :W]
        mask = np.zeros((H, W), dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)

        for angle in angles:
            line_x = np.cos(angle)
            line_y = np.sin(angle)
            for r in range(max(H, W) // 2):
                x = int(center[1] + r * line_x)
                y = int(center[0] + r * line_y)
                if 0 <= x < W and 0 <= y < H:
                    mask[y, x] = 1
        return mask

    def apply_radial_mask_to_kspace(self, kspace):
        """Apply a radial mask to the k-space data."""
        H, W, _ = kspace.shape
        radial_mask = self.create_radial_mask((H, W))
        radial_mask = torch.from_numpy(radial_mask).to(kspace.device).unsqueeze(-1)
        return kspace * radial_mask

    def apply_linear_mask_to_kspace(self, kspace):
        """Apply a linear mask to the k-space data."""
        mask_func = RandomMaskFunc(center_fractions=[0.1], accelerations=[6])
        mask = mask_func(kspace.shape, seed=None)[0]
        mask = mask.to(kspace.device).unsqueeze(-1)
        return kspace * mask

    def undersample_slice(self, slice_tensor):
        """Undersample an MRI slice using specified mask."""
        # Convert real slice to complex-valued tensor
        complex_slice = self.convert_to_complex(slice_tensor)

        # Transform to k-space
        kspace = fft2c(complex_slice)

        # Apply mask
        if self.sampling_mask == "radial":
            undersampled_kspace = self.apply_radial_mask_to_kspace(kspace)
        elif self.sampling_mask == "linear":
            undersampled_kspace = self.apply_linear_mask_to_kspace(kspace)
        else:
            raise ValueError(f"Unsupported sampling mask: {self.sampling_mask}")

        # Inverse transform
        undersampled_image = ifft2c(undersampled_kspace)
        return torch.abs(undersampled_image[..., 0])
    
    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        row = self.metadata.row(index, named=True)

        # Load the original image
        nifti_img = nib.load(str(self.data_root / row["file_path"]))
        scan = nifti_img.get_fdata()
        slice_tensor = torch.from_numpy(scan[:, :, row["slice_id"]]).float()

        # Add channel dimension before applying transforms
        slice_tensor = slice_tensor.unsqueeze(0)

        # Apply transforms
        if self.transform:
            slice_tensor = self.transform(slice_tensor)

        # Create undersampled version
        undersampled_tensor = self.undersample_slice(slice_tensor.squeeze(0))
        undersampled_tensor = undersampled_tensor.unsqueeze(0)

        segmentation_path = row["file_path"].replace(self.type, "tumor_segmentation")
        nifti_img = nib.load(str(self.data_root / segmentation_path))
        segmentation = nifti_img.get_fdata()
        segmentation_slice = segmentation[:, :, row["slice_id"]]
        segmentation_slice[segmentation_slice == 2] = 1
        segmentation_slice[segmentation_slice == 4] = 0

        segmentation_slice = torch.from_numpy(segmentation_slice).float().unsqueeze(0)
        segmentation_slice = transforms.functional.resize(
            segmentation_slice, (256, 256)
        )


        return {
            "image_A": undersampled_tensor,  # undersampled image
            "image_B": slice_tensor,  # fully sampled image
            "patient_id": row["patient_id"],
            "slice_id": row["slice_id"],
            "file_path": row["file_path"],
            "row_idx": index,
            "segmentation_slice": segmentation_slice
        }

    def __len__(self):
        """Return the total number of images."""
        return len(self.metadata)


def create_dataset(config):
    """
    Create appropriate dataset based on config
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        Dataset: Appropriate dataset instance
    """
    if config["name"] == "chex":
        return ChexDataset(config)
    elif config["name"] == "ucsf":
        return UCSFDataset(config)
    else:
        raise ValueError(f"Dataset {config['name']} not supported") 