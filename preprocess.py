import nibabel as nib
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def load_nifti(file_path):
    """Load a NIfTI file and return the image data."""
    img = nib.load(file_path)
    return img.get_fdata()

def normalize_image(image):
    """Normalize image intensity to [0, 1]."""
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def pad_image_to_multiple(image, multiple=16):
    """Pad the image dimensions to the nearest multiple of the specified value."""
    target_shape = [(dim + multiple - 1) // multiple * multiple for dim in image.shape]
    pad_width = [(0, target - dim) for dim, target in zip(image.shape, target_shape)]
    return np.pad(image, pad_width, mode='constant', constant_values=0)

def preprocess_image(file_path):
    """Load and preprocess a neuroimaging file."""
    logging.info(f"Preprocessing file: {file_path}")
    image = load_nifti(file_path)
    normalized_image = normalize_image(image)
    padded_image = pad_image_to_multiple(normalized_image)
    logging.info("Preprocessing complete.")
    return padded_image
