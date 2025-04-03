import numpy as np
from scipy.ndimage import affine_transform

def random_affine(image, max_rotation=10, max_translation=5):
    """Apply random affine transformations."""
    rotation = np.deg2rad(np.random.uniform(-max_rotation, max_rotation))
    translation = np.random.uniform(-max_translation, max_translation, size=3)
    matrix = np.eye(3)
    matrix[:2, :2] = [[np.cos(rotation), -np.sin(rotation)],
                      [np.sin(rotation), np.cos(rotation)]]
    return affine_transform(image, matrix, offset=translation)

def augment_image(image):
    """Apply augmentation to an image."""
    return random_affine(image)
