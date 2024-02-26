import numpy as np
from src.Common.utils import *
from skimage.morphology import opening, black_tophat, erosion, reconstruction
from skimage.filters import median
from PIL import ImageEnhance


# -----------------------------------------------------------------------------
# enhance_contrast
# -----------------------------------------------------------------------------
def enhance_contrast(img, contrast_factor):
    enhancer = ImageEnhance.Contrast(img)
    enhanced_image = enhancer.enhance(contrast_factor)
    return enhanced_image


# -----------------------------------------------------------------------------
# max_opening_by_structures
# -----------------------------------------------------------------------------
def max_opening_by_structures(img, structures):
    # Loop on all structs
    opened_imgs = [opening(img, struct) for struct in structures]
    
    return np.maximum.reduce(opened_imgs)


# -----------------------------------------------------------------------------
# sum_black_tophats_by_structures
# -----------------------------------------------------------------------------
def sum_black_tophats_by_structures(img, structures):
    # Loop on all structs
    summed_tophats = np.zeros(img.shape, dtype=float)
    for struct in structures:
        summed_tophats += black_tophat(img, struct)

    return summed_tophats


# -----------------------------------------------------------------------------
# max_erosion_with_reconstruct
# -----------------------------------------------------------------------------
def max_erosion_with_reconstruct(img, structures):
    # Loop on all structs
    eroded_imgs = [erosion(img, struct) for struct in structures]
    max_eroded = np.maximum.reduce(eroded_imgs)
    reconstructed = reconstruction(max_eroded, img, method='dilation')

    return reconstructed


# -----------------------------------------------------------------------------
# median_image_by_structures
# -----------------------------------------------------------------------------
def median_image_by_structures(img, structures):
    median_imgs = [median(img, struct) for struct in structures]

    return np.maximum.reduce(median_imgs)


