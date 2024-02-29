import numpy as np
from src.Common.utils import *
from skimage.morphology import opening, black_tophat, erosion, reconstruction
from skimage.filters import median
from PIL import ImageEnhance
import itertools
from scipy.ndimage import rotate


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


# -----------------------------------------------------------------------------
# create_rectangle
# -----------------------------------------------------------------------------
def create_rectangle(width, height, rotation=0):
    # Create a rectangle with ones
    rectangle = np.ones((height, width))
    # Rotate the rectangle if needed
    if rotation != 0:
        # Increase size before rotation to avoid clipping
        larger_side = max(width, height) * 2
        larger_rectangle = np.zeros((larger_side, larger_side))
        start_x = (larger_side - height) // 2
        start_y = (larger_side - width) // 2
        larger_rectangle[start_x:start_x+height, start_y:start_y+width] = rectangle
        # Rotate and return
        rotated = rotate(larger_rectangle, rotation, reshape=False, mode='constant', cval=0)
        return rotated
    return rectangle


# -----------------------------------------------------------------------------
# generate_rectangles_p_structures
# -----------------------------------------------------------------------------
def generate_rectangles_p_structures(widths, heights, orientations, max_rectangles=3):
    p_structures = []
    single_rect_combinations = list(itertools.product(widths, heights, orientations))
    
    for r in range(1, max_rectangles + 1):
        # Generate all combinations of r rectangles
        for combination in itertools.combinations(single_rect_combinations, r):
            # Convert each combination to the desired format
            p_structure = [create_rectangle(w,h,o) for w, h, o in combination]
            p_structures.append(p_structure)
    
    return p_structures