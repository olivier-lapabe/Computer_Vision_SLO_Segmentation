from src.Common.utils import *
from src.SegmentationFunctions.SegmentationFunctions import *
import itertools
from skimage.morphology import disk, square, rectangle, remove_small_objects
from skimage.filters import gaussian
import time


# -----------------------------------------------------------------------------
# Image Config
# -----------------------------------------------------------------------------
img_star_array = [
    "star01_OSC.jpg", "star02_OSC.jpg", "star03_OSN.jpg", "star08_OSN.jpg",
    "star21_OSC.jpg", "star26_ODC.jpg", "star28_ODN.jpg", "star32_ODC.jpg",
    "star37_ODN.jpg", "star48_OSN.jpg"
]
img_GT_array = [
    "GT_01.png", "GT_02.png", "GT_03.png", "GT_08.png",
    "GT_21.png", "GT_26.png", "GT_28.png", "GT_32.png",
    "GT_37.png", "GT_48.png"
]


# -----------------------------------------------------------------------------
# Segmentation Parameters
# -----------------------------------------------------------------------------
# First grid search:
# contrast_factors = np.arange(1, 2, 0.1)
# p_structures = [
#     [disk(1), disk(2)],
#     [disk(1), disk(2), disk(3)],
#     [disk(1), disk(2), disk(3), disk(4)]
# ]
# gaussian_sigmas = np.arange(0.5, 1.5, 0.1)
# thresholds = range(5, 15)
# remove_min_sizes = range(50, 100, 5)

# Second grid search:
# widths = [2, 3, 4]  # Example widths
# heights = [1, 2, 3]  # Example heights
# orientations = [0, 45, 90]  # Example orientations
# p_structures = generate_rectangles_p_structures(widths, heights, orientations, max_rectangles=3)
# contrast_factors = [1.5]
# gaussian_sigmas = [1]
# thresholds = [10]
# remove_min_sizes = [75]

# Third grid search:
contrast_factors = np.arange(1, 2, 0.1)
p_structures = [[create_rectangle(3, 3, 0), create_rectangle(3, 3, 45), create_rectangle(3, 3, 90)]]
gaussian_sigmas = np.arange(0.5, 1.5, 0.1)
thresholds = range(5, 15)
remove_min_sizes = range(50, 100, 5)

# Create all possible parameter combinations
parameter_combinations = itertools.product(contrast_factors,
                                           p_structures,
                                           gaussian_sigmas,
                                           thresholds,
                                           remove_min_sizes)


# -----------------------------------------------------------------------------
# segmentation_pipeline
# -----------------------------------------------------------------------------
def segmentation_pipeline(img, contrast_factor, p_structure, gaussian_sigma, threshold, remove_min_size):
    # Preprocessing
    img = enhance_contrast(img, contrast_factor)

    # Morphological operations
    img = max_opening_by_structures(img, p_structure)
    img = sum_black_tophats_by_structures(img, p_structure)
    img = max_erosion_with_reconstruct(img, p_structure)

    # Filtering
    img = gaussian(img, sigma=gaussian_sigma)
    img = img > threshold
    img = remove_small_objects(img, min_size=remove_min_size)
    img = median_image_by_structures(img, p_structure)

    return img


if __name__ == "__main__":
    start_time = time.time()
    best_params, best_precision_mean, best_recall_mean, best_f1score_mean = find_best_parameters(img_star_array,
                                                                                                 img_GT_array,
                                                                                                 segmentation_pipeline,
                                                                                                 parameter_combinations)
    end_time = time.time()

    print(f"Best Parameters: {best_params}")
    print(f"Best Precision Mean: {best_precision_mean:.2%}")
    print(f"Best Recall Mean: {best_recall_mean:.2%}")
    print(f"Best F1 Score Mean: {best_f1score_mean:.2%}")
    print(f"Time taken: {end_time - start_time:.3f} seconds")