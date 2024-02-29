from src.Common.utils import *
from src.SegmentationFunctions.SegmentationFunctions import *
from skimage.morphology import disk, square, rectangle, remove_small_objects
from skimage.filters import gaussian
import time
from scipy.ndimage import rotate

# -----------------------------------------------------------------------------
# Image Config
# -----------------------------------------------------------------------------
# img_star_array = [
#     "star01_OSC.jpg"
# ]
# img_GT_array = [
#     "GT_01.png"
# ]
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
# Optimal set of Parameters
# -----------------------------------------------------------------------------
contrast_factor = 1.5
#p_structure = [create_rectangle(3, 3, 0), create_rectangle(4, 4, 0), create_rectangle(5, 5, 0)] --> 76.26%
#p_structure = [create_rectangle(3, 3, 0), create_rectangle(3, 3, 45), create_rectangle(3, 3, 90)] --> 77.38%
p_structure = [create_rectangle(3, 3, 0), create_rectangle(3, 3, 45), create_rectangle(3, 3, 90)]
#p_structure = [rect(4, 4, 0), rect(4, 4, 45), rect(4, 4, 90)] --> 76.44%
#p_structure = [disk(1), disk(2), disk(3)]
gaussian_sigma = 0.9
threshold = 10
remove_min_size = 75


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
    # Store segmentation results in dictionary
    results_dict = {}
    precision_array, recall_array, f1score_array = [], [], []
    for img_star, img_GT in zip(img_star_array, img_GT_array):
        precision, recall, f1score = evaluate_segmentation_picture(img_star, 
                                    img_GT, 
                                    segmentation_pipeline, 
                                    contrast_factor,
                                    p_structure,
                                    gaussian_sigma,
                                    threshold,
                                    remove_min_size,
                                    save_plot_path = f"./data/results/{img_star[4:6]}_segmented", 
                                    printing = True)
        precision_array.append(precision)
        recall_array.append(recall)
        f1score_array.append(f1score)
        results_dict[img_star[4:6]] = precision, recall, f1score
    results_dict['average'] = np.mean(precision_array), np.mean(recall_array), np.mean(f1score_array)
    
    # Format and store results in latex format
    formatted_text = results_latex_format(results_dict)
    filename_txt = "./data/results/results_latex_format.txt"
    with open(filename_txt, 'w') as f:
        f.write(formatted_text)
        
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.3f} seconds")