import numpy as np
import os
from skimage.morphology import thin
from PIL import Image
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor


# -----------------------------------------------------------------------------
# evaluate_picture
# -----------------------------------------------------------------------------
def evaluate_picture(img_out, img_GT):
    GT_skel  = thin(img_GT, max_num_iter = 15) # On suppose que la demie epaisseur maximum 
    img_out_skel  = thin(img_out, max_num_iter = 15) # d'un vaisseau est de 15 pixels...
    TP = np.sum(img_out_skel & img_GT) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs

    PRECIS = TP / (TP + FP) # Precision
    RECALL = TP / (TP + FN) # Rappel
    F1SCORE = 2 / (1/PRECIS + 1/RECALL)
    return PRECIS, RECALL, F1SCORE, img_out_skel, GT_skel


# -----------------------------------------------------------------------------
# plot_segmentation
# -----------------------------------------------------------------------------
def plot_segmentation(img, img_out, img_out_skel, img_GT, GT_skel, save_plot_path=None):
    plt.subplot(231)
    plt.imshow(img,cmap = 'gray')
    plt.title('Image Originale')
    plt.subplot(232)
    plt.imshow(img_out)
    plt.title('Segmentation')
    plt.subplot(233)
    plt.imshow(img_out_skel)
    plt.title('Segmentation squelette')
    plt.subplot(235)
    plt.imshow(img_GT)
    plt.title('Vérité Terrain')
    plt.subplot(236)
    plt.imshow(GT_skel)
    plt.title('Vérité Terrain Squelette')

    if save_plot_path:
        if not os.path.exists(os.path.dirname(save_plot_path)):
            os.makedirs(os.path.dirname(save_plot_path))
        plt.savefig(save_plot_path)
    else:
        plt.show()


# -----------------------------------------------------------------------------
# evaluate_segmentation_picture
# -----------------------------------------------------------------------------
def evaluate_segmentation_picture(img_star, img_GT, seg_function, *args, save_plot_path=None, printing=False):
    #Ouvrir l'image originale en niveau de gris
    img =  Image.open(f'./data/images_IOSTAR/{img_star}')
    img_out = seg_function(img, *args)

    #Ouvrir l'image Verite Terrain en booleen
    img_GT =  np.asarray(Image.open(f'./data/images_IOSTAR/{img_GT}')).astype(np.bool_)

    precision, recall, f1score, img_out_skel, GT_skel = evaluate_picture(img_out, img_GT)

    if printing:
        print(f'{img_star}: \t Precision = {precision:0.2%}, Recall = {recall:0.2%}, F1 Score = {f1score:0.2%}')

    if save_plot_path:
        plot_segmentation(img, img_out, img_out_skel, img_GT, GT_skel, save_plot_path)

    return precision, recall, f1score


# -----------------------------------------------------------------------------
# evaluate_param_combination
# -----------------------------------------------------------------------------
def evaluate_param_combination(img_star_array, img_GT_array, seg_function, params):
    """
    Function to evaluate a single combination of parameters

    Args:
        params (_type_): _description_

    Returns:
        _type_: _description_
    """
    contrast_factor, p_structure, gaussian_sigma, threshold, remove_min_size = params # Extract parameters
    precisions, recalls, f1scores = [], [], [] # Initialize arrays for metrics across images

    # Iterate over images
    for img_star, img_GT in zip(img_star_array, img_GT_array):
        precision, recall, f1score = evaluate_segmentation_picture(img_star, 
                                                                   img_GT, 
                                                                   seg_function, 
                                                                   contrast_factor, 
                                                                   p_structure, 
                                                                   gaussian_sigma, 
                                                                   threshold, 
                                                                   remove_min_size)
        precisions.append(precision)
        recalls.append(recall)
        f1scores.append(f1score)

    # Calculate mean indicators
    precision_mean = np.mean(precisions)
    recall_mean = np.mean(recalls)
    f1score_mean = np.mean(f1scores)
    return params, precision_mean, recall_mean, f1score_mean


# -----------------------------------------------------------------------------
# find_best_parameters
# -----------------------------------------------------------------------------
def find_best_parameters(img_star_array, img_GT_array, seg_function, parameter_combinations):
    """
    Function to find the best parameter set

    Returns:
        _type_: _description_
    """
    # Initialization
    best_params = None
    best_f1score_mean = 0

    # Utilize Process Pool Executor for best parallelizing
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_param_combination, img_star_array, img_GT_array, seg_function, params) for params in parameter_combinations]

        # Iterate for each combination of parameters
        for future in futures:
            params, precision_mean, recall_mean, f1score_mean = future.result()
            if f1score_mean > best_f1score_mean: # When f1score is maximized...
                best_params = params # ... store best combination of parameters
                best_f1score_mean = f1score_mean # ... store best f1 score
                best_precision_mean = precision_mean # ... store related mean precision
                best_recall_mean = recall_mean # ... store related mean recall
               
    return best_params, best_precision_mean, best_recall_mean, best_f1score_mean


# -----------------------------------------------------------------------------
# results_latex_format
# -----------------------------------------------------------------------------
def results_latex_format(dict):
    lines = []
    for key, (precision, recall, f1score) in dict.items():
        lines.append(f"Image {key} & {f1score*100:.2f}\% & {precision*100:.2f}\% & {recall*100:.2f}\% \\\\")

    # Join all lines into a single string
    formatted_text = "\n".join(lines)
    return formatted_text