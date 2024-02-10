import numpy as np
from skimage.morphology import thin
from PIL import Image
from matplotlib import pyplot as plt


# -----------------------------------------------------------------------------
# masking
# -----------------------------------------------------------------------------
def masking(img):
    nrows, ncols = img.shape
    row, col = np.ogrid[:nrows, :ncols]
    #On ne considere que les pixels dans le disque inscrit 
    img_mask = (np.ones(img.shape)).astype(np.bool_)
    invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
    img_mask[invalid_pixels] = 0
    return img_mask


# -----------------------------------------------------------------------------
# evaluate
# -----------------------------------------------------------------------------
def evaluate(img_out, img_GT):
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
def plot_segmentation(img, img_out, img_out_skel, img_GT, GT_skel):
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
    plt.show()


# -----------------------------------------------------------------------------
# evaluate_picture
# -----------------------------------------------------------------------------
def evaluate_picture(img_index, seg_function, *args, plotting = True, printing = True):
    #Ouvrir l'image originale en niveau de gris
    img =  np.asarray(Image.open(f'./images_IOSTAR/star{img_index}_OSC.jpg')).astype(np.uint8)
    img_out = seg_function(img, *args)

    #Ouvrir l'image Verite Terrain en booleen
    img_GT =  np.asarray(Image.open(f'./images_IOSTAR/GT_{img_index}.png')).astype(np.bool_)

    PRECIS, RECALL, F1SCORE, img_out_skel, GT_skel = evaluate(img_out, img_GT)

    if printing:
        print(f'Precision = {PRECIS:0.2%}, Recall = {RECALL:0.2%}, F1 Score = {F1SCORE:0.2%}')

    if plotting:
        plot_segmentation(img, img_out, img_out_skel, img_GT, GT_skel)

    return PRECIS, RECALL, F1SCORE