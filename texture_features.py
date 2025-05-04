import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from mahotas import features as mh_features

def extract_glcm_features(gray):
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return [graycoprops(glcm, prop)[0, 0] for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]

def extract_lbp_features(gray):
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist.tolist()

def extract_haralick_features(gray):
    return mh_features.haralick(gray).mean(axis=0).tolist()

def extract_combined_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    glcm = extract_glcm_features(gray)
    lbp = extract_lbp_features(gray)
    haralick = extract_haralick_features(gray)

    return glcm + lbp + haralick
