import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize for consistency
    gray = cv2.resize(gray, (128, 128))

    # Compute GLCM
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Extract GLCM features
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        val = graycoprops(glcm, prop)[0, 0]
        features.append(val)

    return features
