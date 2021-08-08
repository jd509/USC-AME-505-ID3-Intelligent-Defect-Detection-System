#importing pillow module for images
from PIL import Image

#importing the GLCM and LBP module
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern

#importing numpy and pandas
import numpy as np
import pandas as pd

#function to extract features for a ***single image***
def extract_features(directory, dist, angle, radius):

# make list for each feature and a dictionary to have all features
    features = []
    directory = str(directory)
    contrasts = []
    dissimilarities = []
    homogeneties = []
    correlations = []
    energies = []

# load an image from file
    image = Image.open(directory)

# convert the image pixels to a numpy array
    img = np.array(image.getdata()).reshape(image.size[0], image.size[1])

#Calulate LBP Features and normalized LBP Histogram
    feat_lbp = local_binary_pattern(img, 8*radius, radius, 'uniform')
    feat_lbp = np.uint64((feat_lbp/feat_lbp.max())*255)

#Calculate GLCM Matrix and features from the LBP Histogram
    gcom = greycomatrix(feat_lbp, [dist], [angle], 256, symmetric=True, normed=True)
    contrast = greycoprops(gcom, prop='contrast')
    dissimilarity = greycoprops(gcom, prop='dissimilarity')
    homogeneity = greycoprops(gcom, prop='homogeneity')
    energy = greycoprops(gcom, prop='energy')
    correlation = greycoprops(gcom, prop='correlation')

# store feature
    contrasts.append(contrast[0][0])
    dissimilarities.append(dissimilarity[0][0])
    homogeneties.append(homogeneity[0][0])
    energies.append(energy[0][0])
    correlations.append(correlation[0][0])

#Add features to dictionary of features
    features['contrast'] = contrasts
    features['dissimilarity'] = dissimilarities
    features['homogeneity'] = homogeneties
    features['energy'] = energies
    features['correlation'] = correlations

#convert dictionary to dataframe
    df = pd.DataFrame(features)
    return df
