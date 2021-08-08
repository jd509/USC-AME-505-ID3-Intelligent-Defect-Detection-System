#importing os module (file handling in os) and pillow module for images
import os
from PIL import Image

#importing the GLCM and LBP module
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern

#importing numpy and pandas
import numpy as np
import pandas as pd

#function to extract features for a ***collection of images***
def extract_features(directory, dist, angle,radius):

# make list for each feature and a dictionary to have all features
    directory = str(directory)
    features = {}
    names = ['Crazing','Inclusion','Patches','Pitted Surface','RS','Scratch']
    contrasts = []
    dissimilarities = []
    homogeneties = []
    correlations = []
    energies = []
    type = []

#Iterating through each image and collecting features
    for defect_name in names:
        foldername = directory + '/' + defect_name
        for name in os.listdir(foldername):
            filename = foldername + '/' + name
            image = Image.open(filename) # load an image from file
            img = np.array(image.getdata()).reshape(image.size[0], image.size[1]) # convert the image pixels to a numpy array

    #Calulate LBP Matrix and its normalized histogram
            feat_lbp = local_binary_pattern(img, 8*radius, radius, 'uniform')
            feat_lbp = np.uint64((feat_lbp/feat_lbp.max())*255)

    #Calculate GLCM features for LBP histogram
            gcom = greycomatrix(feat_lbp, [dist], [angle], 256, symmetric=True, normed=True)
            contrast = greycoprops(gcom, prop='contrast')
            dissimilarity = greycoprops(gcom, prop='dissimilarity')
            homogeneity = greycoprops(gcom, prop='homogeneity')
            energy = greycoprops(gcom, prop='energy')
            correlation = greycoprops(gcom, prop='correlation')

    # Storing features in the lists
            contrasts.append(contrast[0][0])
            dissimilarities.append(dissimilarity[0][0])
            homogeneties.append(homogeneity[0][0])
            energies.append(energy[0][0])
            correlations.append(correlation[0][0])
            type.append(defect_name)
            print('>%s' % name)

#Adding features to dictionary of features
    features['contrast'] = contrasts
    features['dissimilarity'] = dissimilarities
    features['homogeneity'] = homogeneties
    features['energy'] = energies
    features['correlation'] = correlations
    features['type'] = type

#Converting dictionary to dataframe
    df = pd.DataFrame(features)
    return df
