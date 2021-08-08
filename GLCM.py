#importing os module (file handling in os) and pillow module for images
import os
from PIL import Image

#importing the GLCM module
from skimage.feature import greycomatrix, greycoprops

#importing numpy and pandas
import numpy as np
import pandas as pd


#function to extract features for a ***collection of images***
def extract_features(directory, dist, angle):

# make list for each feature and a dictionary to have all features
    directory = str(directory)
    features = {}
    names = ['Crazing','Inclusion','Patches','Pitted Surface','RS','Scratch']
    contrasts = []
    dissimilarities = []
    homogeneities = []
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

    #Calulating GLCM Features and GLCM Matrix
            gcom = greycomatrix(img, [dist], [angle], 256, symmetric=True, normed=True)
            contrast = greycoprops(gcom, prop='contrast')
            dissimilarity = greycoprops(gcom, prop='dissimilarity')
            homogeneity = greycoprops(gcom, prop='homogeneity')
            energy = greycoprops(gcom, prop='energy')
            correlation = greycoprops(gcom, prop='correlation')

    # Storing features in the lists
            contrasts.append(contrast[0][0])
            dissimilarities.append(dissimilarity[0][0])
            homogeneities.append(homogeneity[0][0])
            energies.append(energy[0][0])
            correlations.append(correlation[0][0])
            type.append(defect_name)
            print('>%s' % name)

#Adding features to dictionary of features
    features['contrast'] = contrasts
    features['dissimilarity'] = dissimilarities
    features['homogeneity'] = homogeneities
    features['energy'] = energies
    features['correlation'] = correlations
    features['type'] = type

#convert dictionary to dataframe
    df = pd.DataFrame(features)
    return df

