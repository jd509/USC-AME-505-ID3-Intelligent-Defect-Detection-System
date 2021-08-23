#!/usr/bin/python3

# importing os module
import os

#importing pillow module for images
from PIL import Image

#importing the GLCM module
from skimage.feature import greycomatrix, greycoprops

#importing numpy and pandas
import numpy as np
import pandas as pd
from keras_preprocessing import image

# import module for collecting parameters for models
import json

class Predict: 

    def __init__(self, image_directory):
        self.image_directory = image_directory

        self.config_directory = os.path.dirname(os.getcwd()) + '/config/'
        self.image_extraction_config_file = self.config_directory + \
            '/image_feature_extraction_config.json'

        with open(self.image_extraction_config_file) as f:
            self.image_extraction_params = json.load(f)


    def extract_glcm_features(self, dist=None,
                                    angle=None):
        """Function to extract glcm features for single image prediction

        Args:
            dist ([int], optional): Defaults to None.
            angle ([int], optional): Defaults to None.
        """
        if dist is None:
            dist = self.image_extraction_params['glcm']['pixel_offset_distance']
            pass
        if angle is None:
            angle = self.image_extraction_params['glcm']['pixel_pair_angles']
            pass

    # make list for each feature and a dictionary to have all features
        features = {}
        contrasts = []
        dissimilarities = []
        homogeneties = []
        correlations = []
        energies = []

    # load an image from file
        image = Image.open(self.image_directory)

    # convert the image pixels to a numpy array
        img = np.array(image.getdata()).reshape(image.size[0], image.size[1])

    #Calulate GLCM Features and Matrix
        gcom = greycomatrix(img, [dist], [angle], 256, symmetric=True, normed=True)
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

    
    def extract_lbglcm_features(self,dist=None,
                                    angle=None,
                                    radius=None):
        """Function to generate lbglcm features

        Args:
            dist ([int], optional): Defaults to None.
            angle ([int], optional): Defaults to None.
            radius ([int], optional): Defaults to None.
        """
        if dist is None:
            dist = self.image_extraction_params['lbglcm']['pixel_offset_distance']
            pass
        if angle is None:
            angle = self.image_extraction_params['lbglcm']['pixel_pair_angles']
            pass
        if radius is None:
            radius_of_neighbors = self.image_extraction_params['lbglcm']['radius_of_neighbors']
            pass
        
    # make list for each feature and a dictionary to have all features
        features = {}
        contrasts = []
        dissimilarities = []
        homogeneties = []
        correlations = []
        energies = []

    # load an image from file
        image = Image.open(self.image_directory)


    # convert the image pixels to a numpy array
        img = np.array(image.getdata()).reshape(image.size[0], image.size[1])

    #Calulate GLCM Features and Matrix
        gcom = greycomatrix(img, [dist], [angle], 256, symmetric=True, normed=True)
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


    #Predicting a new image or dataset of images
    def classify_image(self, trained_model, feature):
        return trained_model.predict(feature)
    
    def predict_defect(self,feature_extractor,
                            select_machine_learning_model,
                            trained_classification_model,
                            labels = None):
        """Function to predict the defect for selected image

        Args:
            feature_extractor ([str])
            select_machine_learning_model ([str])
            trained_classification_model ([dict])
            labels ([dict], optional): Defaults to None.

        Returns:
            [str]: type of defect
        """
        if feature_extractor == 'GLCM':
            glcm_feature = self.extract_glcm_features()

            for key in trained_classification_model:
                if key == select_machine_learning_model:
                    self.classification = labels[self.classify_image(trained_classification_model[select_machine_learning_model], glcm_feature)[0]]

        elif feature_extractor == 'LBGLCM':
            lbglcm_feature = self.extract_lbglcm_features()

            for key in trained_classification_model:
                if key == select_machine_learning_model:
                    self.classification = labels[self.classify_image(trained_classification_model[select_machine_learning_model], lbglcm_feature)[0]]

        elif feature_extractor == ' ':
            predict_image = image.load_img(self.image_directory, target_size = (64, 64))
            predict_image = image.img_to_array(predict_image)
            predict_image = np.expand_dims(predict_image, axis = 0)
            predict_image /= 255

            classification = self.classify_image(trained_classification_model['CNN'], predict_image)[0]
            detect_category = {0 : 'Crazing', 1: 'Inclusion', 2: 'Patches', 3: 'Pitted Surface', 4: 'RS', 5: 'Scratch'}
            self.classification = detect_category[np.argmax(classification)]

        return self.classification