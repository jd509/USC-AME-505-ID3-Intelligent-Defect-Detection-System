# importing os module
import os

#importing pillow module for images
from PIL import Image

#importing the GLCM module
from skimage.feature import greycomatrix, greycoprops

#importing numpy and pandas
import numpy as np
import pandas as pd

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
        pred = trained_model.predict(feature)
        return pred
    
    def predict_defect(self,feature_extractor,
                            machine_learning_model,
                            trained_model,
                            lables):

        if feature_extractor == 'GLCM':
            glcm_feature = Predict.extract_glcm_features()

            for key in trained_model:
                if key == machine_learning_model:
                    classification = Predict.classify_image(trained_model[machine_learning_model], glcm_feature)
                    return classification

            # if machine_learning_model == 'Random Forest':
            #     rf_prediction = Predict.pred(trained_model['random_forest'], glcm_feature)
            #     return rf_prediction
                
            # elif machine_learning_model == 'Extra Trees Classifier':
            #     xt_prediction = Predict.pred(trained_model['extra_tree_classifier'], glcm_feature)
            #     return xt_prediction

            # elif machine_learning_model == 'Gradient Boosting Classifier':
            #     gb_prediction = Predict.pred(trained_model['gradient_boosting_classifier'], glcm_feature)
            #     return gb_prediction

        elif feature_extractor == 'LBGLCM':
            lbglcm_feature = Predict.extract_lbglcm_features()

            for key in trained_model:
                if key == machine_learning_model:
                    classification = Predict.classify_image(trained_model[machine_learning_model], lbglcm_feature)
                    return classification

            # if machine_learning_model == 'Random Forest':
            #     rf_prediction = Predict.pred(trained_model['random_forest'], lbglcm_feature)
            #     return rf_prediction

            # elif machine_learning_model == 'Extra Trees Classifier':
            #     xt_prediction = Predict.pred(trained_model['extra_tree_classifier'], lbglcm_feature)
            #     return xt_prediction

            # else:
            #     gb_prediction = Predict.pred(trained_model['gradient_boosting_classifier'], lbglcm_feature)
            #     return gb_prediction

        elif feature_extractor == ' ':



