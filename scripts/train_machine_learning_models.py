#!/usr/bin/python3

# importing os module
import os
from keras.metrics import accuracy

# importing numpy and pandas for computation and storage
import numpy as np
import pandas as pd

# keras for CNN
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# importing modules for supervised learning algorithms
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# importing module for computing accuracy and splitting dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# importing pillow module for images
from PIL import Image

# importing the GLCM module
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern

# import module for collecting parameters for models
import json


class Train:

    def __init__(self, dataset_directory):
        self.dataset_directory = dataset_directory

        self.config_directory = os.path.dirname(os.getcwd()) + '/config/'
        self.image_extraction_config_file = self.config_directory + \
            '/image_feature_extraction_config.json'
        self.machine_learning_config_file = self.config_directory + \
            '/machine_learning_params.json'

        with open(self.image_extraction_config_file) as f:
            self.image_extraction_params = json.load(f)

        with open(self.machine_learning_config_file) as f:
            self.machine_learning_params = json.load(f)

        self.classification_labels = []
        for label in os.listdir(self.dataset_directory):
            if os.path.isdir(label):
                self.classification_labels.append(label)

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.coded_y_values = {}

    def extract_features_using_glcm(self,
                                    dist=None,
                                    angle=None,
                                    num_grey_levels=None,
                                    symmetric=True, normed=True):

        # make list for each feature and a dictionary to have all features
        if dist is None:
            dist = self.image_extraction_params['glcm']['pixel_offset_distance']
            pass
        if angle is None:
            angle = self.image_extraction_params['glcm']['pixel_pair_angles']
            pass
        if num_grey_levels is None:
            num_grey_levels = self.image_extraction_params['glcm']['number_of_grey_levels']
            pass

        features = {}
        contrasts = []
        dissimilarities = []
        homogeneities = []
        correlations = []
        energies = []
        type = []

        for defect_name in self.classification_labels:
            classification_dir_path = self.dataset_directory + '/' + defect_name
            for image_name in os.listdir(classification_dir_path):
                image_file_path = classification_dir_path + '/' + image_name
                image = Image.open(image_file_path)  # load an image from file
                image_array = np.array(image.getdata()).reshape(
                    image.size[0], image.size[1])  # convert the image pixels to a numpy array

        # Calulating GLCM Features and GLCM Matrix

                gcom = greycomatrix(image_array, [dist], [
                                    angle], num_grey_levels, symmetric=symmetric, normed=normed)
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
                print('>%s' % defect_name)

    # Adding features to dictionary of features

        features['contrast'] = contrasts
        features['dissimilarity'] = dissimilarities
        features['homogeneity'] = homogeneities
        features['energy'] = energies
        features['correlation'] = correlations
        features['type'] = type

    # convert dictionary to dataframe

        self.glcm_image_features = pd.DataFrame(features)

    def extract_features_using_lbglcm(self,
                                      dist=None,
                                      angle=None,
                                      num_grey_levels=None,
                                      symmetric=True, normed=True,
                                      num_neighbors=None, radius_of_neighbors=None,
                                      method=None):

        # make list for each feature and a dictionary to have all features
        if dist is None:
            dist = self.image_extraction_params['lbglcm']['pixel_offset_distance']
            pass
        if angle is None:
            angle = self.image_extraction_params['lbglcm']['pixel_pair_angles']
            pass
        if num_grey_levels is None:
            num_grey_levels = self.image_extraction_params['lbglcm']['number_of_grey_levels']
            pass
        if num_neighbors is None:
            num_neighbors = self.image_extraction_params['lbglcm']['number_of_neighbors']
            pass
        if radius_of_neighbors is None:
            radius_of_neighbors = self.image_extraction_params['lbglcm']['radius_of_neighbors']
            pass
        if method is None:
            method = self.image_extraction_params['lbglcm']['method']
            pass

        features = {}
        contrasts = []
        dissimilarities = []
        homogeneities = []
        correlations = []
        energies = []
        type = []

        for defect_name in self.classification_labels:
            classification_dir_path = self.dataset_directory + '/' + defect_name
            for image_name in os.listdir(classification_dir_path):
                image_file_path = classification_dir_path + '/' + image_name
                image = Image.open(image_file_path)  # load an image from file
                image_array = np.array(image.getdata()).reshape(
                    image.size[0], image.size[1])  # convert the image pixels to a numpy array

        # Calulate LBP Matrix and its normalized histogram

                feat_lbp = local_binary_pattern(
                    image_array, num_neighbors, radius_of_neighbors, method)
                feat_lbp = np.uint64((feat_lbp/feat_lbp.max())*255)

        # Calulating GLCM Features and GLCM Matrix

                gcom = greycomatrix(feat_lbp, [dist], [
                                    angle], num_grey_levels, symmetric=symmetric, normed=normed)
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
                print('>%s' % defect_name)

    # Adding features to dictionary of features

        features['contrast'] = contrasts
        features['dissimilarity'] = dissimilarities
        features['homogeneity'] = homogeneities
        features['energy'] = energies
        features['correlation'] = correlations
        features['type'] = type

    # convert dictionary to dataframe

        self.lbglcm_image_features = pd.DataFrame(features)

    def prepare_dataset_for_supervised_learning(self, image_feature_dataframe, test_size):

        y = image_feature_dataframe.pop('type')
        x = image_feature_dataframe
        y_encoded, y_unique = pd.factorize(y)

        j = 0
        for i in range(len(y_encoded)):
            if y_encoded[i] not in self.coded_y_values:
                self.coded_y_values[y_encoded[i]] = y_unique[j]
                j += 1

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y_encoded, test_size=test_size, random_state=42)

    def train_random_forest_classifier(self,number_of_trees=None, 
                                            max_features_to_classify = None,
                                            min_sample_leaf=None, 
                                            max_leaf_nodes=None, 
                                            number_of_parallel_workers=None):

        if number_of_trees is None:
            number_of_trees = self.machine_learning_params['random_forest_classifier']['number_of_tress']
            pass
        
        if max_features_to_classify is None:
            max_features_to_classify = self.machine_learning_params['random_forest_classifier']['max_features_to_classify']
            pass

        if min_sample_leaf is None:
            min_sample_leaf = self.machine_learning_params['random_forest_classifier']['min_sample_leaf']
            pass

        if max_leaf_nodes is None:
            max_leaf_nodes = self.machine_learning_params['random_forest_classifier']['max_leaf_nodes']
            pass

        if number_of_parallel_workers is None:
            number_of_parallel_workers = self.machine_learning_params['random_forest_classifier']['number_of_parallel_workers']
            pass
        
        classifier = RandomForestClassifier(n_estimators=number_of_trees, n_jobs=number_of_parallel_workers, random_state=25, max_features=max_features_to_classify,
                                            max_leaf_nodes=max_leaf_nodes, oob_score=True, max_depth=None, min_samples_leaf=min_sample_leaf)

        classifier.fit(self.x_train, self.y_train)

        y_prediction = classifier.predict(self.x_test)
        accuracy_of_model = accuracy_score(self.y_test, y_prediction)

        return classifier, accuracy_of_model

    def train_xtra_trees_classifier(self,   number_of_trees=None,
                                            max_features_to_classify=None,
                                            min_sample_leaf=None, 
                                            max_leaf_nodes=None, 
                                            number_of_parallel_workers=None):

        if number_of_trees is None:
            number_of_trees = self.machine_learning_params['xtra_trees_classifier']['number_of_tress']
            pass
        
        if max_features_to_classify is None:
            max_features_to_classify = self.machine_learning_params['xtra_trees_classifier']['max_features_to_classify']
            pass

        if min_sample_leaf is None:
            min_sample_leaf = self.machine_learning_params['xtra_trees_classifier']['min_sample_leaf']
            pass

        if max_leaf_nodes is None:
            max_leaf_nodes = self.machine_learning_params['xtra_trees_classifier']['max_leaf_nodes']
            pass

        if number_of_parallel_workers is None:
            number_of_parallel_workers = self.machine_learning_params['xtra_trees_classifier']['number_of_parallel_workers']
            pass

        classifier = ExtraTreesClassifier(n_estimators=number_of_trees, n_jobs=number_of_parallel_workers, random_state=0, max_leaf_nodes=max_leaf_nodes,
                                          max_features=max_features_to_classify, oob_score=True, max_depth=15, min_samples_leaf=min_sample_leaf,
                                          bootstrap=True)

        classifier.fit(self.x_train, self.y_train)

        y_prediction = classifier.predict(self.x_test)
        accuracy_of_model = accuracy_score(self.y_test, y_prediction)

        return classifier, accuracy_of_model

    def train_gradient_boosting_classifier(self,number_of_estimators=None, 
                                                max_features_to_classify=None,
                                                loss_function=None, 
                                                learning_rate=None, 
                                                max_leaf_nodes=None):

        if number_of_estimators is None:
            number_of_estimators = self.machine_learning_params['gradient_boosting_classifier']['number_of_estimators']
            pass
        
        if max_features_to_classify is None:
            max_features_to_classify = self.machine_learning_params['gradient_boosting_classifier']['max_features_to_classify']
            pass

        if loss_function is None:
            loss_function = self.machine_learning_params['gradient_boosting_classifier']['loss_function']
            pass

        if learning_rate is None:
            learning_rate = self.machine_learning_params['gradient_boosting_classifier']['learning_rate']
            pass

        if max_leaf_nodes is None:
            max_leaf_nodes = self.machine_learning_params['gradient_boosting_classifier']['max_leaf_nodes']
            pass
        
        classifier = GradientBoostingClassifier(loss=loss_function, n_estimators=number_of_estimators, learning_rate=learning_rate,
                                                max_features=max_features_to_classify, max_depth=None, max_leaf_nodes=max_leaf_nodes, random_state=9,
                                                subsample=0.5)

        classifier.fit(self.x_train, self.y_train)

        y_prediction = classifier.predict(self.x_test)
        accuracy_of_model = accuracy_score(self.y_test, y_prediction)

        return classifier, accuracy_of_model

    def CNN(self, epoch, val_split, save_model):

        # Creating the model

        def create_model():
            model = Sequential([
                Conv2D(16, 3, padding='same', activation='relu',
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                MaxPooling2D(),
                Conv2D(32, 3, padding='same', activation='relu'),
                MaxPooling2D(),
                Conv2D(64, 3, padding='same', activation='relu'),
                MaxPooling2D(),
                Flatten(),
                Dense(512, activation='relu'),
                Dense(6, activation='softmax')])

        # Compiling Model using optimizer and loss functions
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            model.summary()
            return model

        # Setting up directory and validation split for the dataset
        data_dir = self.dataset_directory
        if val_split is None:
            val_split = self.machine_learning_params['CNN']['val_split']
        val_split = val_split
        dataset_image_generator = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True,
                                                     validation_split=val_split)

        # Setting up batch size and image parameters
        batch_size_train = 600
        batch_size_test = 400
        if epoch is None:
            epoch = self.machine_learning_params['CNN']['epoch']
        epochs = epoch
        IMG_HEIGHT = 64
        IMG_WIDTH = 64

        # Generating training and test dataset
        train_data_gen = dataset_image_generator.flow_from_directory(batch_size=batch_size_train, directory=data_dir,
                                                                     subset="training", shuffle=True,
                                                                     target_size=(
                                                                         IMG_HEIGHT, IMG_WIDTH),
                                                                     class_mode='categorical')
        val_data_gen = dataset_image_generator.flow_from_directory(batch_size=batch_size_test, directory=data_dir,
                                                                   target_size=(
                                                                       IMG_HEIGHT, IMG_WIDTH),
                                                                   class_mode='categorical', subset="validation")

        model = create_model()

        if save_model:
            filepath = os.path.dirname(os.getcwd()) + '/' + 'CNN.h5'
            model.save(filepath, overwrite=True, include_optimizer=True)

        # Generating history of the model and fitting dataset
        history = model.fit(
            train_data_gen,
            steps_per_epoch=batch_size_train,
            epochs=epochs,
            validation_data=val_data_gen, validation_steps=batch_size_test)

        # Getting validation accuracy
        val_acc = history.history['val_accuracy']

        return val_acc, model, val_data_gen

    # Load Pretrained CNN Model

    def pretrained_CNN(self):

        # give validation split here
        val_split = 0.2

        # Training and test data generation with needed batch size
        dataset_image_generator = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True,
                                                     validation_split=val_split)

        batch_size_train = 600
        batch_size_test = 400
        IMG_HEIGHT = 64
        IMG_WIDTH = 64

        # Creating a model
        def create_model():
            model = Sequential([
                Conv2D(16, 3, padding='same', activation='relu',
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                MaxPooling2D(),
                Conv2D(32, 3, padding='same', activation='relu'),
                MaxPooling2D(),
                Conv2D(64, 3, padding='same', activation='relu'),
                MaxPooling2D(),
                Flatten(),
                Dense(512, activation='relu'),
                Dense(6, activation='softmax')])

            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            model.summary()
            return model

        val_data_gen = dataset_image_generator.flow_from_directory(batch_size=batch_size_test, directory=self.dataset_directory,
                                                                   target_size=(
                                                                       IMG_HEIGHT, IMG_WIDTH),
                                                                   class_mode='categorical', subset="validation")

        # Model creation to load the checkpoint
        new_model = create_model()

        #*************************Loading Checkpoint Path***********************************#
        check_path = ''

        # Loading weights from the checkpoint
        new_model.load_weights(check_path)

        # Getting loss and accuracy values
        loss, acc = new_model.evaluate(val_data_gen)
        return acc, new_model
