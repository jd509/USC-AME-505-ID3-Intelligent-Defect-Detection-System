#!/usr/bin/python3

#importing os module
import os
import pathlib
from pathlib import Path

# importing numpy and pandas for computation and storage
import numpy as np
import pandas as pd

# keras for CNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

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
        self.dataset_directory = dataset_directory + '/'

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
        print(self.dataset_directory)
        for label in os.listdir(self.dataset_directory):
            if os.path.isdir(self.dataset_directory + label + '/'):
                self.classification_labels.append(label)
        print(self.classification_labels)
        
        self.coded_y_values = {}
        self.trained_classifier_models = {}
        self.model_accuracies = {}


    def extract_features_using_glcm(self, 
                                    dist = None, 
                                    angle = None, 
                                    num_grey_levels = None, 
                                    symmetric = True, normed = True):
        """Function to generate glcm features

        Args:
            dist ([int], optional): Defaults to None.
            angle ([int], optional): Defaults to None.
            num_grey_levels ([int], optional): Defaults to None.
            symmetric (bool, optional): Defaults to True.
            normed (bool, optional): Defaults to True.
        """
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

        print('Dist is: ',dist)
        print('Angle is: ',angle)

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

    #convert dictionary to dataframe
        
        self.glcm_image_features = pd.DataFrame(features)
        print(self.glcm_image_features)

    def extract_features_using_lbglcm(self,
                                      dist=None,
                                      angle=None,
                                      num_grey_levels=None,
                                      symmetric=True, normed=True,
                                      num_neighbors=None, radius_of_neighbors=None,
                                      method=None):
        """Function to generate LBGLCM features

        Args:
            dist ([int], optional): Defaults to None.
            angle ([int], optional): Defaults to None.
            num_grey_levels ([int], optional): Defaults to None.
            symmetric (bool, optional): Defaults to True.
            normed (bool, optional): Defaults to True.
            num_neighbors ([int], optional): Defaults to None.
            radius_of_neighbors ([int], optional): Defaults to None.
            method ([str], optional): Defaults to None.
        """

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
        if radius_of_neighbors is None:
            radius_of_neighbors = self.image_extraction_params['lbglcm']['radius_of_neighbors']
            pass
        if num_neighbors is None and radius_of_neighbors is None:
            num_neighbors = self.image_extraction_params['lbglcm']['number_of_neighbors']
            pass
        else:
            num_neighbors = int(8*radius_of_neighbors)
            pass
        if method is None:
            method = self.image_extraction_params['lbglcm']['method']        
            pass

        print('Dist is: ',dist)
        print('Angle is: ',angle)
        print('Radius is: ', radius_of_neighbors)
        print('Num neighbor is', num_neighbors)
        
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
        print(self.lbglcm_image_features)


    def prepare_dataset_for_supervised_learning(self, image_feature_dataframe, test_size, feature_type):
        """Function to prepare dataset for supervised learning

        Args:
            image_feature_dataframe ([pandas.Dataframe]): Image Features Dataframe
            test_size ([float]): Split size for test dataset
            feature_type ([str]): Type of feature extraction
        """
        y = image_feature_dataframe.pop('type')
        x = image_feature_dataframe
        y_encoded, y_unique = pd.factorize(y)

        j = 0
        for i in range(len(y_encoded)):
            if y_encoded[i] not in self.coded_y_values:
                self.coded_y_values[y_encoded[i]] = y_unique[j]
                j += 1
        
        if feature_type == "GLCM":
            x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=test_size, random_state=42)
            self.glcm_split_dataset = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

        if feature_type == "LBGLCM":
            x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=test_size, random_state=42)
            self.lbglcm_split_dataset = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

    def train_random_forest_classifier(self,number_of_trees=None, 
                                            max_features_to_classify = None,
                                            min_sample_leaf=None, 
                                            max_leaf_nodes=None, 
                                            number_of_parallel_workers=None):
        """Function to train random forest model

        Args:
            number_of_trees ([int], optional): Defaults to None.
            max_features_to_classify ([str], optional): Defaults to None.
            min_sample_leaf ([int], optional): Defaults to None.
            max_leaf_nodes ([int], optional): Defaults to None.
            number_of_parallel_workers ([int], optional): Defaults to None.
        """
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

        self.trained_classifier_models['random_forest_glcm'] = RandomForestClassifier(n_estimators=number_of_trees, n_jobs=number_of_parallel_workers, random_state=25, max_features=max_features_to_classify,
                                            max_leaf_nodes=max_leaf_nodes, oob_score=True, max_depth=None, min_samples_leaf=min_sample_leaf)

        self.trained_classifier_models['random_forest_lbglcm'] = RandomForestClassifier(n_estimators=number_of_trees, n_jobs=number_of_parallel_workers, random_state=25, max_features=max_features_to_classify,
                                            max_leaf_nodes=max_leaf_nodes, oob_score=True, max_depth=None, min_samples_leaf=min_sample_leaf)

        self.trained_classifier_models['random_forest_glcm'].fit(self.glcm_split_dataset['x_train'], self.glcm_split_dataset['y_train'])
        self.trained_classifier_models['random_forest_lbglcm'].fit(self.lbglcm_split_dataset['x_train'], self.lbglcm_split_dataset['y_train'])

        y_prediction = self.trained_classifier_models['random_forest_glcm'].predict(self.glcm_split_dataset['x_test'])
        self.model_accuracies['random_forest_glcm']  = accuracy_score(self.glcm_split_dataset['y_test'], y_prediction)

        y_prediction = self.trained_classifier_models['random_forest_lbglcm'].predict(self.lbglcm_split_dataset['x_test'])
        self.model_accuracies['random_forest_lbglcm'] = accuracy_score(self.lbglcm_split_dataset['y_test'], y_prediction)

    def train_xtra_trees_classifier(self,   number_of_trees=None,
                                            max_features_to_classify=None,
                                            min_sample_leaf=None, 
                                            max_leaf_nodes=None, 
                                            number_of_parallel_workers=None):
        """Function to train extra trees model

        Args:
            number_of_trees ([int], optional): Defaults to None.
            max_features_to_classify ([str], optional): Defaults to None.
            min_sample_leaf ([int], optional): Defaults to None.
            max_leaf_nodes ([int], optional): Defaults to None.
            number_of_parallel_workers ([int], optional): Defaults to None.
        """
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

        self.trained_classifier_models['xtra_trees_glcm'] = ExtraTreesClassifier(n_estimators=number_of_trees, n_jobs=number_of_parallel_workers, random_state=0, max_leaf_nodes=max_leaf_nodes,
                               max_features=max_features_to_classify, oob_score=True, max_depth=15, min_samples_leaf=min_sample_leaf,
                               bootstrap=True)

        self.trained_classifier_models['xtra_trees_lbglcm'] = ExtraTreesClassifier(n_estimators=number_of_trees, n_jobs=number_of_parallel_workers, random_state=0, max_leaf_nodes=max_leaf_nodes,
                               max_features=max_features_to_classify, oob_score=True, max_depth=15, min_samples_leaf=min_sample_leaf,
                               bootstrap=True)

        self.trained_classifier_models['xtra_trees_glcm'].fit(self.glcm_split_dataset['x_train'], self.glcm_split_dataset['y_train'])
        self.trained_classifier_models['xtra_trees_lbglcm'].fit(self.lbglcm_split_dataset['x_train'], self.lbglcm_split_dataset['y_train'])
        
        y_prediction = self.trained_classifier_models['xtra_trees_glcm'].predict(self.glcm_split_dataset['x_test'])
        self.model_accuracies['xtra_trees_glcm']  = accuracy_score(self.glcm_split_dataset['y_test'], y_prediction)

        y_prediction = self.trained_classifier_models['xtra_trees_lbglcm'].predict(self.lbglcm_split_dataset['x_test'])
        self.model_accuracies['xtra_trees_lbglcm']  = accuracy_score(self.lbglcm_split_dataset['y_test'], y_prediction)

    def train_gradient_boosting_classifier(self,number_of_estimators=None, 
                                                max_features_to_classify=None,
                                                loss_function=None, 
                                                learning_rate=None, 
                                                max_leaf_nodes=None):
        """Function to train gradient boosting model

        Args:
            number_of_estimators ([int], optional): Defaults to None.
            max_features_to_classify ([str], optional): Defaults to None.
            loss_function ([str], optional): Defaults to None.
            learning_rate ([float], optional): Defaults to None.
            max_leaf_nodes ([int], optional): Defaults to None.
        """
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
        
        self.trained_classifier_models['gradient_boosting_glcm'] = GradientBoostingClassifier(loss=loss_function, n_estimators=number_of_estimators, learning_rate=learning_rate,
                                                max_features=max_features_to_classify, max_depth=None, max_leaf_nodes=max_leaf_nodes, random_state=9,
                                                subsample=0.5)

        self.trained_classifier_models['gradient_boosting_lbglcm'] = GradientBoostingClassifier(loss=loss_function, n_estimators=number_of_estimators, learning_rate=learning_rate,
                                                max_features=max_features_to_classify, max_depth=None, max_leaf_nodes=max_leaf_nodes, random_state=9,
                                                subsample=0.5)

        self.trained_classifier_models['gradient_boosting_glcm'].fit(self.glcm_split_dataset['x_train'], self.glcm_split_dataset['y_train'])
        self.trained_classifier_models['gradient_boosting_lbglcm'].fit(self.lbglcm_split_dataset['x_train'], self.lbglcm_split_dataset['y_train'])
        
        y_prediction = self.trained_classifier_models['gradient_boosting_glcm'].predict(self.glcm_split_dataset['x_test'])
        self.model_accuracies['gradient_boosting_glcm']  = accuracy_score(self.glcm_split_dataset['y_test'], y_prediction)

        y_prediction = self.trained_classifier_models['gradient_boosting_lbglcm'].predict(self.lbglcm_split_dataset['x_test'])
        self.model_accuracies['gradient_boosting_lbglcm']  = accuracy_score(self.lbglcm_split_dataset['y_test'], y_prediction)
                                                        
    def CNN(self, epoch = None, val_split = None, save_model = True):
        """Function to train Neural Network

        Args:
            epoch ([int], optional): Defaults to None.
            val_split ([float], optional): Defaults to None.
            save_model (bool, optional): Defaults to True.
        """
        dataset_folder = pathlib.Path(self.dataset_directory)

        if val_split is None:
            val_split = self.machine_learning_params['CNN']['val_split']
        
        if epoch is None:
            epoch = self.machine_learning_params['CNN']['epoch']
        
        image_count = len(list(dataset_folder.glob('*/*.jpg')))
        print(image_count)

        batch_size = self.machine_learning_params['CNN']['batch_size']
        img_height = self.machine_learning_params['CNN']['image_dims'][0]
        img_width = self.machine_learning_params['CNN']['image_dims'][1]

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_folder,
        validation_split=val_split,
        subset="training",
        seed=123,
        color_mode="rgb",
        image_size=(img_height, img_width),
        label_mode="categorical",
        batch_size=batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_folder,
        validation_split=val_split,
        subset="validation",
        seed=123,
        color_mode="rgb",
        label_mode="categorical",
        image_size=(img_height, img_width),
        batch_size=batch_size)

        class_names = train_ds.class_names
        print(class_names)

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        num_classes = len(class_names)


        data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                        input_shape=(img_height, 
                                                                    img_width,
                                                                    3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
        )

        model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])

        model.summary()

        epochs = epoch

        history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
        )

        if save_model:
            save_model_path =  os.path.dirname(os.getcwd()) + '/saved_model/'
            save_model_dir = save_model_path + 'training_model_1.h5'
            model.save(save_model_dir)

        # Getting validation accuracy
        val_acc = history.history['val_accuracy']

        self.trained_classifier_models['CNN'] = model       
        self.model_accuracies['CNN'] = val_acc

    #Load Pretrained CNN Model
    def pretrained_CNN(self, validation_split = None):
        """Function to load a pretrained model

        Args:
            validation_split ([float], optional): Defaults to None.
        """
        dataset_folder = pathlib.Path(self.dataset_directory)

        if validation_split is None:
            validation_split = self.machine_learning_params['CNN']['val_split']
                
        image_count = len(list(dataset_folder.glob('*/*.jpg')))
        print(image_count)

        batch_size = self.machine_learning_params['CNN']['batch_size']
        img_height = self.machine_learning_params['CNN']['image_dims'][0]
        img_width = self.machine_learning_params['CNN']['image_dims'][1]

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_folder,
        validation_split=validation_split,
        subset="training",
        seed=123,
        color_mode="rgb",
        image_size=(img_height, img_width),
        label_mode="categorical",
        batch_size=batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_folder,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        color_mode="rgb",
        label_mode="categorical",
        image_size=(img_height, img_width),
        batch_size=batch_size)

        class_names = train_ds.class_names
        print(class_names)

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        save_model_path =  os.path.dirname(os.getcwd()) + '/saved_model/'
        save_model_dir = save_model_path + 'training_model_1.h5'

        # Loading weights from the checkpoint
        model = tf.keras.models.load_model(save_model_dir)
        model.summary()

        # Getting loss and accuracy values
        loss, acc = model.evaluate(val_ds)

        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

        self.trained_classifier_models['CNN'] = model       
        self.model_accuracies['CNN'] = acc

