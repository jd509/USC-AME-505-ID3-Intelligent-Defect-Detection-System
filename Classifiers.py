#importing os module
import os

#importing numpy and pandas for computation and storage
import numpy as np
import pandas as pd

#keras for CNN
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

#importing modules for supervised learning algorithms
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

#importing module for computing accuracy and splitting dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#function to capture labels of images (defactorizing labels post classification)
def keep_dict(Y_codes, Y_unique):
    dict = {}
    j = 0
    for i in range(len(Y_codes)):
        if Y_codes[i] in dict:
            continue
        else:
            dict[Y_codes[i]] = Y_unique[j]
            j += 1
    return dict

#Random Forest Classifier
def RF_train(feat, n_trees, max_feat):
    Y = feat.pop('type')
    X = feat
    Y_codes, Y_unique = pd.factorize(Y) #factorizing labels
    dict1 = keep_dict(Y_codes, Y_unique)

    # Make training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y_codes, test_size=0.25, random_state=42)

    # classify using Random Forest
    clf = RandomForestClassifier(n_estimators=int(n_trees), n_jobs=-1, random_state=25, max_features=str(max_feat),
                                 max_leaf_nodes=1500, oob_score=True, max_depth=None, min_samples_leaf=1)
    #fitting data using the classifier
    clf.fit(X_train, y_train)
    return clf, X_test, y_test, dict1


#Extra Trees Classifier
def Xtra(feat, n_trees, max_feat):
    Y = feat.pop('type')
    X = feat
    Y_codes, Y_unique = pd.factorize(Y) #factorizing labels
    dict2 = keep_dict(Y_codes, Y_unique)

    # Make training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y_codes, test_size=0.25, random_state=42)

    # classify using Extra Trees Classifier
    clf = ExtraTreesClassifier(n_estimators=n_trees, n_jobs=-1, random_state=0, max_leaf_nodes=1500,
                               max_features=str(max_feat), oob_score=True, max_depth=15, min_samples_leaf=1,
                               bootstrap=True)
    #fitting data using classifier
    clf.fit(X_train, y_train)
    return clf, X_test, y_test, dict2


#Gradient Boosting Classifier
def GB(feat, n_est, max_feat, lrate):
    Y = feat.pop('type')
    X = feat
    Y_codes, Y_unique = pd.factorize(Y) #factorizing labels
    dict3 = keep_dict(Y_codes, Y_unique)

    # Make training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y_codes, test_size=0.25, random_state=42)

    #classify using GB
    gb = GradientBoostingClassifier(loss='deviance', n_estimators=n_est, learning_rate=float(lrate),
                                    max_features=str(max_feat), max_depth=None, max_leaf_nodes=81, random_state=9,
                                    subsample=0.5)
    #fitting data using classifier
    gb.fit(X_train, y_train)
    return gb, X_test, y_test, dict3


#Convolutional Neural Networks
def CNN(dataset_loc, epoch, val_split):

    #Creating the model
    def create_model():
        model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(6, activation='softmax')])

    #Compiling Model using optimizer and loss functions
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model

    #Defining class labels
    class_labels = np.array(['Crazing', 'Inclusion', 'Patches', 'Pitted Surface', 'RS', 'Scratch'])

    #Setting up directory and validation split for the dataset
    data_dir = dataset_loc
    val_split = val_split
    dataset_image_generator = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True,
                                                 validation_split=val_split)

    #Accessing directories to get images
    data_Cr_dir = os.path.join(data_dir, 'Crazing')  # directory with our Cr defect pictures
    data_In_dir = os.path.join(data_dir, 'Inclusion')  # directory with our In defect pictures
    data_Pa_dir = os.path.join(data_dir, 'Patches')  # directory with our Pa defect pictures
    data_Ps_dir = os.path.join(data_dir, 'Pitted Surface')  # directory with our Ps defect pictures
    data_Rs_dir = os.path.join(data_dir, 'RS')  # directory with our Rs pictures
    data_Sc_dir = os.path.join(data_dir, 'Scratch')  # directory with our Sc defect pictures

    #Setting up batch size and image parameters
    batch_size_train = 600
    batch_size_test = 400
    epochs = epoch
    IMG_HEIGHT = 64
    IMG_WIDTH = 64

    #Generating training and test dataset
    train_data_gen = dataset_image_generator.flow_from_directory(batch_size=batch_size_train, directory=data_dir,
                                                                 subset="training", shuffle=True,
                                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                 class_mode='categorical')
    val_data_gen = dataset_image_generator.flow_from_directory(batch_size=batch_size_test, directory=data_dir,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='categorical', subset="validation")

    model = create_model()

    #  ******for saving model if necessary******
    # filepath = "D:/Work/Academics/AME 505-Engineering Information Modelling/Project/CNN.h5"
    # model.save(filepath, overwrite=True, include_optimizer=True)

    #Generating history of the model and fitting dataset
    history = model.fit(
        train_data_gen,
        steps_per_epoch=batch_size_train,
        epochs=epochs,
        validation_data=val_data_gen, validation_steps=batch_size_test)

    #Getting validation accuracy
    val_acc = history.history['val_accuracy']

    return val_acc, model, val_data_gen


#Load Pretrained CNN Model
def pretrained_CNN(data_dir):

    #Defining Class Labels
    class_labels = np.array(['Crazing', 'Inclusion', 'Patches', 'Pitted Surface', 'RS', 'Scratch'])

    # give validation split here
    val_split = 0.2


    #Training and test data generation with needed batch size
    dataset_image_generator = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True,
                                                 validation_split=val_split)

    data_Cr_dir = os.path.join(data_dir, 'Crazing')  # directory with our Cr defect pictures
    data_In_dir = os.path.join(data_dir, 'Inclusion')  # directory with our In defect pictures
    data_Pa_dir = os.path.join(data_dir, 'Patches')  # directory with our Pa defect pictures
    data_Ps_dir = os.path.join(data_dir, 'Pitted Surface')  # directory with our Ps defect pictures
    data_Rs_dir = os.path.join(data_dir, 'RS')  # directory with our Rs pictures
    data_Sc_dir = os.path.join(data_dir, 'Scratch')  # directory with our Sc defect pictures

    batch_size_train = 600
    batch_size_test = 400
    IMG_HEIGHT = 64
    IMG_WIDTH = 64

    #Creating a model
    def create_model():
        model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
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

    train_data_gen = dataset_image_generator.flow_from_directory(batch_size=batch_size_train, directory=data_dir,
                                                                 subset="training", shuffle=True,
                                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                 class_mode='categorical')
    val_data_gen = dataset_image_generator.flow_from_directory(batch_size=batch_size_test, directory=data_dir,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='categorical', subset="validation")

    #Model creation to load the checkpoint
    new_model = create_model()

    #*************************Loading Checkpoint Path***********************************#
    check_path = "/Users/Shaz/Google Drive/AME505Project/AME 505_Final/cp2.ckpt" #Need to specify the .ckpt file location

    #Loading weights from the checkpoint
    new_model.load_weights(check_path)

    #Getting loss and accuracy values
    loss, acc = new_model.evaluate(val_data_gen)
    return acc, new_model

#Predicting a new image or dataset of images
def pred(clf, X_test):
    Y_pred = clf.predict(X_test)
    return Y_pred

#Displaying the validation accuracy of an algorithm
def display_results(Y_pred, y_test):
    return accuracy_score(y_test, Y_pred)
