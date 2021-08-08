#Importing .py files for GLCM and LBGLCM along with classifier
import GLCM_for_single_image, LBGLCM_for_single_image, Classifiers

#Importing numpy and keras
import numpy as np
from keras_preprocessing import image

#Extracting Features for single image
def extract(selected_classifier, directory_of_image):
    if 'GLCM' in selected_classifier:
        GLCM_feats = GLCM_for_single_image.extract_features(directory_of_image, angle= 0, dist= 1.25)
        return GLCM_feats
    else:
        LBGLCM_feats = LBGLCM_for_single_image.extract_features(directory_of_image, angle=0, dist= 1.25, radius= 1.2)
        return LBGLCM_feats

#Classifying the image using the selected classifier on the operator window
def classify(selected_classifier, directory_of_image,trained_classifiers, labels):
    if selected_classifier == 'GLCM+Random Forest':
        feat = extract(selected_classifier, directory_of_image)
        Ans = Classifiers.pred(trained_classifiers[0], feat)
        dict = labels[0]
        return dict[Ans[0]] 

    if selected_classifier == "LBGLCM + Random Forest":
       feat = extract(selected_classifier,directory_of_image)
       Ans = Classifiers.pred(trained_classifiers[1], feat)
       dict = labels[1]
       return dict[Ans[0]]

    if selected_classifier == "GLCM + Extra Trees Classifier":
        feat = extract(selected_classifier,directory_of_image)
        Ans = Classifiers.pred(trained_classifiers[2], feat)
        dict = labels[2]
        return dict[Ans[0]]

    if selected_classifier == "LBGLCM + Extra Trees Classifier":
        feat = extract(selected_classifier,directory_of_image)
        Ans = Classifiers.pred(trained_classifiers[3], feat)
        dict = labels[3]
        return dict[Ans[0]]

    if selected_classifier == "GLCM + Gradient Boosting Classifier":
        feat = extract(selected_classifier,directory_of_image)
        Ans = Classifiers.pred(trained_classifiers[4], feat)
        dict = labels[4]
        return dict[Ans[0]]

    if selected_classifier == "LBGLCM + Gradient Boosting Classifier":
        feat = extract(selected_classifier,directory_of_image)
        Ans = Classifiers.pred(trained_classifiers[5], feat)
        dict = labels[5]
        return dict[Ans[0]]

    if selected_classifier == 'Convolutional Neural Networks':
        test_image = image.load_img(directory_of_image, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image /= 255.
        Ans = Classifiers.pred(trained_classifiers[6], test_image)
        final_ans = Ans[0]
        dict = {}
        dict[0] = 'Crazing'
        dict[1] = 'Inclusion'
        dict[2] = 'Patches'
        dict[3] = 'Pitted Surface'
        dict[4] = 'RS'
        dict[5] = 'Scratch'
        return dict[np.argmax(final_ans)]




