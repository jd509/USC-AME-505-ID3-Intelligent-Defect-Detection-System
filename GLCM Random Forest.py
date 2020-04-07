import os
from PIL import Image

from skimage.feature import greycomatrix, greycoprops
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


# extract features and store them as dataframe from each photo in the directory
def extract_features(directory):
    # make list for each feature and a list to have all features
    features = {}
    names = ['Crazing','Inclusion','Patches','Pitted Surface','RS','Scratch']
    contrasts = []
    dissimilarities = []
    homo = []
    correlate = []
    energe = []
    type = []
    for naam in names:
        foldername = directory + '/' + naam
        for name in os.listdir(foldername):
            filename = foldername + '/' + name
            # load an image from file
            image = Image.open(filename)
            # convert the image pixels to a numpy array
            img = np.array(image.getdata()).reshape(image.size[0], image.size[1])
            #Calulate GLCM Features and Matrix
            gcom = greycomatrix(img, [1], [0], 256, symmetric=True, normed=True)
            contrast = greycoprops(gcom, prop='contrast')
            dissimilarity = greycoprops(gcom, prop='dissimilarity')
            homogeneity = greycoprops(gcom, prop='homogeneity')
            energy = greycoprops(gcom, prop='energy')
            correlation = greycoprops(gcom, prop='correlation')
            # store feature
            contrasts.append(contrast[0][0])
            dissimilarities.append(dissimilarity[0][0])
            homo.append(homogeneity[0][0])
            energe.append(energy[0][0])
            correlate.append(correlation[0][0])
            type.append(naam)
            print('>%s' % name)
    #Add features to dictionary of features
    features['contrast'] = contrasts
    features['dissimilarity'] = dissimilarities
    features['homogeneity'] = homo
    features['energy'] = energe
    features['correlation'] = correlate
    features['type'] = type
    #convert dictionary to dataframe
    df = pd.DataFrame(features)
    return df

directory = r'C:/Users/jaine/Desktop/Pyjaineel/Project/NEU-CLS-64/NEU-CLS-64/Datasets'
features = extract_features(directory)
print(features.head())
#features.to_csv('C:/Users/jaine/Desktop/GLCMv9.csv', sep = ',', index=None)
#Make X (features) and Y (Labels)
Y = features.pop('type')
X = features
Y_codes, Y_unique = pd.factorize(Y)
#Make training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X,Y_codes,test_size=0.25,random_state=42)

#classify using Random Forest
clf = RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state= 0, max_features='log2', max_leaf_nodes=1500, oob_score=True, max_depth=15, min_samples_leaf=1)
clf.fit(X_train, y_train)
Y_pred = clf.predict(X_test)
print(accuracy_score(y_test,Y_pred))
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test, Y_pred))
