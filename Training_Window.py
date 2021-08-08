#importing modules
import os

#import GUI modules
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QImage, QPalette, QBrush
from PyQt5.QtWidgets import QFileDialog

#importing files for classifiers and feature extraction methods
import Classifiers
import GLCM
import LBGLCM

#importing other GUIs (testing window and training-result window
from Operator_Window import Ui_Dialog2
from Training_Result_Window import Ui_Dialog1

#declaring variables which would be used to collect accuracies of different algorithms, classifiers and image labels
accuracies = []
all_classifiers = []
labels = []


class Ui_Dialog(object):

#Method for opening training results window
    def opentrainresults(self):
        global accuracies
        self.window = QtWidgets.QDialog()
        self.ui = Ui_Dialog1()
        self.ui.setupUi(self.window)
        self.ui.glcmrf.setText(str(accuracies[0]))
        self.ui.lbglcmrf.setText(str(accuracies[1]))
        self.ui.glcmxt.setText(str(accuracies[2]))
        self.ui.lbglcmxt.setText(str(accuracies[3]))
        self.ui.glcmgb.setText(str(accuracies[4]))
        self.ui.lbglcmgb.setText(str(accuracies[5]))
        self.ui.cnn.setText(str(accuracies[6]))
        self.window.show()

#Method for oepning operator window
    def operatorwindow(self):
        self.window = QtWidgets.QDialog()
        self.ui = Ui_Dialog2()
        self.ui.setupUi(self.window)
        self.window.show()
        Ui_Dialog2.getclf(Ui_Dialog2, all_classifiers)
        Ui_Dialog2.getlabels(Ui_Dialog2, labels)

#Defining setup and other methods
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1140, 766)

        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setGeometry(QtCore.QRect(30, 80, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")

        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_10.setGeometry(QtCore.QRect(30, 280, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")

#Setting button actions for Random Forest
        self.TrainRF = QtWidgets.QPushButton(Dialog)
        self.TrainRF.setGeometry(QtCore.QRect(140, 460, 93, 28))
        self.TrainRF.setAutoDefault(False)
        self.TrainRF.setObjectName("TrainRF")
        self.TrainRF.clicked.connect(self.RandomTrees_GLCM)
        self.TrainRF.clicked.connect(self.RandomTrees_LBGLCM)

#Setting button actions for Extra Trees Classifiers
        self.TrainXtra = QtWidgets.QPushButton(Dialog)
        self.TrainXtra.setGeometry(QtCore.QRect(510, 460, 121, 31))
        self.TrainXtra.setAutoDefault(False)
        self.TrainXtra.setObjectName("TrainXtra")
        self.TrainXtra.clicked.connect(self.ExtraTrees_GLCM)
        self.TrainXtra.clicked.connect(self.ExtraTrees_LBGLCM)

#Setting button actions for Gradient Boosting
        self.TrainGB = QtWidgets.QPushButton(Dialog)
        self.TrainGB.setGeometry(QtCore.QRect(890, 470, 93, 28))
        self.TrainGB.setAutoDefault(False)
        self.TrainGB.setObjectName("TrainGB")
        self.TrainGB.clicked.connect(self.GB_GLCM)
        self.TrainGB.clicked.connect(self.GB_LBGLCM)

#Setting button actions for displaying training result
        self.displaytrainres = QtWidgets.QPushButton(Dialog)
        self.displaytrainres.setGeometry(QtCore.QRect(480, 630, 181, 51))
        self.displaytrainres.setStyleSheet("background-color: rgb(252, 1, 7);")
        self.displaytrainres.setAutoDefault(False)
        self.displaytrainres.setObjectName("displaytrainres")
        self.displaytrainres.clicked.connect(self.opentrainresults)

#Setting button actions for proceeding to operator window
        self.Proceedtoclass = QtWidgets.QPushButton(Dialog)
        self.Proceedtoclass.setGeometry(QtCore.QRect(870, 630, 221, 51))
        self.Proceedtoclass.setStyleSheet("background-color: rgb(51, 153, 102);")
        self.Proceedtoclass.setAutoDefault(False)
        self.Proceedtoclass.setObjectName("Proceedtoclass")
        self.Proceedtoclass.clicked.connect(self.operatorwindow)

#Setting up the layouts
        self.layoutWidget = QtWidgets.QWidget(Dialog)
        self.layoutWidget.setGeometry(QtCore.QRect(240, 30, 581, 41))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.FileLocation = QtWidgets.QLineEdit(self.layoutWidget)
        self.FileLocation.setObjectName("FileLocation")
        self.horizontalLayout.addWidget(self.FileLocation)
        self.Browse = QtWidgets.QPushButton(self.layoutWidget)
        self.Browse.setAutoDefault(True)
        self.Browse.setObjectName("Browse")
        self.horizontalLayout.addWidget(self.Browse)
        self.layoutWidget1 = QtWidgets.QWidget(Dialog)
        self.layoutWidget1.setGeometry(QtCore.QRect(50, 120, 291, 131))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.AngleforGLCM = QtWidgets.QLineEdit(self.layoutWidget1)
        self.AngleforGLCM.setObjectName("AngleforGLCM")
        self.gridLayout.addWidget(self.AngleforGLCM, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)
        self.DistanceforGLCM = QtWidgets.QLineEdit(self.layoutWidget1)
        self.DistanceforGLCM.setObjectName("DistanceforGLCM")
        self.gridLayout.addWidget(self.DistanceforGLCM, 2, 1, 1, 1)
        self.layoutWidget2 = QtWidgets.QWidget(Dialog)
        self.layoutWidget2.setGeometry(QtCore.QRect(690, 120, 401, 121))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.AngleforLBGLCM = QtWidgets.QLineEdit(self.layoutWidget2)
        self.AngleforLBGLCM.setObjectName("AngleforLBGLCM")
        self.gridLayout_2.addWidget(self.AngleforLBGLCM, 2, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 1, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 1, 2, 1, 1)
        self.RadiusforLBGLCM = QtWidgets.QLineEdit(self.layoutWidget2)
        self.RadiusforLBGLCM.setObjectName("RadiusforLBGLCM")
        self.gridLayout_2.addWidget(self.RadiusforLBGLCM, 1, 1, 1, 1)
        self.DistanceforLBGLCM = QtWidgets.QLineEdit(self.layoutWidget2)
        self.DistanceforLBGLCM.setObjectName("DistanceforLBGLCM")
        self.gridLayout_2.addWidget(self.DistanceforLBGLCM, 1, 3, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 1, 1, 1)
        self.layoutWidget3 = QtWidgets.QWidget(Dialog)
        self.layoutWidget3.setGeometry(QtCore.QRect(50, 340, 291, 111))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.layoutWidget3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_13 = QtWidgets.QLabel(self.layoutWidget3)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout_3.addWidget(self.label_13, 2, 0, 1, 1)
        self.notreesRF = QtWidgets.QLineEdit(self.layoutWidget3)
        self.notreesRF.setObjectName("notreesRF")
        self.gridLayout_3.addWidget(self.notreesRF, 1, 1, 1, 1)
        self.FeaturesRF = QtWidgets.QComboBox(self.layoutWidget3)
        self.FeaturesRF.setFrame(True)
        self.FeaturesRF.setObjectName("FeaturesRF")
        self.FeaturesRF.addItem("")
        self.FeaturesRF.addItem("")
        self.FeaturesRF.addItem("")
        self.gridLayout_3.addWidget(self.FeaturesRF, 2, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.layoutWidget3)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.gridLayout_3.addWidget(self.label_12, 1, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.layoutWidget3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.gridLayout_3.addWidget(self.label_11, 0, 0, 1, 2)
        self.layoutWidget4 = QtWidgets.QWidget(Dialog)
        self.layoutWidget4.setGeometry(QtCore.QRect(420, 340, 291, 111))
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.layoutWidget4)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.NotreesXtra = QtWidgets.QLineEdit(self.layoutWidget4)
        self.NotreesXtra.setObjectName("NotreesXtra")
        self.gridLayout_4.addWidget(self.NotreesXtra, 1, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.layoutWidget4)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.gridLayout_4.addWidget(self.label_15, 1, 0, 1, 1)
        self.FeaturesXtra = QtWidgets.QComboBox(self.layoutWidget4)
        self.FeaturesXtra.setObjectName("FeaturesXtra")
        self.FeaturesXtra.addItem("")
        self.FeaturesXtra.addItem("")
        self.FeaturesXtra.addItem("")
        self.gridLayout_4.addWidget(self.FeaturesXtra, 2, 1, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.layoutWidget4)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.gridLayout_4.addWidget(self.label_16, 2, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.layoutWidget4)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.gridLayout_4.addWidget(self.label_14, 0, 0, 1, 2)
        self.layoutWidget5 = QtWidgets.QWidget(Dialog)
        self.layoutWidget5.setGeometry(QtCore.QRect(790, 340, 301, 121))
        self.layoutWidget5.setObjectName("layoutWidget5")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.layoutWidget5)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.Estimators_gb = QtWidgets.QLineEdit(self.layoutWidget5)
        self.Estimators_gb.setObjectName("Estimators_gb")
        self.gridLayout_5.addWidget(self.Estimators_gb, 1, 1, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.layoutWidget5)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.gridLayout_5.addWidget(self.label_18, 1, 0, 1, 1)
        self.Features_gb = QtWidgets.QComboBox(self.layoutWidget5)
        self.Features_gb.setObjectName("Features_gb")
        self.Features_gb.addItem("")
        self.Features_gb.addItem("")
        self.Features_gb.addItem("")
        self.gridLayout_5.addWidget(self.Features_gb, 2, 1, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.layoutWidget5)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.gridLayout_5.addWidget(self.label_19, 2, 0, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.layoutWidget5)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.gridLayout_5.addWidget(self.label_17, 0, 0, 1, 2)
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 2)
        self.label_20 = QtWidgets.QLabel(self.layoutWidget5)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.gridLayout_6.addWidget(self.label_20, 1, 0, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.layoutWidget5)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout_6.addWidget(self.lineEdit_4, 1, 1, 1, 1)
        self.Train_CNN = QtWidgets.QPushButton(Dialog)
        self.Train_CNN.setGeometry(QtCore.QRect(40, 710, 93, 28))
        self.Train_CNN.setAutoDefault(False)
        self.Train_CNN.setObjectName("Train_CNN")
        self.layoutWidget6 = QtWidgets.QWidget(Dialog)
        self.layoutWidget6.setGeometry(QtCore.QRect(40, 530, 291, 171))
        self.layoutWidget6.setObjectName("layoutWidget6")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.layoutWidget6)
        self.gridLayout_7.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_21 = QtWidgets.QLabel(self.layoutWidget6)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.gridLayout_7.addWidget(self.label_21, 0, 0, 1, 2)
        self.label_22 = QtWidgets.QLabel(self.layoutWidget6)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.gridLayout_7.addWidget(self.label_22, 1, 0, 1, 1)
        self.epochs = QtWidgets.QLineEdit(self.layoutWidget6)
        self.epochs.setObjectName("epochs")
        self.gridLayout_7.addWidget(self.epochs, 1, 1, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.layoutWidget6)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.gridLayout_7.addWidget(self.label_23, 2, 0, 1, 1)
        self.validation_split = QtWidgets.QLineEdit(self.layoutWidget6)
        self.validation_split.setObjectName("validation_split")
        self.gridLayout_7.addWidget(self.validation_split, 2, 1, 1, 1)
        self.Pretrainmodel = QtWidgets.QPushButton(Dialog)
        self.Pretrainmodel.setGeometry(QtCore.QRect(190, 710, 141, 31))
        self.Pretrainmodel.setAutoDefault(False)
        self.Pretrainmodel.setObjectName("Pretrainmodel")
        self.topBrowseHorizLine = QtWidgets.QFrame(Dialog)
        self.topBrowseHorizLine.setGeometry(QtCore.QRect(20, 10, 1101, 21))
        self.topBrowseHorizLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.topBrowseHorizLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.topBrowseHorizLine.setObjectName("topBrowseHorizLine")
        self.bottomBrowseHorizLine = QtWidgets.QFrame(Dialog)
        self.bottomBrowseHorizLine.setGeometry(QtCore.QRect(20, 70, 1101, 21))
        self.bottomBrowseHorizLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.bottomBrowseHorizLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.bottomBrowseHorizLine.setObjectName("bottomBrowseHorizLine")
        self.bottomFEHorizLine = QtWidgets.QFrame(Dialog)
        self.bottomFEHorizLine.setGeometry(QtCore.QRect(20, 270, 1101, 16))
        self.bottomFEHorizLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.bottomFEHorizLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.bottomFEHorizLine.setObjectName("bottomFEHorizLine")
        self.bottomClassifierHorizLine = QtWidgets.QFrame(Dialog)
        self.bottomClassifierHorizLine.setGeometry(QtCore.QRect(20, 500, 1101, 21))
        self.bottomClassifierHorizLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.bottomClassifierHorizLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.bottomClassifierHorizLine.setObjectName("bottomClassifierHorizLine")
        self.windowLeftVertLine = QtWidgets.QFrame(Dialog)
        self.windowLeftVertLine.setGeometry(QtCore.QRect(3, 20, 31, 731))
        self.windowLeftVertLine.setFrameShape(QtWidgets.QFrame.VLine)
        self.windowLeftVertLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.windowLeftVertLine.setObjectName("windowLeftVertLine")
        self.windowRightVertLine = QtWidgets.QFrame(Dialog)
        self.windowRightVertLine.setGeometry(QtCore.QRect(1100, 20, 41, 731))
        self.windowRightVertLine.setFrameShape(QtWidgets.QFrame.VLine)
        self.windowRightVertLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.windowRightVertLine.setObjectName("windowRightVertLine")
        self.bottomCNNHorizLine = QtWidgets.QFrame(Dialog)
        self.bottomCNNHorizLine.setGeometry(QtCore.QRect(20, 740, 1101, 21))
        self.bottomCNNHorizLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.bottomCNNHorizLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.bottomCNNHorizLine.setObjectName("bottomCNNHorizLine")

        self.retranslateUi(Dialog)
        self.Browse.clicked.connect(self.browseSlot)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        #Setting button actions for CNN model
        self.Pretrainmodel.clicked.connect(self.load_pretrained_model)
        self.Train_CNN.clicked.connect(self.CNN)


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
#Setting up the logo for GUI
        Dialog.setWindowTitle(_translate("Dialog", "ID3"))
        Dialog.setWindowIcon(QtGui.QIcon('logo.png'))

#Setting other text box labels
        self.label_9.setText(_translate("Dialog", "Feature Extraction"))
        self.label_10.setText(_translate("Dialog", "Classifiers"))
        self.TrainRF.setText(_translate("Dialog", "Train RF"))
        self.TrainXtra.setText(_translate("Dialog", "Train XtraTrees"))
        self.TrainGB.setText(_translate("Dialog", "Train GB"))
        self.displaytrainres.setText(_translate("Dialog", "Display Validation Accuracy"))
        self.Proceedtoclass.setText(_translate("Dialog", "Proceed to Classification"))
        self.label.setText(_translate("Dialog", "Dataset Location"))
        self.Browse.setText(_translate("Dialog", "Browse"))
        self.label_2.setText(_translate("Dialog", "GLCM"))
        self.label_3.setText(_translate("Dialog", "Angle"))
        self.label_4.setText(_translate("Dialog", "Distance"))
        self.label_5.setText(_translate("Dialog", "LBGLCM"))
        self.label_6.setText(_translate("Dialog", "Radius"))
        self.label_8.setText(_translate("Dialog", "Distance"))
        self.label_7.setText(_translate("Dialog", "Angle"))
        self.label_11.setText(_translate("Dialog", "Random Forest"))
        self.label_12.setText(_translate("Dialog", "No. of Trees"))
        self.label_13.setText(_translate("Dialog", "Max_Features"))
        self.FeaturesRF.setItemText(0, _translate("Dialog", "auto"))
        self.FeaturesRF.setItemText(1, _translate("Dialog", "sqrt"))
        self.FeaturesRF.setItemText(2, _translate("Dialog", "log2"))
        self.label_14.setText(_translate("Dialog", "Extra Trees Classifier"))
        self.label_15.setText(_translate("Dialog", "No. of Trees"))
        self.label_16.setText(_translate("Dialog", "Max_Features"))
        self.FeaturesXtra.setItemText(0, _translate("Dialog", "auto"))
        self.FeaturesXtra.setItemText(1, _translate("Dialog", "sqrt"))
        self.FeaturesXtra.setItemText(2, _translate("Dialog", "log2"))
        self.label_18.setText(_translate("Dialog", "No. of est "))
        self.label_19.setText(_translate("Dialog", "Max_Features"))
        self.Features_gb.setItemText(0, _translate("Dialog", "auto"))
        self.Features_gb.setItemText(1, _translate("Dialog", "sqrt"))
        self.Features_gb.setItemText(2, _translate("Dialog", "log2"))
        self.label_17.setText(_translate("Dialog", "Gradient Boosting"))
        self.label_20.setText(_translate("Dialog", "Learning Rate"))
        self.Train_CNN.setText(_translate("Dialog", "Train CNN"))
        self.label_21.setText(_translate("Dialog", "Convolutional Neural Networks"))
        self.label_22.setText(_translate("Dialog", "Epochs"))
        self.label_23.setText(_translate("Dialog", "Validation Split"))
        self.Pretrainmodel.setText(_translate("Dialog", "Pre-trained Model"))


#Method for browse button
    def browseSlot(self):
        folder_path = str(QFileDialog.getExistingDirectory())
        self.FileLocation.setText(folder_path)

#Method for computing GLCM
    def compute_GLCM(self):
        ang_glcm = self.AngleforGLCM.text()
        dist_glcm = self.DistanceforGLCM.text()
        loc_glcm = self.FileLocation.text()
        glcm_feat = GLCM.extract_features(loc_glcm, dist_glcm, ang_glcm)
        return glcm_feat

#Method for computing LBGLCM
    def compute_LBGLCM(self):
        ang_lbglcm = self.AngleforLBGLCM.text()
        dist_lbglcm = self.DistanceforLBGLCM.text()
        loc_lbglcm = self.FileLocation.text()
        rad_lbglcm = int(self.RadiusforLBGLCM.text())
        lbglcm_feat = LBGLCM.extract_features(loc_lbglcm, dist_lbglcm, ang_lbglcm, rad_lbglcm)
        return lbglcm_feat

#Method for training Random Forest with GLCM
    def RandomTrees_GLCM(self):
        global accuracies, all_classifiers, labels
        glcm_feat = self.compute_GLCM()
        n_trees = self.notreesRF.text()
        max_feats = self.FeaturesRF.currentText()
        clf, x_rf1, y_rf1, dict1 = Classifiers.RF_train(glcm_feat, n_trees, max_feats) #Collecting the trained classifier, x_test, y_test and labels
        Y_pred_rf1 = Classifiers.pred(clf, x_rf1) #Predicting the x_test labels
        acc_test = Classifiers.display_results(Y_pred_rf1, y_rf1) #accuracy of prediction
        all_classifiers.append(clf)
        accuracies.append(acc_test)
        labels.append(dict1)

#Method for training Random Forest with LBGLCM
    def RandomTrees_LBGLCM(self):
        global accuracies, all_classifiers, labels
        lbglcm_feat = self.compute_LBGLCM()
        n_trees = self.notreesRF.text()
        max_feats = self.FeaturesRF.currentText()
        clf, x_rf2, y_rf2, dict2 = Classifiers.RF_train(lbglcm_feat, n_trees, max_feats)#Collecting the trained classifier, x_test, y_test and labels
        Y_pred_rf2 = Classifiers.pred(clf, x_rf2)
        acc_test = Classifiers.display_results(Y_pred_rf2, y_rf2)
        all_classifiers.append(clf)
        accuracies.append(acc_test)
        labels.append(dict2)

#Method for training Extra Trees Classifiers with GLCM
    def ExtraTrees_GLCM(self):
        global accuracies, all_classifiers, labels
        glcm_feat = self.compute_GLCM()
        n_trees = int(self.NotreesXtra.text())
        max_feats = self.FeaturesXtra.currentText()
        clf, x_x1, y_x1, dict3 = Classifiers.Xtra(glcm_feat, n_trees, max_feats)#Collecting the trained classifier, x_test, y_test and labels
        Y_pred_x1 = Classifiers.pred(clf, x_x1)
        acc_test = Classifiers.display_results(Y_pred_x1, y_x1)
        all_classifiers.append(clf)
        accuracies.append(acc_test)
        labels.append(dict3)

#Method for training Extra Trees Classifiers with LBGLCM
    def ExtraTrees_LBGLCM(self):
        global accuracies, all_classifiers, labels
        lbglcm_feat = self.compute_LBGLCM()
        n_trees = int(self.NotreesXtra.text())
        max_feats = self.FeaturesXtra.currentText()
        clf, x_x2, y_x2, dict4 = Classifiers.Xtra(lbglcm_feat, n_trees, max_feats)#Collecting the trained classifier, x_test, y_test and labels
        Y_pred_x2 = Classifiers.pred(clf, x_x2)
        acc_test = Classifiers.display_results(Y_pred_x2, y_x2)
        all_classifiers.append(clf)
        accuracies.append(acc_test)
        labels.append(dict4)

#Method for training Gradient Boosting with GLCM
    def GB_GLCM(self):
        global accuracies, all_classifiers
        glcm_feat = self.compute_GLCM()
        n_est = int(self.Estimators_gb.text())
        max_feats = self.Features_gb.currentText()
        Lrate = float(self.lineEdit_4.text())
        clf, x_g1, y_g1, dict5 = Classifiers.GB(glcm_feat, n_est, max_feats, Lrate)#Collecting the trained classifier, x_test, y_test and labels
        Y_pred_gb1 = Classifiers.pred(clf, x_g1)
        acc_test = Classifiers.display_results(Y_pred_gb1, y_g1)
        all_classifiers.append(clf)
        accuracies.append(acc_test)
        labels.append(dict5)

#Method for training Gradient Boosting with LBGLCM
    def GB_LBGLCM(self):
        global accuracies, all_classifiers
        lbglcm_feat = self.compute_LBGLCM()
        n_est = int(self.Estimators_gb.text())
        max_feats = self.Features_gb.currentText()
        Lrate = float(self.lineEdit_4.text())
        clf, x_g2, y_g2, dict6 = Classifiers.GB(lbglcm_feat, n_est, max_feats, Lrate)#Collecting the trained classifier, x_test, y_test and labels
        Y_pred_g2 = Classifiers.pred(clf, x_g2)
        acc_test = Classifiers.display_results(Y_pred_g2, y_g2)
        all_classifiers.append(clf)
        accuracies.append(acc_test)
        labels.append(dict6)

#Method for training CNN
    def CNN(self):
        global accuracies, all_classifiers
        epoch = int(self.epochs.text())
        dataset_loc = self.FileLocation.text()
        val_split = float(self.validation_split.text())
        accuracy, clf, val_datagen = Classifiers.CNN(dataset_loc, epoch, val_split)#Collecting the test accuracy, trained classifier and y_test
        accuracies.append(accuracy[0])
        all_classifiers.append(clf)

#Method for loading pretrained model of CNN
    def load_pretrained_model(self):
        global accuracies, all_classifiers
        acc1, clf = Classifiers.pretrained_CNN(self.FileLocation.text())#Collecting accuracy and the trained classifier
        accuracies.append(acc1)
        all_classifiers.append(clf)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'appLogo-1.png')
    app.setWindowIcon(QIcon(path))
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
