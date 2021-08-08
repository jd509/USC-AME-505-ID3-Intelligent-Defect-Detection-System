#Loading Modules for GUI
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QWidget

#Loading Trained Classifiers
import Trained_Classifier_Predictions

#Loading the Final results window
from Final_Results_Window import Ui_Dialog3

#declaring variables to be used for storing the name of defect, directory of image under consideration, classifier selected, trained classifiers and labels
defect_name = None
directory_of_image = None
classifier_selected = None
trained_classifiers = []
labels_for_classifiers = []


class Ui_Dialog2(object):

#Method for displaying final classification result
    def finalresults(self):
        self.window = QtWidgets.QDialog()
        self.ui = Ui_Dialog3()
        self.ui.setupUi(self.window)
        self.ui.Namofclassifier.setText(classifier_selected)
        self.ui.Typeofdefect.setText(defect_name)
        pixmap = QtGui.QPixmap(directory_of_image)
        self.ui.Photo.setPixmap(pixmap.scaled(192, 192))
        self.window.show()

#Method for setting up the UI
    def setupUi(self, Dialog2):
        Dialog2.setObjectName("Dialog2")
        Dialog2.resize(592, 400)
        self.comboBox = QtWidgets.QComboBox(Dialog2)
        self.comboBox.setGeometry(QtCore.QRect(150, 130, 291, 61))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.comboBox.setFont(font)
        self.comboBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")

#Setting up button action for classifying image
        self.Classify = QtWidgets.QPushButton(Dialog2)
        self.Classify.setGeometry(QtCore.QRect(230, 270, 121, 41))
        self.Classify.setObjectName("Classify")
        self.Classify.clicked.connect(self.Classifies)
        self.Classify.clicked.connect(self.finalresults)

#Setting up layout
        self.layoutWidget = QtWidgets.QWidget(Dialog2)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 30, 531, 51))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.Imageloc = QtWidgets.QLineEdit(self.layoutWidget)
        self.Imageloc.setObjectName("Imageloc")
        self.gridLayout.addWidget(self.Imageloc, 0, 1, 1, 1)
        self.Browseforimage = QtWidgets.QPushButton(self.layoutWidget)
        self.Browseforimage.setObjectName("Browseforimage")
        self.Browseforimage.clicked.connect(self.openImage)
        self.gridLayout.addWidget(self.Browseforimage, 0, 2, 1, 1)

        self.retranslateUi(Dialog2)
        QtCore.QMetaObject.connectSlotsByName(Dialog2)

    def retranslateUi(self, Dialog2):
        _translate = QtCore.QCoreApplication.translate
        Dialog2.setWindowTitle(_translate("Dialog2", "ID3"))
        self.comboBox.setItemText(0, _translate("Dialog2", "GLCM+Random Forest"))
        self.comboBox.setItemText(1, _translate("Dialog2", "LBGLCM + Random Forest"))
        self.comboBox.setItemText(2, _translate("Dialog2", "GLCM + Extra Trees Classifier"))
        self.comboBox.setItemText(3, _translate("Dialog2", "LBGLCM + Extra Trees Classifier"))
        self.comboBox.setItemText(4, _translate("Dialog2", "GLCM + Gradient Boosting Classifier"))
        self.comboBox.setItemText(5, _translate("Dialog2", "LBGLCM + Gradient Boosting Classifier"))
        self.comboBox.setItemText(6, _translate("Dialog2", "Convolutional Neural Networks"))
        self.Classify.setText(_translate("Dialog2", "Classify"))
        self.label.setText(_translate("Dialog2", "Image Location:"))
        self.Browseforimage.setText(_translate("Dialog2", "Browse"))

#Setting up the browse button
    def openImage(self):
        folder_path = QFileDialog.getOpenFileNames()
        self.Imageloc.setText(str(folder_path[0][0]))

#Collecting trained classifiers from Training Window
    def getclf(self, clf):
        global trained_classifiers
        trained_classifiers = clf

#Collecting labels associated with each classifier
    def getlabels(self, labels):
        global labels_for_classifiers
        labels_for_classifiers = labels

#Classifying the image
    def Classifies(self):
        global trained_classifiers, defect_name, directory_of_image, classifier_selected, labels_for_classifiers
        classifier_selected = self.comboBox.currentText()
        directory_of_image = self.Imageloc.text()
        defect_name = Trained_Classifier_Predictions.classify(classifier_selected, directory_of_image, trained_classifiers, labels_for_classifiers)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog2 = QtWidgets.QDialog()
    ui = Ui_Dialog2()
    ui.setupUi(Dialog2)
    Dialog2.show()
    sys.exit(app.exec_())
