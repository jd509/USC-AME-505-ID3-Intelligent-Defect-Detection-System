# ID3: Intelligent Steel Surface Defect Detection System using Supervised and Unsupervised Learning Techniques
The project was a part of the AME 505 course at USC. the objective of the project is to create an application to defect steel surface defects using supervised and unsupervised learning methods.

## Dependencies
The following dependencies need to be installed before running the project.
- [Docker](https://docs.docker.com/get-docker/) - framework for running the application on any platforms.
- [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) - Linux shell platform. (Only for Windows Users)


## Installation
 
Clone the master branch of the repository in your selected destination using 

```bash
git clone -b master https://github.com/jd509/USC-AME-505-ID3-Intelligent-Defect-Detection-System.git
```

Source into the cloned folder and run the install.sh script to generate a docker image for the project.

```bash
./install.sh
```
This will generate the docker image with the necessary dependencies. 

## Usage

To run the application:

Run the launch_app.sh script.
```bash
./launch_app.sh
```
This will start a docker container with the name "defect_detector" which will load and run the application. 

![User Interface for ID3](https://github.com/jd509/USC-AME-505-ID3-Intelligent-Defect-Detection-System/blob/master/images/user_interface_entrypoint.png)

The user can specify the input values for all the machine learning models and train the models individually.

To read further about these models:

### Feature Extraction Methods:

- [Grey-Level Co-occurence Matrix](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html
) - Feature extraction method using GLCM algorithm.

- [Local Binary Patterns + Grey-Level Co-occurence Matrix](https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b
) - Feature extraction method using LBGLCM algorithm.

### Machine Learning Models:

- [Random Forest Classifier](https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/) 

- [Extra Trees Classifier](https://www.geeksforgeeks.org/ml-extra-tree-classifier-for-feature-selection/)

- [Gradient Boosting Model](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)

- [Convolutional Neural Networks](https://www.tensorflow.org/tutorials/images/cnn)

The training results can be viewed on the user interface after all the models have been trained.

![Sample Training Results for Machine Learning Models](https://github.com/jd509/USC-AME-505-ID3-Intelligent-Defect-Detection-System/blob/master/images/sample_accuracy_results.png)

Post-training, the user interface can be used to predict the classification label for a given image and identify the defect associated with it.

![Defect Predictor](https://github.com/jd509/USC-AME-505-ID3-Intelligent-Defect-Detection-System/blob/master/images/defect_prediction.png)


## Inside the Repo

The repository contains python scripts to train and test the model. It also contains the dataset on which the model was trained.

### Scripts
- train_machine_learning_models.py : Python script to train the models and extract features.
- predict_defects.py : Python script to predict the defect for a single image using trained models.
- user_interface.py : UI for interacting with the application.

### Configuration Files
- image_feature_extraction_config.json : JSON file for default parameters to extract features from training dataset
- machine_learning_params.json : JSON file for default parameters to train the machine learning models
### Folders
- train_dataset: Contains the dataset for six types of surface defects : Crazing, Inclusion, Patches, Pitted Surfaces, Scratches, Residues
- test_dataset: Sample images to test the model
  
## Known Issues
Currently, there are no known issues to run the application. However, if any issues have been found, please open an issue forum for the same
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
