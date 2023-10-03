# Residual Neural Network for Earths Classification with Bayesian Data Augmentation

We have implemented a Residual Convolutional Neural Network (ResNet) for Earth's dataset classification tasks. ResNet effectively addresses challenges related to the vanishing gradient problem, which enables us to construct deeper networks with more layers, thus making the algorithm easier to train. In order to efficiently utilize the available computational resources, we have incorporated a dimensional reduction technique, similar to Principal Component Analysis (PCA).

Additionally, we have adopted a data augmentation approach to augment the dataset. This strategy allows us to generate more diverse images for training the algorithm effectively. Consequently, we have selected nine hyperparameters that are associated with model configurations and data augmentation operations.

To ensure that the chosen hyperparameters are optimal, we applied a Bayesian hyperparameter optimization approach. This approach has proven effective in enhancing model performance and accuracy.

The experimental results indicate that these techniques, when applied to the ResNet architecture for Earth's dataset classification, contribute to achieving high accuracy and robust training performance, particularly when dealing with limited datasets and computational resources.

# Files
You will find two .ipynb files. The 'Pre_processing.ipynb' file contains code related to the preprocessing of the entire Earth's dataset, including tasks such as data enhancement, dimensionality reduction, and other necessary transformations. 

The second file, 'Earth_Classifier.ipynb,' is dedicated to handling the processed Earth's dataset. This includes tasks such as loading the enhanced images, conducting Exploratory Data Analysis (EDA) specific to Earth-related features, implementing Principal Component Analysis (PCA) for dimensionality reduction, applying Bayesian Optimization for hyperparameter tuning, and training a ResNet model. The final results, including classification performance, are also presented in this file.

Please note that while the example you provided originally referred to COVID-19 medical image classification, we have adapted it to a scenario involving the classification of Earth-related features or data. You can use the provided code as a template for your Earth's dataset classification task. Be sure to update the directories and specific details related to your Earth's dataset in the 'Earth_Classifier.ipynb' file to fit your specific project requirements.

# Computational Requeriments
This is a deep learning convolutional neural networK, so it will require good computational resources, in order to obtain the final result. The Bayesian Hyperparameter Optimization needs 2.79 available memory RAM while the ResNet model with the optimized hyperparameters needs 15.79 memory RAM. If this resources are not available for running the code you can use Google Colaboratory 

# Test Results
<img width="403" alt="image" src="https://github.com/A-Bitz/EARTH-RESNET/assets/118044372/cffb5bbe-99e5-4454-824a-1d6183937efa">

# Software Dependecies
This model was writen in Google Colab and some libraries were used or installed, such as:

[Matplotlib] (https://matplotlib.org/)
[Seaborn] (https://seaborn.pydata.org/)
[Scikit-Image] (https://scikit-image.org/)
[Keras] (https://keras.io/)
[TensorFlow] (https://www.tensorflow.org/?hl=es-419)
[Numpy] (https://numpy.org/)
Install bayesian-optimization, GPy and GpyOpt.

#  Classification
<img width="488" alt="image" src="https://github.com/A-Bitz/EARTH-RESNET/assets/118044372/ff8a704d-8b8f-4b70-94b5-2f702e6430f3">

Classification is a fundamental task in remote sensing data analysis, where the goal is to assign a semantic label to each image, such as 'urban', 'forest', 'agricultural land', etc. The process of assigning labels to an image is known as image-level classification. However, in some cases, a single image might contain multiple different land cover types, such as a forest with a river running through it, or a city with both residential and commercial areas. In these cases, image-level classification becomes more complex and involves assigning multiple labels to a single image. This can be accomplished using a combination of feature extraction and machine learning algorithms to accurately identify the different land cover types. It is important to note that image-level classification should not be confused with pixel-level classification, also known as semantic segmentation. While image-level classification assigns a single label to an entire image, semantic segmentation assigns a label to each individual pixel in an image, resulting in a highly detailed and accurate representation of the land cover types in an image.




