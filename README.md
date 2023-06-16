# KulkasKita: Image Classification Model

### Summary
KulkasKita is a mobile application designed to help users manage their food ingredients. It offers several features to simplify the process of storing, classifying, and managing food records. This repository contains code for an image classification model using the MobileNetV2 architecture. The model is trained to classify images into 50 different classes.

### Dataset
The dataset used for training and validation is located in the following directories:

Training data: /content/drive/MyDrive/Dataset baru/train 50
Validation data: /content/drive/MyDrive/Dataset baru/val 50
The dataset consists of 50 classes, and there are a total of 18,588 training images and 4,764 validation images.

### Data Preparation
The image data is preprocessed using the MobileNetV2 preprocessing function. Data augmentation techniques are applied to the training data, including rotation, brightness adjustment, zooming, horizontal and vertical flipping, and shifting.

### Model Building
The MobileNetV2 architecture is used as the base model, with the pre-trained weights obtained from ImageNet. The top layer of the base model is removed, and a custom classification head is added on top. The classification head consists of a flatten layer, a dropout layer, a dense layer with ReLU activation, and a dense layer with softmax activation for multi-class classification.


### Model Training
The model is trained with the training data using the fit function. The training process is carried out for 32 epochs.

### Model Evaluation
The model's performance is evaluated based on accuracy and loss. The training and validation accuracy are plotted in a graph.
*graph*
The model achieves an accuracy of 0.88 on the validation set. Additional evaluation metrics such as precision score are also calculated.

### Model Inference
The model is used for image classification on unseen images. The user can upload an image and the model will predict its class. The predicted class is displayed along with the uploaded image.

### Model Deployment
The trained model is saved for deployment using TensorFlow Lite. The model is saved in the tflite format.

### Usage
To use the trained model for image classification, follow these steps:

1. Mount Google Drive to access the dataset:
from google.colab import drive
drive.mount('/content/drive')
Set the paths to the training and validation directories and
preprocess the data and create data generators for training and validation.

2. Build the MobileNetV2-based model for image classification.

3. Compile and train the model using the training data and evaluate it on the validation data.

4. Use the model for inference by uploading an image and running the prediction code.

5. Save the trained model for deployment using TensorFlow Lite.

Note: Some code snippets, such as mounting Google Drive and uploading files, may require running the code in a specific environment like Google Colab.

Feel free to modify and adapt the code according to your specific requirements.
