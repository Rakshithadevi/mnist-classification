# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
## Problem Statement:
The task at hand involves developing a Convolutional Neural Network (CNN) that can accurately classify handwritten digits ranging from 0 to 9. This CNN should be capable of processing scanned images of handwritten digits, even those not included in the standard dataset.

## Dataset:
The MNIST dataset is widely recognized as a foundational resource in both machine learning and computer vision. It consists of grayscale images measuring 28x28 pixels, each depicting a handwritten digit from 0 to 9. The dataset includes 60,000 training images and 10,000 test images, meticulously labeled for model evaluation. Grayscale representations of these images range from 0 to 255, with 0 representing black and 255 representing white. MNIST serves as a benchmark for assessing various machine learning models, particularly for digit recognition tasks. By utilizing MNIST, we aim to develop and evaluate a specialized CNN for digit classification while also testing its ability to generalize to real-world handwritten images not present in the dataset.
## Neural Network Model

![image](https://github.com/Rakshithadevi/mnist-classification/assets/94165326/2e8b7dc6-582c-42ed-a41b-9c39367d3dea)

## DESIGN STEPS

## STEP 1:
Preprocess the MNIST dataset by scaling the pixel values to the range [0, 1] and converting labels to one-hot encoded format.

## STEP 2:
Build a convolutional neural network (CNN) model with specified architecture using TensorFlow Keras.

## STEP 3:
Compile the model with categorical cross-entropy loss function and the Adam optimizer.

## STEP 4:
Train the compiled model on the preprocessed training data for 5 epochs with a batch size of 64.

## STEP 5:
Evaluate the trained model's performance on the test set by plotting training/validation metrics and generating a confusion matrix and classification report. Additionally, make predictions on sample images to demonstrate model inference.



## PROGRAM

### Name: RAKSHITHA DEVI
### Register Number:212221230082

```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print("Rakshitha Devi J","212221230082")
print(confusion_matrix(y_test,x_test_predictions))
print("Rakshitha Devi J","212221230082")
print(classification_report(y_test,x_test_predictions))

img = image.load_img('/content/hw no 1.png')
type(img)

img = image.load_img('/content/hw no 1.png')
plt.imshow(img)
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)

print("Rakshitha Devi J","212221230082")
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img1 = image.load_img('/content/hw no 1.png')
plt.imshow(img1)
img_28_gray_inverted1 = 255.0-img_28_gray1
img_28_gray_inverted_scaled1 = img_28_gray_inverted1.numpy()/255.0

x_single_prediction1 = np.argmax(model.predict(img_28_gray_inverted_scaled1.reshape(1,28,28,1)),axis=1)
print("Rakshitha Devi J","212221230082")
print(x_single_prediction1)

```
## OUTPUT
## Training data:
![image](https://github.com/Rakshithadevi/mnist-classification/assets/94165326/b505ca5d-b3f8-45f0-84e3-04a23b64aec5)


### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/Rakshithadevi/mnist-classification/assets/94165326/8d8c2ae1-e20a-48fe-acf6-9491f64e2f8b)
![image](https://github.com/Rakshithadevi/mnist-classification/assets/94165326/be1616b8-5cd7-4868-ade6-6834bf9c5ff9)


### Classification Report

![image](https://github.com/Rakshithadevi/mnist-classification/assets/94165326/3ffe91dd-a27d-439a-9ba1-552c45083e30)


### Confusion Matrix

![image](https://github.com/Rakshithadevi/mnist-classification/assets/94165326/2d2680c8-d8b8-4d16-8592-99dd26cadbe5)


### New Sample Data Prediction

![image](https://github.com/Rakshithadevi/mnist-classification/assets/94165326/2a0799b1-7e8b-465d-81aa-eea583255bf4)

![image](https://github.com/Rakshithadevi/mnist-classification/assets/94165326/ca30c4e3-db2d-46b2-9dba-6604a9059b01)

![image](https://github.com/Rakshithadevi/mnist-classification/assets/94165326/582cf37d-2f3c-4c53-baa8-75c1e6fa02fc)

![image](https://github.com/Rakshithadevi/mnist-classification/assets/94165326/3f38ce5b-e33f-4a12-b108-e5cc03df91bb)

![image](https://github.com/Rakshithadevi/mnist-classification/assets/94165326/53e3bb2f-4d3f-42ad-b6da-88a588a28d68)




## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed successfully.
