<h1>PetFace_ClassificationModel Using ResNet50</h1>

<h2>Description</h2>
<p>This project is all about classifying pet images using a deep learning model based on ResNet50. We take a bunch of pet images, preprocess them, and train a model to recognize different breeds. The goal is to make an accurate classifier using TensorFlow and Keras while applying data augmentation and feature extraction techniques.</p>

<h2>Features</h2>
<p>Loads and preprocesses images automatically</p>
<p>Uses data augmentation to improve model generalization</p>
<p>Leverages the pre-trained ResNet50 model for feature extraction</p>
<p>Trains a neural network to classify pet breeds</p>
<p>Evaluates the model and visualizes accuracy/loss trends</p>

<h2>Installation</h2>
<p>Before running the code, install the required Python packages:</p>
<p>pip install pandas, matplotlib, glob2 os-sys, numpy, tensorflow, scikit-learn</p>

<h2>Dataset Setup</h2>
<P>Place all pet images in a folder called images. Make sure the filenames follow this format:</P>
<P><pet_breed>_<image_number>.jpg</P>
<p>Example : bengal_01.jpg ,
chihuahua_02.jpg (This naming convention helps in extracting breed labels easily.)</p>

<h2>How to Run</h2>
<P>1. Import necessary libraries</P>
<P>import numpy as np</P>
<P>import pandas as pd</P>
<P>import matplotlib.pyplot as plt</P>
<P>import tensorflow as tf</P>
<P>from tensorflow.keras.preprocessing.image import load_img, img_to_array</P>
<P>from sklearn.model_selection import train_test_split</P>
<P>from tensorflow.keras.applications import ResNet50</P>
<P>from tensorflow.keras.applications.resnet50 import preprocess_input as pp_i</P>
<P>from tensorflow.keras.models import Model, Sequential</P>
<P>from tensorflow.keras.layers import RandomFlip, RandomRotation, Dense, Dropout, Input</P>
<P>from tensorflow.keras.losses import CategoricalCrossentropy</P>
<P>from tensorflow.keras.optimizers import Adam</P>
<P>import glob</P>
<P>import os</P>


# Pet Image Classification using ResNet50

## Overview
This project is all about classifying pet images using a deep learning model based on ResNet50. We take a bunch of pet images, preprocess them, and train a model to recognize different breeds. The goal is to make an accurate classifier using TensorFlow and Keras while applying data augmentation and feature extraction techniques.

## Features
- Loads and preprocesses images automatically
- Uses data augmentation to improve model generalization
- Leverages the pre-trained ResNet50 model for feature extraction
- Trains a neural network to classify pet breeds
- Evaluates the model and visualizes accuracy/loss trends

## Installation
Before running the code, install the required Python packages:
```bash
pip install pandas matplotlib glob2 os-sys numpy tensorflow scikit-learn
```

## Dataset Setup
Place all pet images in a folder called `images`. Make sure the filenames follow this format:
```
<pet_breed>_<image_number>.jpg
```
Example:
```
bengal_01.jpg
chihuahua_02.jpg
```
This naming convention helps in extracting breed labels easily.

## How to Run
### 1. Import necessary libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_i
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, Dense, Dropout, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import glob
import os
```

### 2. Load and preprocess images
```python
images_fp = './images'
image_names = [os.path.basename(file) for file in glob.glob(os.path.join(images_fp, '*.jpg'))]
labels = [' '.join(name.split('_')[:-1]) for name in image_names]
```

### 3. Encode breed labels
```python
def label_encode(label):
    mapping = {
        'abyssinian': 0, 'american bulldog': 1, 'american pit bull terrier': 2,
        'basset hound': 3, 'Birman': 4, 'Bombay': 5, 'boxer': 6, 'chihuahua': 7,
        'Egyptian mau': 8, 'german shorthaired': 9, 'newfoundland': 10,
        'japanese chin': 11, 'english cocker spaniel': 12, 'Bengal': 13,
        'pomeranian': 14, 'english setter': 15, 'wheaten terrier': 16
    }
    return mapping.get(label, None)
```

### 4. Convert images to numerical format
```python
features, labels_encoded = [], []
IMAGE_SIZE = (224, 224)
for name in image_names:
    label = ' '.join(name.split('_')[:-1])
    label_code = label_encode(label)
    if label_code is not None:
        img = load_img(os.path.join(images_fp, name))
        img = tf.image.resize_with_pad(img_to_array(img, dtype='uint8'), *IMAGE_SIZE).numpy().astype('uint8')
        features.append(img)
        labels_encoded.append(label_code)
```

### 5. Split the dataset
```python
features_array = np.array(features)
labels_array = np.array(labels_encoded)
labels_one_hot = pd.get_dummies(labels_array)

x_train, x_test, y_train, y_test = train_test_split(features_array, labels_one_hot, test_size=0.3, random_state=38)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.5, random_state=4)
```

### 6. Build the deep learning model
```python
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.4)
])
prediction_layers = Dense(15, activation='softmax')
resnet_model = ResNet50(include_top=False, pooling='avg', weights='imagenet')
resnet_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = pp_i(x)
x = resnet_model(x, training=False)
x = Dropout(0.2)(x)
outputs = prediction_layers(x)
model = Model(inputs, outputs)
```

### 7. Compile and train the model
```python
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
model_history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=15)
```

### 8. Plot accuracy and loss trends
```python
acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs_range = range(15)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

### 9. Evaluate and predict
```python
model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
```

## Results
After training for 15 epochs, the model achieves a decent accuracy. You can improve it further by fine-tuning the ResNet50 layers or experimenting with different hyperparameters.

## Future Improvements
- Add real-time classification using a webcam
- Deploy as a web app using Flask or FastAPI
- Try other models like EfficientNet or MobileNet

## Author
[Your Name]

## License
This project is open-source under the MIT License.













