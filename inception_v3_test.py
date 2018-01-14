import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras.applications.inception_v3 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

import image_cropping
import data_generator

# Resize all images (uncomment if
# need to be resized)
image_cropping.resize_all_images()
#data_generator.read_gt_m_data()

'''
# Load Keras' ResNet50 model that was pre-trained against the ImageNet database
model = inception_v3.InceptionV3()

# Load the image file, resizing it to 224x224 pixels (required by this model)
img = image.load_img('malignant_tumor.jpg', target_size=(299,299))

# Convert the image to a numpy array
x = image.img_to_array(img)

# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis = 0)

# Scale the input image to the range used in the trained network
x =inception_v3.preprocess_input(x)

# Run the image through the deep neural network to make a prediction
predictions = model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = inception_v3.decode_predictions(predictions, top=9)

print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))

'''