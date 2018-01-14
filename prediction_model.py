import numpy as np
import global_def as gf
import cv2
from keras.preprocessing import image
from keras.applications import inception_v3
from keras.applications.inception_v3 import preprocess_input

images = []
image_name = 'ISIC_0000025'
image_with_full_name = gf.target_train_dir + image_name + gf.target_img_ext
print('Image with full name', image_with_full_name)
imageMat = cv2.imread(image_with_full_name)
print(imageMat.shape)
#print('image shape', imageMat.shape)
test_sample = np.array(imageMat)
print(test_sample.shape)
ret_val = model.predict(test_sample[None, :, :, :], batch_size=1, verbose=1)

print(ret_val)