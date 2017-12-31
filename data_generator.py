import csv
import cv2
import numpy as np
import sklearn
import global_def as gf
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# store each line from the GroundTruth.csv from the ISIC training data
def read_gt_data():
    lines = []

    # Open the GroundTruth.csv file as csvfile
    with open(gf.data_folder+'ISIC-2017_Training_Part3_GroundTruth.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if 'ISIC_' in line[0]:
                lines.append(line)
    print('Number of lines read so far:', len(lines))

    return lines



# This function reads the numbers of records from the log file as specified by batch size (which is 32 by default)
# essentially 5*batch_size training records





def generator(samples, batch_size=32):
    label_binarizer = LabelBinarizer()
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            target = []
            msm_init_count = len(target)
            for batch_sample in batch_samples:

                # Read the center image (first one from the parsed list)
                image_name = batch_sample[0]
                image_with_full_name = gf.target_train_dir + image_name + gf.target_img_ext
                # print('Image with full name', image_with_full_name)
                #imageMat = cv2.imread(image_with_full_name)
                imageMat = image.load_img(image_with_full_name, target_size=(229, 229))
                # print('image shape', imageMat.shape)
                detected_value = float(batch_sample[1])
                if (detected_value > 0):
                    detect_arr = [1, 0]
                else:
                    detect_arr = [0, 1]
                images.append(imageMat)
                target.append(detect_arr)
                # print(image_name, detect_arr)

            # Convert the images into numpy array
            #x = np.array(images);
            x = np.expand_dims(x, axis=0)
            X_train = preprocess_input(x)

            y_train = np.array(target)
            #y_one_hot = label_binarizer.fit_transform(y_train)
            #print(offset)

            yield sklearn.utils.shuffle(X_train, y_train)