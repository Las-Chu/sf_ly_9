import csv
import cv2
import numpy as np
import sklearn
import global_def as gf
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
import skimage

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# store each line from the GroundTruth.csv from the ISIC training data
def read_gt_data(limit=-1):
    lines = []

    # Open the GroundTruth.csv file as csvfile
    with open(gf.data_folder + gf.gt_file) as csvfile:
        reader = csv.reader(csvfile)
        num_lines = 0
        for line in reader:
            if gf.sCheckData in line[0]:
                lines.append(line)
                num_lines += 1
                if (limit > 0) and (num_lines > limit):
                    break
    print('Number of lines read so far:', len(lines))

    return lines

# store each line which is melenoma detected from the GroundTruth.csv from the ISIC training data
# this is used to create more samples of melenoma images for training purposes
def read_gt_m_data():
    lines = []

    # Open the GroundTruth.csv file as csvfile
    with open(gf.data_folder + gf.gt_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if gf.sCheckData in line[0] and line[1] == '1.0':
                lines.append(line)
                augment_image(line[0])
    print('Number of melanoma lines read so far:', len(lines))

    return lines

def augment_image(file_key_name):

    #compose the origfile name and dest file name
    orig_file = gf.data_folder + file_key_name + gf.target_img_ext
    #Load the file
    image_init = cv2.imread(orig_file)

    # Add Gaussian noise to images for each class type
    #fileNamewithFolder = gf.target_train_dir_aug + file_key_name + '_gn'+ gf.target_img_ext
    #noisy_image = skimage.util.random_noise(image_init)
    #cv2.imwrite(fileNamewithFolder, noisy_image)

    # Add sharpen images to the list as well
    #fileNamewithFolder = gf.target_train_dir_aug + file_key_name + '_gb' + gf.target_img_ext
    #blurred_f = cv2.GaussianBlur(image_init, (3, 3), 10.0)
    #sharpened = cv2.addWeighted(image_init, 2, blurred_f, -1, 0)
    #cv2.imwrite(fileNamewithFolder, sharpened)

    # Add rotation at 10
    fileNamewithFolder = gf.target_train_dir_aug + file_key_name + '_rtr' + gf.target_img_ext
    rows, cols, channels = image_init.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
    rotated = cv2.warpAffine(image_init, M, (cols, rows))
    cv2.imwrite(fileNamewithFolder, rotated)

    # Add rotation at -10
    fileNamewithFolder = gf.target_train_dir_aug + file_key_name + '_rtl' + gf.target_img_ext
    rows, cols, channels = image_init.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -15, 1)
    rotated1 = cv2.warpAffine(image_init, M, (cols, rows))
    cv2.imwrite(fileNamewithFolder, rotated1)

    # Add shift 100, 100 pixels
    fileNamewithFolder = gf.target_train_dir_aug + file_key_name + '_shr' + gf.target_img_ext
    rows, cols, channels = image_init.shape
    M = np.float32([[1, 0, 5], [0, 1, 5]])
    shifted = cv2.warpAffine(image_init, M, (cols, rows))
    cv2.imwrite(fileNamewithFolder, shifted)

    # Add shift -100, -100 pixels
    fileNamewithFolder = gf.target_train_dir_aug + file_key_name + '_shl' + gf.target_img_ext
    rows, cols, channels = image_init.shape
    M = np.float32([[1, 0, -5], [0, 1, -5]])
    shifted1 = cv2.warpAffine(image_init, M, (cols, rows))
    cv2.imwrite(fileNamewithFolder, shifted1)

    # Since most of the images are dark, use
    # Equilize hostograms to enhance the image
    fileNamewithFolder = gf.target_train_dir_aug + file_key_name + '_enh' + gf.target_img_ext
    img_yuv = cv2.cvtColor(image_init, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    enhanced_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(fileNamewithFolder, enhanced_img)


# For ISIC data, two columns are important for classification
# second colun notifies if it is melenoma or benign
# third column notifies if it is benign, whether it is Seborrheic keratosis (look a like to melenoma but not a cancer ) or not
# The returned values are as follows: 0 - Benign, 1 - Seborrheic keratosis, 2 - Melenoma
def measure_target_value_for_isic_data(prim, sub):
    return prim #2*prim+sub

# This function reads the numbers of records from the log file as specified by batch size (which is 32 by default)
# essentially 5*batch_size training records


def generator(samples, batch_size=32):
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(gf.lb_fit_array)
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
                # imageMat = cv2.imread(image_with_full_name)
                imageMat = image.load_img(image_with_full_name, target_size=(229, 229))
                x = image.img_to_array(imageMat)
                #print("xshape", x.shape)
                # print('image shape', imageMat.shape)
                if gf.sCheckData in batch_sample[0]:
                    detected_value = float(measure_target_value_for_isic_data(float(batch_sample[1]), float(batch_sample[2])))
                else:
                    detected_value = float(batch_sample[1])
                '''
                if (detected_value > 0):
                    detect_arr = [1, 0]
                else:
                    detect_arr = [0, 1]
                '''
                images.append(x)
                target.append(detected_value)
                # print(image_name, detect_arr)

            # Convert the images into numpy array
            #print("xshape2", images.shape)
            x = np.array(images)
            #print("xshape2", x.shape)
            x = preprocess_input(x)
            #print("xshape2", x.shape)
            X_train = x
            #print("xshape3", X_train.shape)
            Y_train = np.array(target)

            y_one_hot = label_binarizer.transform(Y_train)
            #print(Y_train, y_one_hot)

            yield sklearn.utils.shuffle(X_train, y_one_hot)
