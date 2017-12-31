import cv2
import numpy as np

import global_def as gf

def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


# get ImageMat for the given file location in the Data log
def process_image(folder, imageField):
    imagePath = imageField
    fileName = imagePath.split('/')[-1]
    fileNamewithFolder = folder + '/' + fileName
    imageMat = cv2.imread(imageField)

    im_h, im_w, im_c = imageMat.shape
    #print(im_h, im_w, im_c)
    print(fileNamewithFolder, imageMat.shape)

    r_of_w = im_w / gf.small_width
    r_of_h = im_h / gf.small_height

    # See if need to be resized
    if ((gf.small_width == im_w) or (gf.small_height == im_h)):
        imageMat = imageMat
    else:
        r_of_w = 1.0 * im_w / gf.small_width
        r_of_h = 1.0 * im_h / gf.small_height

    if (r_of_h < r_of_w):
        resize_height = gf.small_height
        resize_width = int(resize_height * (1.0 * im_w / im_h))
    else:
        resize_width = gf.small_width
        resize_height = int(resize_width * (1.0 * im_h / im_w))

    imageMat = cv2.resize(imageMat, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    imageMat = crop_center(imageMat, gf.target_img_width, gf.target_img_height)
    cv2.imwrite(fileNamewithFolder, imageMat)

def ret_proc_image(image):
    imageMat = cv2.imread(imageField)

    im_h, im_w, im_c = imageMat.shape
    # print(im_h, im_w, im_c)
    print(fileNamewithFolder, imageMat.shape)

    r_of_w = im_w / gf.small_width
    r_of_h = im_h / gf.small_height

    # See if need to be resized
    if ((gf.small_width == im_w) or (gf.small_height == im_h)):
        imageMat = imageMat
    else:
        r_of_w = 1.0 * im_w / gf.small_width
        r_of_h = 1.0 * im_h / gf.small_height

    if (r_of_h < r_of_w):
        resize_height = gf.small_height
        resize_width = int(resize_height * (1.0 * im_w / im_h))
    else:
        resize_width = gf.small_width
        resize_height = int(resize_width * (1.0 * im_h / im_w))

    imageMat = cv2.resize(imageMat, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    imageMat = crop_center(imageMat, gf.target_img_width, gf.target_img_height)

    return imageMat

#process_image(gf.target_train_dir, '/home/ramesh/Data/isic2017_data/ISIC-2017_Training_Data/ISIC_0000000.jpg')

import glob
import time
import random

def resize_all_images():

    images = glob.glob('/home/ramesh/Data/isic2017_data/ISIC-2017_Orig_train_data/*.jpg')
    print(len(images))

    # Code that resizes the images from /home/ramesh/Data/isic2017_data/ISIC-2017_Training_Data/ and resizes them to 299x299 (smallest size of all images
    #
    for image in images:
        process_image(gf.target_train_dir, image)