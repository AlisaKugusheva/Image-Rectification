import numpy as np
import cv2
from skimage.transform import radon
import matplotlib.pyplot as plt
import os
import time

def rotate(PATH, resize=None):
    
    img = cv2.imread(PATH)
    t = time.time()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    height, width = img_gray.shape
    if resize is not None:
        img_gray = cv2.resize(img_gray, resize)
    degrees = np.arange(180)
    img_gray = img_gray - np.mean(img_gray)
    sinogram = radon(img_gray, theta=degrees)  # radon transform

    # using variance as it was suggested in the paper
    angle = np.argmax(np.array([np.var(line)
                                for line in sinogram.transpose()]))
    angle_to_rotate = 90 - angle

    M = cv2.getRotationMatrix2D(
        (width / 2, height / 2), angle_to_rotate, 1)  # rotation matrix
    dst = cv2.warpAffine(img, M, (width, height))

    name_file = f'{os.path.splitext(PATH)[0]}_rotated{os.path.splitext(PATH)[1]}'

    cv2.imwrite(name_file, dst)
    elapsed = time.time() - t
   
    return sinogram, angle_to_rotate, img, dst, elapsed
