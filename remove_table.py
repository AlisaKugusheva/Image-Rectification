import numpy as np
import cv2

from os import listdir
from os.path import isfile, join

from PIL import Image
import matplotlib.pyplot as plt


def remove(img, kernel_horizontal=(1, 30), kernel_vertical=(30, 1)):

    kernel = np.ones(kernel_horizontal)
    horizontal = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones(kernel_vertical)
    vertical = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    table = np.minimum(vertical, horizontal)

    gray_table = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_img_inv = cv2.bitwise_not(gray_img)
    gray_table_inv = cv2.bitwise_not(gray_table)
    inv_img_no_table = gray_img_inv - gray_table_inv
    img_no_table = cv2.bitwise_not(inv_img_no_table)

    return img, table, img_no_table
