import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
import scipy.ndimage
import xlrd
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import csv
import pandas

from PIL import Image
import numpy as np


def img_and_mask(img, mask_file, sp_or_ht):
    
    img_name = img[:-4]


    # Open the input image as numpy array
    npImage=np.array(Image.open(img))
    # Open the mask image as numpy array
    npMask=np.array(Image.open(mask_file).convert("RGB"))

    # Make a binary array identifying where the mask is black
    cond = npMask<128

    # Select image or mask according to condition array
    pixels=np.where(cond, npImage, npMask)

    # Save resulting image
    result=Image.fromarray(pixels)

    if sp_or_ht == "sp":
        segmented_image = "sp_seg_"+img_name+".png"
    elif sp_or_ht == "ht":
        segmented_image = "hand_truth_"+img_name+".png"

    result.save(segmented_image)
