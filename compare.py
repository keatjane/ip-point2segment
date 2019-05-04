def compare(ori, ref, k):
    ori_int = []
    ref_int = []
    ori_range = []

## ORI & REF MUST BE IN FOLLOWING FORMAT (R,G,B)
    for i in ori:
        ori_int.append(int(i))
    for i in ref:
        ref_int.append(int(i))
    for i in ori_int:
        ori_range.append(int(i*k))

    
    ori_total = sum(ori_int)
    ref_total = sum(ref_int)
    ori_range_total = sum(ori_range)

    
    low_r = ori_range_total
    upp_r = (ori_total-ori_range_total)+ori_total
    

    if ref_total in range(low_r, upp_r):
        print(ref_total)
        return('yes')
    else:
        print("Range: ",low_r, "-", upp_r)
        print(ref_total)
        print('no')
        print("\n")

        return('no')


import numpy as np
import cv2
from scipy.misc import imread
import scipy.ndimage
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import csv

#x,y coords of slices start at top left corner //
#   origin with (0,0) is at top left

##    print("LEFT SLICE")
##    with open('next_slice.csv', mode='w') as csv_file:
##        spec_csv = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
##        spec_csv.writerow(['Slice Number', 'x', 'y', 'bool'])
##        y = 0 #2056
##        while y != height:
##            for x in range(width): #2464
##                if segments[y][x] == roi_seg_left:
##                    spec_csv.writerow([segments[y][x], x, y, 1])
##                else:
##                    pass
##            y += 1

def write_slice(csv_file, name_slice, img, n):
    print(name_slice, 'for', csv_file)
    height, width, channels = scipy.ndimage.imread(img).shape
    roi_sp = img_as_float(io.imread(img))
    n_segments = n
    segments = slic(roi_sp, n_segments,
                    compactness=10, sigma = 5)
    
    with open(csv_file, mode='w') as csv_file:
        spec_csv = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spec_csv.writerow(['Slice Number', 'x', 'y', 'bool'])
        y = 0 #2056
        while y != height:
            for x in range(width): #2464
                if segments[y][x] == name_slice:
                    spec_csv.writerow([segments[y][x], x, y, 1])
                else:
                    pass
            y += 1


    
    
    
