import pandas
from scipy.misc import imread
import cv2
import scipy.ndimage
import numpy as np

def slice_rgb(file_name, csv_name):

    img1 = cv2.imread(file_name, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    height, width, channels = scipy.ndimage.imread(file_name).shape

    df = pandas.read_csv(csv_name)

    r = []
    g = []
    b1 = []


    
    #origin starts at top left for img[y,x]
    for i in range(len(df['x'])):
        col = img[ df['y'][i] , df['x'][i]]
        r.append(col[0])
        g.append(col[1])
        b1.append(col[2])

    r_ave = sum(r)/len(r)

    b_ave = sum(g)/len(g)
    g_ave = sum(b1)/len(b1)

    ri = int(r_ave)
    bi = int(b_ave)
    gi = int(g_ave)
    print((ri,gi,bi),"\n")
    return(ri,bi,gi)
