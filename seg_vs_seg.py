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
from extras import slice_rgb
from compare import compare
from compare import write_slice


def seg_vs_seg(seg_sp, seg_truth, height, width, x, y, bbox, k_thres, id_no, n_segments):
    
    img_sp1 = cv2.imread(seg_sp)
    img_truth1 = cv2.imread(seg_truth)

    img_name = seg_sp[7:-4]

    b,g,r = cv2.split(img_sp1)
    img_sp = cv2.merge([r,g,b])

    b1,g1,r1 = cv2.split(img_truth1)
    img_truth = cv2.merge([r1,g1,b1])

    bbox_y1 = (height - y) - (bbox/2)
    bbox_x1 = x - (bbox/2)

    bbox_y2 = bbox_y1 + bbox
    bbox_x2 = bbox_x1 + bbox

    y1 = int(bbox_y1)
    x1 = int(bbox_x1)

    y2 = int(bbox_y2)
    x2 = int(bbox_x2)

    save_fig = img_name+"_id_no_"+str(id_no)+"_bbox_"+str(bbox)+"_kthres_"+str(k_thres)+"_segments_"+str(n_segments)



    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    pixel_count = 0

    w = img_sp[y1:y2, x1:x2]
    z = img_truth[y1:y2, x1:x2]

    cv2.imwrite(save_fig+".png", w)

    #cv2.imshow('Superpixel',w)
    #cv2.imshow('Hand Truth',z)

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(w)
    a.set_title('Superpixel')
    
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(z)
    a.set_title('Hand Truth')

    plt.show()
                    
    fig.savefig(save_fig+".png")

    with open("color_map.csv", mode='a') as csv_file:
        spec_csv = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spec_csv.writerow(['TP_x', 'TP_y', 'TN_x', 'TN_y', 'FP_x', 'FP_y', 'FN_x', 'FN_y'])
        
        y = y1 #2056
        while y != y2:
            for x in range(x1, x2): #2464
                if np.all(img_sp[y][x] == img_truth[y][x]):
                    pixel_count += 1
                    TP += 1
                    spec_csv.writerow([x, y, 0, 0, 0, 0, 0, 0])
                else:
                    pixel_count += 1
                    TN += 1
                    spec_csv.writerow([0, 0, x, y, 0, 0, 0, 0])
            y += 1
            
        y_new = y1 #2056
        while y_new != y2:
            for x_new in range(x1, x2): #2464
                if np.all(img_sp[y_new][x_new] != [255, 255, 255]) and np.all(img_sp[y_new][x_new] != img_truth[y_new][x_new]):
                    FP += 1
                    spec_csv.writerow([0, 0, 0, 0, x, y, 0, 0])
                    
                elif np.all(img_sp[y_new][x_new] == [255, 255, 255]) and np.all(img_sp[y_new][x_new] != img_truth[y_new][x_new]):
                    FN += 1
                    spec_csv.writerow([0, 0, 0, 0, 0, 0, x, y])
               
                else:
                    pass
                
            y_new += 1



    #print("True: ",truth)
    print("Total: ",pixel_count)
    #print("False: ",wrong)

    t = (TP/pixel_count)*100
    f = (TN/pixel_count)*100

    accuracy = ((TP+TN)/(TP+FP+TN+FN))*100
    precision = ((TP)/(TP+FP))*100
    recall = ((TP)/(TP+FN))*100

    print("T%: ", t)
    print("F%: ", f)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("\n")
    with open("Final_Results.csv", mode='a') as csv_file:
        spec_csv = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spec_csv.writerow([img_name, id_no, bbox, k_thres, n_segments, accuracy, precision, recall, TP, TN, FP, FN, t, f])
        
    df = pandas.read_csv("Final_Results.csv")
    print("-----Prev Results-----")
    print(df.tail(4))

    
    cv2.waitKey(0)



