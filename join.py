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
from comparing_mask_original import img_and_mask
from seg_vs_seg import seg_vs_seg


## roi_sp('PR_20180806_160117_403_FC16.png', 1300, 550, 100, 200) 

def roi_sp(img, x, y, bb_edge, n, k_thres, id_no):
    bbox = bb_edge*2

    x_co = [x,x-bb_edge,x+bb_edge,x,x]
    y_co = [y,y,y,y-bb_edge,y+bb_edge]

    x_2 = [x-bb_edge,x+bb_edge,x+bb_edge,x-bb_edge] 
    y_2 = [y+bb_edge,y+bb_edge,y-bb_edge,y-bb_edge]
    
    #what = cv2.imread(img, cv2.IMREAD_COLOR)
    #fixed = cv2.cvtColor(what, cv2.COLOR_BGR2RGB)
    roi_sp = img_as_float(io.imread(img))
    ##    for numSegments in (100,200): #(100,200,300)
    #n_segments = 200 #(100,200,300)
    #np.set_printoptions(threshold=np.inf)
    n_segments = n
    segments = slic(roi_sp, n_segments,
                    compactness=10, sigma = 5)

    roi_fig = plt.figure("Superpixel"+img)
    roi_ax = roi_fig.add_subplot(1, 1, 1)
    height, width, channels = scipy.ndimage.imread(img).shape
    area = np.pi * 1.3**2
    colors = "red"
    scatter = roi_ax.scatter(x_co, y_co, s=area, c=colors, alpha=0.5)
    scatter = roi_ax.scatter(x_2, y_2, s=area, c='blue', alpha=0.5)
    w = (mark_boundaries(roi_sp, segments))
    #roi_ax.imshow(w)
    what = plt.imshow(w,zorder=0, extent=[0.0, width, 0.0, height])
    plt.axis("off")
    
    #plt.show()
    
    # x,y starts at bottom left so use height-y to start at top left
    roi_seg_no = (segments[height-y][x])
    roi_seg_4 = (segments[height-y][x-bb_edge])
    roi_seg_5 = (segments[height-y][x+bb_edge])
    roi_seg_2 = (segments[height-y-bb_edge][x])
    roi_seg_7 = (segments[height-y+bb_edge][x])

    roi_seg_1 = (segments[height-(y+bb_edge)][x-bb_edge])
    roi_seg_3 = (segments[height-(y+bb_edge)][x+bb_edge])
    roi_seg_6 = (segments[height-(y-bb_edge)][x-bb_edge])
    roi_seg_8 = (segments[height-(y-bb_edge)][x+bb_edge])
    
    
    print("""
    Slicing is done as below: \n
    1     2     3
    
    4     C     5
    
    6     7     8
    """)
    
    print("Center",roi_seg_no)
    print("-------------")
    print("1: ",roi_seg_1)
    print("2: ",roi_seg_2)
    print("3: ",roi_seg_3)
    print("4: ",roi_seg_4)
    print("C: ",roi_seg_no)
    print("5: ",roi_seg_5)
    print("6: ",roi_seg_6)
    print("7: ",roi_seg_7)
    print("8: ",roi_seg_8)
    print("\n")
    
    roi_new = plt.figure("ROI")
    roi_xa = roi_new.add_subplot(1,1,1)
    scatter = roi_xa.scatter(x_co, y_co, s=area, c=colors, alpha=0.5)
    scatter = roi_xa.scatter(x_2, y_2, s=area, c='blue', alpha=0.5)
    roi = w[(height-y)-bb_edge:(height-y)+bb_edge , x-bb_edge:x+bb_edge]
    
    #roi_xa.imshow(roi)
    #cv2.imshow('roi', roi)
    fig_file_name = img+"_id_no_"+str(id_no)+"_bbox_"+str(bbox)+"_segments_"+str(n_segments)
    what3 = plt.imshow(roi,zorder=0, extent=[0.0, bbox, 0.0, bbox])
    roi_fig.savefig("Superpixel_"+fig_file_name+".png", bbox_inches='tight', pad_inches = 0)
    roi_new.savefig("ROI_Superpixel_"+fig_file_name+".png", bbox_inches='tight', pad_inches = 0)
    #plt.show()
    
    ##cv2.waitKey(0)
    #cv2.destroyAllWindows()

    
    yes_slic = []

    write_slice('selected_slice.csv', roi_seg_no, img, n)
    r_sel,g_sel,b_sel = slice_rgb(img, 'selected_slice.csv')
    ori = (r_sel,g_sel,b_sel)
    yes_slic.append(roi_seg_no)

#SLICE NUMBER 1
    print("-----Slice 1-----")
    if roi_seg_1 != roi_seg_no:
        write_slice('next_slice.csv', roi_seg_1, img, n)    
        r_1,g_1,b_1 = slice_rgb(img, 'next_slice.csv')
        res_1 = compare(ori,(r_1,g_1,b_1),k_thres)
        if res_1  == 'yes':
            yes_slic.append(roi_seg_1)
        else:
            pass       
    elif roi_seg_1 == roi_seg_no:
        print("1 = Center")

#SLICE NUMBER 2
    print("-----Slice 2-----")
    if roi_seg_2 != roi_seg_no:
        write_slice('next_slice.csv', roi_seg_2, img, n)
        r_2,g_2,b_2 = slice_rgb(img, 'next_slice.csv')
        res_2  = compare(ori,(r_2,g_2,b_2),k_thres)
        if res_2  == 'yes':
            yes_slic.append(roi_seg_2)
        else:
            pass
    elif roi_seg_2 == roi_seg_no:
        print("2 = Center")

#SLICE NUMBER 3
    print("-----Slice 3-----")
    if roi_seg_3 != roi_seg_no:
        write_slice('next_slice.csv', roi_seg_3, img, n)    
        r_3,g_3,b_3 = slice_rgb(img, 'next_slice.csv')
        res_3 = compare(ori,(r_3,g_3,b_3),k_thres)
        if res_3 == 'yes':
            yes_slic.append(roi_seg_3)
        else:
            pass
    elif roi_seg_3 == roi_seg_no:
        print("3 = Center")
          
#SLICE NUMBER 4
    print("-----Slice 4-----")
    if roi_seg_4 != roi_seg_no:
        write_slice('next_slice.csv', roi_seg_4, img, n)    
        r_4,g_4,b_4 = slice_rgb(img, 'next_slice.csv')
        res_4 = compare(ori,(r_4,g_4,b_4),k_thres)
        if res_4 == 'yes':
            yes_slic.append(roi_seg_4)
        else:
            pass
        
    elif roi_seg_4 == roi_seg_no:
        print("4 = Center")
    
#SLICE NUMBER 5
    print("-----Slice 5-----")
    if roi_seg_5 != roi_seg_no:
        write_slice('next_slice.csv', roi_seg_5, img, n)
        r_5,g_5,b_5 = slice_rgb(img, 'next_slice.csv')
        res_5 = compare(ori,(r_5,g_5,b_5),k_thres)
        if res_5 == 'yes':
            yes_slic.append(roi_seg_5)
        else:
            pass
    elif roi_seg_5 == roi_seg_no:
        print("5 = Center")
        
#SLICE NUMBER 6
    print("-----Slice 6-----")
    if roi_seg_6 != roi_seg_no:
        write_slice('next_slice.csv', roi_seg_6, img, n)
        r_6,g_6,b_6 = slice_rgb(img, 'next_slice.csv')
        res_6 = compare(ori,(r_6,g_6,b_6),k_thres)
        if res_6 == 'yes':
            yes_slic.append(roi_seg_6)
        else:
            pass
    elif roi_seg_6 == roi_seg_no:
         print("6 = Center")    
    
#SLICE NUMBER 7
    print("-----Slice 7-----")
    if roi_seg_7 != roi_seg_no:
        write_slice('next_slice.csv', roi_seg_7, img, n)
        r_7,g_7,b_7 = slice_rgb(img, 'next_slice.csv')
        res_7 = compare(ori,(r_7,g_7,b_7),k_thres)
        if res_7 == 'yes':
            yes_slic.append(roi_seg_7)
        else:
            pass
    elif roi_seg_7 == roi_seg_no:
         print("7 = Center")
        
#SLICE NUMBER 8
    print("-----Slice 8-----")
    if roi_seg_8 != roi_seg_no:
        write_slice('next_slice.csv', roi_seg_8, img, n)
        r_8,g_8,b_8 = slice_rgb(img, 'next_slice.csv')
        res_8 = compare(ori,(r_8,g_8,b_8),k_thres)
        if res_8 == 'yes':
            yes_slic.append(roi_seg_8)
        else:
            pass
    elif roi_seg_8 == roi_seg_no:
         print("8 = Center")

    cmon = set(yes_slic)
    print(cmon)

    plt.show()

    #img1 = cv2.imread(img, cv2.IMREAD_COLOR)
    img_fix = scipy.ndimage.imread(img)

    y_seg = 0 #2056
    while y_seg != height:
        for x_seg in range(width): #2464
            if segments[y_seg][x_seg] in cmon:
                img_fix[y_seg][x_seg] = [0,0,0]
            else:
                img_fix[y_seg][x_seg] = [255,255,255]

        y_seg += 1

    img_name = img[:-4]

    
    cv2.imwrite('seg.png',img_fix)

    
    img_and_mask(img, 'seg.png', 'sp')

    
    seg_imgfile = "sp_seg_"+img
    ht_imgfile = "hand_truth_"+img
    print("----------------")
    print("Name of File:", img)
    print("Bounding Box size:", bbox)
    print("k constant Threshold:", k_thres)
    seg_vs_seg(seg_imgfile, ht_imgfile, height, width, x, y, bbox, k_thres, id_no, n_segments)


    
    


    print("DONE")

