import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
import scipy.ndimage
import xlrd
import matplotlib.patches as pth
import csv
import pandas
from join import roi_sp
import yaml

        
def proj():
    
##    file_name: Name of picture to read.
##    id_no: Index of point being used for Bounding Box
##    bbox: Size of bounding box
##    graphing: "on" if graph of point labels to be shown, "on" is deafault
##    superpixel: "on" if superpixel is needed, "on" is default

    with open('mission.yaml', 'r') as stream:
        load_data = yaml.load(stream)

    file_name = load_data['image']['file_name']
    id_no = load_data['image']['id_no']
    bbox = load_data['image']['bbox']
    graphing = load_data['image']['graphing']
    superpixel = load_data['image']['superpixel']
    label_file = load_data['image']['label_file']
    k_thres = load_data['image']['k_thres']
    slices = load_data['image']['slices']
    
    ext_index = file_name.index('.')
    file_ext = file_name[ext_index:]
    filename = file_name[:ext_index] #filename w/o file ext
    img1 = cv2.imread(file_name, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    height, width, channels = scipy.ndimage.imread(file_name).shape

    book = xlrd.open_workbook(label_file)                                           #open the excel sheet
    sheet = book.sheet_by_index(0)                                                          #opens sheet index X of said excel
    img_name = sheet.col_values(0)                                                          #the input names of picture is in column 2
    num_rows = img_name.count(filename)                                                    #counts number of labels of pic
    
    N = num_rows                                                                            #number of labels

    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)

    bb_edge = int(bbox/2)
    
    
    print("Height & Width of image is:\n", height,width)    
                                                                           
    A = 1
    
    while A != N+1:
        row_start = img_name.index(filename)                                               #searches row number of filename
        row_end = row_start + (N)                                                           #gives end of row number of filename

        x_cords = sheet.col_values(1,row_start,row_end)                                     #column 9 = x-coords on excel
        x = []
        for i in x_cords:                                                                   #appends x_coords from excel into empty list x
            x.append(i)                                                               #append above multiplied by dimension of image
            
        y_cords = sheet.col_values(2,row_start,row_end)
        y = []
        for i in y_cords:
            y.append(height+i)        

        #print("Point Number",A-1, ":")
        #print(x[A-1],",", y[A-1])

        p = pth.Rectangle((x[A-1]-bb_edge,y[A-1]-bb_edge), width=bbox, height=bbox, fill=False, color='red')
        ax.add_patch(p)
        
        colors = "red"
        area = np.pi * 1.5**2                                                               # 0 to 15 point radii
        A += 1
        
    types = sheet.col_values(5,row_start,row_end)
    t = list(set(types))
    tn = len(t)
    print(tn, "Types found in this image:", t)
    
    for i in range(0,len(t)):
        counting = []
        counts = types.count(t[i])
        counting.append(counts)
        counting.append(t[i])
        print(counting)
        
    #t_num_rows = type_name.count(types)

    
    print("Max number is:", num_rows-1)

    #for i in tn:
        
    x_i = []    #integers of the x,y values
    for i in x:
        a = int(i)
        x_i.append(a)

    y_i = []
    for j in y:
       b = int(j)
       y_i.append(b)

    #setting up the area for bounding box
    roi_x = x_i[id_no]
    roi_y = y_i[id_no]
    print("X co-ord for Point:",roi_x, "\nY co-ord for Point:", roi_y)

    roi = img[(height-roi_y)-bb_edge:(height-roi_y)+bb_edge , roi_x-bb_edge:roi_x+bb_edge] #y1:y2, x1:x2
    #print(roi)

    #BGR Channels of Bounding Box
    B_cha = []
    B = 0
    while B != bbox:
        for i in range(bbox):
            B_cha.append(roi[B][i][0])
        B += 1
    ave_b = sum(B_cha)/len(B_cha)
    #print("\nAverage for B channel:", ave_b)
    stdd_b = np.std(B_cha)
    #print("Standard Deviation for B channel:", stdd_b)
        
    G_cha = []
    G = 0
    while G != bbox:
        for i in range(bbox):
            G_cha.append(roi[G][i][1])
        G += 1
    ave_g = sum(G_cha)/len(G_cha)
    #print("\nAverage for G channel:", ave_g)
    stdd_g = np.std(G_cha)
    #print("Standard Deviation for G channel:", stdd_g)

    R_cha = []
    R = 0
    while R != bbox:
        for i in range(bbox):
            R_cha.append(roi[R][i][2])
        R += 1
    ave_r = sum(R_cha)/len(R_cha)
    #print("\nAverage for R channel:", ave_r)
    stdd_r = np.std(R_cha)
    #print("Standard Deviation for R channel:", stdd_r)

    #writting into CSV
##    with open('Species_readings.csv', mode='w') as csv_file:
##        spec_csv = csv.writer(csv_file, delimiter=',',
##                                 quotechar='"', quoting=csv.QUOTE_MINIMAL)
##        
##        spec_csv.writerow(['FileName','Bbox','B_ave','B_std',
##                                          'G_ave','G_std','R_ave','R_std'])
##        spec_csv.writerow([file_name,bbox,ave_b,stdd_b,
##                           ave_g,stdd_g,ave_r,stdd_r])
##    

    #df = pandas.read_csv('Species_readings.csv', index_col='FileName')
    #print(df)
    

    #cv2.imshow('image.cmap',cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    #cv2.imwrite("roi.jpg", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    if graphing == "on":

        
        scatter = ax.scatter(x, y, s=area, c=colors, alpha=0.5)
        #plt.scatter(x, y, edgecolors='red', marker = 's',facecolors='none', s = bbox**2)

        what = plt.imshow(img,zorder=0, extent=[0.0, width, 0.0, height])                       #saving plot image (offs axes)
        
        #plt.axis('off')
        #fig.axes.get_xaxis().set_visible(False)
        #fig.axes.get_yaxis().set_visible(False)
        #plt.savefig("Plot_"+file_name, bbox_inches='tight', pad_inches = 0)

        plt.axis("on")                                                                          #shows the plot (ons back axes)
        #fig.axes.get_xaxis().set_visible(True)
        #fig.axes.get_yaxis().set_visible(True)
        plt.show()

    else:
        pass
    
    if superpixel == "on":
        roi_sp(file_name,roi_x,roi_y,bb_edge, slices, k_thres, id_no)

    else:
        pass
    
    cv2.waitKey(0)

    
    
    cv2.destroyAllWindows()

