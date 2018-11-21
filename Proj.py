import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
import scipy.ndimage
import xlrd

file_name = 'PR_20101021_080232_986_LC16'                                               #input name
img = imread(file_name+".jpg")
image = cv2.imread(file_name+".jpg")


height, width, channels = scipy.ndimage.imread(file_name+".jpg").shape                  #gets size of picture

book = xlrd.open_workbook("Test_Labels.xlsx")                                           #open the excel sheet
sheet = book.sheet_by_index(0)                                                          #opens sheet index X of said excel
img_name = sheet.col_values(2)                                                          #the input names of picture is in column 2
num_rows = img_name.count(file_name)                                                    #counts number of labels of pic


N = num_rows                                                                            #number of labels
A = 0

while A != N+1:
    row_start = img_name.index(file_name)                                               #searches row number of filename
    row_end = row_start + (N)                                                           #gives end of row number of filename

    x_cords = sheet.col_values(9,row_start,row_end)                                     #column 9 = x-coords on excel
    x = []
    for i in x_cords:                                                                   #appends x_coords from excel into empty list x
        x.append(i*width)                                                               #append above multiplied by dimension of image
        
    y_cords = sheet.col_values(10,row_start,row_end)
    y = []
    y2 = []
    for i in y_cords:
        y.append(i*height)
        y2.append(height-(i*height))

    print(image[y2,x,2]) #i=0 B, i=1 G, i=2 R
    print(image[y2,x,1])
    print(image[y2,x,0])
   
    colors = "red"
    area = np.pi * 1.5**2                                                               # 0 to 15 point radii
    A += 1
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    
plt.imshow(img,zorder=0, extent=[0.0, width, 0.0, height]) #L,R,T,B


fig = plt.imshow(img,zorder=0, extent=[0.0, width, 0.0, height])                        #saving plot image (offs axes)
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig("Plot_"+file_name, bbox_inches='tight', pad_inches = 0)

plt.axis("on")                                                                          #shows the plot (ons back axes)
fig.axes.get_xaxis().set_visible(True)
fig.axes.get_yaxis().set_visible(True)
plt.show()

### cv2.meanStdDev(src[, mean[, stddev[, mask]]])
### retval = cv.mean(src[, mask])

