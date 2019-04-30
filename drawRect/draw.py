import os
import cv2
import numpy as np


if __name__ == '__main__':
    pathCsv='train_label.csv'
    f=open(pathCsv,'r')
    img=np.zeros((1800,3200,3),np.uint8)
    print('image shape:',img.shape)

    for lines in f:
        if lines.split(',')[-1].strip()=='type' or not lines.strip():continue
        point1=(int( lines.split(',')[1]),int(lines.split(',')[2]))
        point2=(int(lines.split(',')[5]),int(lines.split(',')[6]))
        cv2.rectangle(img,point1,point2,(255,0,0))
        print(lines.split(',')[0])

    # cv2.imshow('rectImg',img)
    # cv2.waitKey()
    cv2.imwrite('rectImg.jpg',img)