#-*- coding: utf-8 -*-

import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os
import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
from lxml import etree, objectify




def save_xml(image_name,oname, bbox, width, height, save_dir='/media/blacktea/DATA/traffic_light/traffic_light/VOC2007/Annotations', channel=3):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    for i in range(len( bbox)):
        left, top, right, bottom = bbox[i]
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = label[i]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom
    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
    save_xml = os.path.join(save_dir, image_name.replace('jpg', 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)

    return


'''
1、安装对应包
————————————————————————————
代码基于Windows，crop图片  大小为  800*450   （可更改）
运行基于Ubuntu的话需要更改路径等
标注默认csv文件格式，可转voc，需调用子函数

'''

if __name__ == '__main__':

    save_w_dir='G:/df/train/'
    textfile = open('./train_label_fix.csv', 'r', encoding='utf-8')
    ff=open('./train_crop_lable.csv','w',encoding='utf-8')
    num=0
    for line in textfile:
        if not line.strip():continue
        st=line.strip().split(',')
        name=st[0]
        if st[-1]=='type':continue
        del st[0]
        bbox = []
        label=[]
        boxlist = []
        x_sum,y_sum=0,0
        cc = ia.BoundingBox(int(st[0]),int(st[1]),int(st[4]),int(st[5]))
        x_sum=x_sum+int(st[0])+int(st[4])
        y_sum=y_sum+int(st[1])+int(st[5])
        boxlist.append(cc)
        label.append(st[8])

        if not os.path.exists('G:/df/Train_fix/'+name):
            continue

        img = cv2.imread('G:/df/Train_fix/' + name)
        height, width = img.shape[:2]
        crop_p=(1-x_sum/(2*width),1-y_sum/(2*height))
        image = ia.quokka(size=(height, width))

        if crop_p[0]<0 or crop_p[0]>1 or crop_p[1]<0 or crop_p[1]>1 :
            continue

        bbs = ia.BoundingBoxesOnImage(boxlist, image.shape)

        seq = iaa.Sequential([
            iaa.CropToFixedSize(width=800,height=450,position=crop_p)
         ]

        )

        seq_det = seq.to_deterministic()
        img_aug = seq_det.augment_images([img])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        print(str(num))
        num +=1
        bbs_aug = bbs_aug.remove_out_of_image().cut_out_of_image()
        box = bbs_aug.bounding_boxes
        if len(box)==0:continue
        for i in range(len(box)):
            bbox.append([box[i].x1_int,box[i].y1_int,box[i].x2_int,box[i].y2_int])

        #img_aug=cv2.rectangle(img_aug,(box[0].x1_int,box[0].y1_int),(box[0].x2_int,box[0].y2_int),(0,255,0))
        cv2.imwrite(save_w_dir+name,img_aug)
        writeline=name+','+str(box[0].x1_int)+','+str(box[0].y1_int)+','+str(box[0].x2_int)+','+str(box[0].y2_int)+','+str(label[0])+'\n'
        print(writeline)
        ff.write(writeline)
        #if num==10:break
        #save_xml(image_name=name,oname=label, bbox=bbox, width=width, height=height,save_dir='/media/blacktea/DATA/traffic_light/traffic_light/VOC2007/Annotations',  channel=3)