import os
import  cv2
import numpy
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os
import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
from lxml import etree, objectify


def save_xml(image_name,label, bbox, width, height, save_dir='./xml', channel=3):
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



if __name__ == '__main__':
    csv='./train_label_fix.csv'
    f=open(csv,'r')
    for line in f:
        line=line.strip().split(',')

        if line[-1]=='type':continue

        name=line[0]
        label=[line[-1]]
        box=[[line[1],line[2],line[5],line[6]]]
        save_xml(name,label,box,3200,1800)
