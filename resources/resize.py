import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

reference_dir = "/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/Abkai_Xanyan/"
fixed_width = 100
fixed_height = 500

filerange = range(1,15783)

height_list = []

for fileindex in filerange:
    imagename = "%d"%fileindex+".jpg"
#    print(imagename)
    path_reference_image = os.path.join(reference_dir, imagename)

    current_image = cv2.imread(path_reference_image)
    image_dimension = (current_image.shape[0], current_image.shape[1])

#    resized = cv2.resize(current_image, (image_dimension[1],100), interpolation = cv2.INTER_AREA)
#    cv2.imwrite("/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/Abkai_Xanyan/Resized/"+'%d' %fileindex + '.jpg', resized)
    height_list.append(image_dimension[1])

base_dir = "/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/Daicing_Xiaokai/"
print(max(height_list))

for fileindex in filerange:
    imagename = "%d"%fileindex+".jpg"
    print(imagename)
    path_base_image = os.path.join(base_dir, imagename)

    current_image = cv2.imread(path_base_image)
    image_dimension = (current_image.shape[0], current_image.shape[1])

    resized = cv2.resize(current_image, (height_list[fileindex-1],100), interpolation = cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(resized, 0, 0, 0, 500 - height_list[fileindex-1], cv2.BORDER_CONSTANT, 0)
    cv2.imwrite("/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/Daicing_Xiaokai/Padded/"+'%d' %fileindex + '.jpg', padded) 