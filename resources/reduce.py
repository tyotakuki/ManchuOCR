import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

reference_dir = "/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/AllSentences_Irregular_Original/"
fixed_width = 32
fixed_height = 100

filerange = range(0,50000)


for fileindex in filerange:
    imagename = "%d"%(fileindex+1)+".jpg"
    print(imagename)
    path_reference_image = os.path.join(reference_dir, imagename)

    current_image = cv2.imread(path_reference_image)
    image_dimension = (current_image.shape[0], current_image.shape[1])

    resized = cv2.resize(current_image, (fixed_height,fixed_width), interpolation = cv2.INTER_AREA)
    cv2.imwrite("/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/AllSentences_Irregular/"+'%d' %(fileindex+1) + '.jpg', resized)

    