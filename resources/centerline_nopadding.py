import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


#for fontname in ["Daicing_Buleku"]:
for fontname in ['Shukai', 'Yingbi', 'Xingshu', 'Wenqin', 'Liuye', 'Daicing']:
#for fontname in ["Abkai_Xanyan"]:
    reference_dir = "/Users/zhuohuizhang/Downloads/ManchuHandwritten/"+fontname+"/Words/"
    fixed_width = 200
    fixed_height = 500
    filerange = range(1,130918)

    ctr = 1

    def find_median(array_vals):
        array_vals.sort()
        mid = len(array_vals) // 2
        return array_vals[mid]


    def detect_centerline(array_vals):
        max_val = max(array_vals)
        index_list = [index for index in range(len(array_vals)) if array_vals[index] == max_val]
        return find_median(index_list)


    height_list = []

    for fileindex in filerange:
        imagename = "%d"%fileindex+".jpg"
    #    print(imagename)
        path_reference_image = os.path.join(reference_dir, imagename)

        current_image = cv2.imread(path_reference_image)
        
        image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        image_dimension = (image.shape[0], image.shape[1])
        # factor = 64/image.shape[0]
        # image_resized = cv2.resize(image, (int(image_dimension[1]*factor),int(image_dimension[0]*factor)), interpolation = cv2.INTER_AREA)
        hor_sum = np.sum(image, axis=1)
        print(fileindex)
    #    print(image_dimension)
        ctr_line = detect_centerline(hor_sum)
        image_dimension_new = (image.shape[0], image.shape[1])
        add_padding = max([ctr_line, image_dimension_new[0]-ctr_line])
        print(ctr_line)
        
        
        if image_dimension_new[1]<=500:
            padded = cv2.copyMakeBorder(image, add_padding-ctr_line, add_padding-image_dimension_new[0]+ctr_line, 0, 0, cv2.BORDER_CONSTANT, 0)
        else:
            pass
    #    resized = cv2.resize(current_image, (image_dimension[1],100), interpolation = cv2.INTER_AREA)
        newfactor = 64/padded.shape[0]
        final_image = cv2.resize(padded, (int(padded.shape[1]*newfactor),int(padded.shape[0]*newfactor)), interpolation = cv2.INTER_AREA)
        print(final_image.shape)
        cv2.imwrite("/Users/zhuohuizhang/Downloads/ManchuHandwritten/"+fontname+"/Centerline/"+'%d' %ctr + '.jpg', final_image)
        ctr += 1