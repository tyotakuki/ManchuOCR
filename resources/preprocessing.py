import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#list_fonts = ['Shukai', 'Yingbi', 'Xingshu', 'Wenqin', 'Liuye', 'Daicing']
list_fonts = ['Daicing']

for fontname in list_fonts:
    base_dir = "/Users/zhuohuizhang/Downloads/ManchuHandwritten/"+fontname+"/PNG/"

    cnt = 1
    if fontname == 'Daicing':
        filerange = range(1,11385)
    else:
        filerange = range(1,2270)

    for fileindex in filerange:
        if fileindex < 10:
            imagename = "0000%d"%fileindex+".png"
        elif fileindex < 100:
            imagename = "000%d"%fileindex+".png"
        elif fileindex < 1000:
            imagename = "00%d"%fileindex+".png"
        elif fileindex < 10000:
            imagename = "0%d"%fileindex+".png"
        else:
            imagename = "%d"%fileindex+".png"

    # for fileindex in filerange:
    #     if fileindex < 10:
    #         imagename = "000%d"%fileindex+".png"
    #     elif fileindex < 100:
    #         imagename = "00%d"%fileindex+".png"
    #     elif fileindex < 1000:
    #         imagename = "0%d"%fileindex+".png"
    #     elif fileindex < 10000:
    #         imagename = "%d"%fileindex+".png"
        
        print(imagename)
        path_test_image = os.path.join(base_dir, imagename)

        image_color = cv2.imread(path_test_image)

        new_shape = (image_color.shape[0], image_color.shape[1])


        image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)
        dilation = cv2.dilate(adaptive_threshold, np.ones((5,5),np.uint8), iterations = 1)
        image_blur = cv2.GaussianBlur(dilation,(9,9),cv2.BORDER_DEFAULT)

        # cv2.imshow('binary image', image_blur)
        # cv2.waitKey(0)

        vertical_sum = np.sum(image_blur, axis=1)
        # if fileindex == 8:
        #     plt.plot(vertical_sum, range(vertical_sum.shape[0]))
        #     plt.gca().invert_yaxis()
        #     plt.show()

        def extract_peak_ranges_from_array(array_vals, minimum_val = 100, minimum_range=30):
            start_i = None
            end_i = None
            peak_ranges = []
            for i, val in enumerate(array_vals):
                if val > minimum_val and start_i is None:
                    start_i = i
                elif val > minimum_val and start_i is not None:
                    pass
                elif val < minimum_val and start_i is not None:
                    end_i = i
                    if end_i - start_i > minimum_range:
                        peak_ranges.append((start_i, end_i))
                        start_i = None
                        end_i = None
                elif val < minimum_val and start_i is None:
                    pass
                else:
                    raise ValueError("Cannot Parse")
            return peak_ranges

        peak_ranges = extract_peak_ranges_from_array(vertical_sum,  minimum_val=100, minimum_range=30)
        # print(peak_ranges)

        line_seg_blur = np.copy(image_blur)
        # for i, peak_range in enumerate(peak_ranges):
        #     x = peak_range[0]
        #     y = 0
        #     w = peak_range[1]
        #     h = 500
        #     pt1 = (y, x)
        #     pt2 = (y + h, w)
        #     cv2.rectangle(line_seg_blur, pt1, pt2, 255, 2)
        # cv2.imshow('Vertical Segmented Image', line_seg_blur)
        # cv2.waitKey(0)

        horizontal_peak_ranges2d = []
        for peak_range in peak_ranges:
            start_y = 0
            end_y = line_seg_blur.shape[1]
            line_image = image_blur[peak_range[0]:peak_range[1], start_y:end_y]
            horizontal_sum = np.sum(line_image,axis = 0)
            horizontal_peak_ranges = extract_peak_ranges_from_array(horizontal_sum,minimum_val=50,minimum_range=30)
            horizontal_peak_ranges2d.append(horizontal_peak_ranges)
        #     cv2.rectangle(line_seg_blur, (horizontal_peak_ranges[0][0], peak_range[0]), (horizontal_peak_ranges[0][1], peak_range[1]), 255, 5)
        # cv2.imshow('Segmented Image', line_seg_blur)
        # cv2.waitKey(0)

        # print(horizontal_peak_ranges2d)


        color = (0, 0, 255)
        for i, peak_range in enumerate(peak_ranges):
            for horizontal_range in horizontal_peak_ranges2d[i]:
                x = peak_range[0]
                y = horizontal_range[0]
                w = peak_range[1]
                h = horizontal_range[1]
                patch = adaptive_threshold[x:w,y:h]
                cv2.rectangle(line_seg_blur, (y,x), (h,w), 255, 2)
        #        print(cnt)
                cv2.imwrite("/Users/zhuohuizhang/Downloads/ManchuHandwritten/"+fontname+"/Words/"+'%d' %cnt + '.jpg', patch)
                cnt += 1
                print(cnt)
        # cv2.imshow('Vertical Segmented Image', line_seg_blur)
        # cv2.waitKey(0)