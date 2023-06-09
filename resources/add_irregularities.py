import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

MAX_NUMBER = 75000

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

for i in range(0,MAX_NUMBER):
    img = cv2.imread("/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/Old_Data/All_Sentences/"+'%d' %(i+1) + '.jpg')
    current_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    random_rotate = random.uniform(-2, 2)
    rotated_image = rotate_image(current_image, random_rotate)
    rotated_size = (rotated_image.shape[0], rotated_image.shape[1])

    random_scale_x = random.uniform(1, 1.02)
    random_scale_y = random.uniform(1, 1.02)
    new_size = (int(rotated_size[1]*random_scale_x), int(rotated_size[0]*random_scale_y))
    resized_image = cv2.resize(rotated_image, new_size, interpolation=cv2.INTER_AREA)

    random_padding_top = random.randint(5,15)
    random_padding_bottom = random.randint(5,15)
    random_padding_left = random.randint(5,15)
    random_padding_right = random.randint(5,15)

    padded_image = cv2.copyMakeBorder(resized_image, random_padding_top, random_padding_bottom, random_padding_left, random_padding_right, cv2.BORDER_CONSTANT, 0)
    padded_size = (padded_image.shape[0], padded_image.shape[1])

    random_trim_top = int(random.uniform(2, 8))
    random_trim_bottom = int(random.uniform(2, 8))

    trimmed_image = padded_image[int(random_trim_top):padded_size[0]-int(random_trim_bottom),0:padded_size[1]]

    trimmed_size = (trimmed_image.shape[0], trimmed_image.shape[1])
    semi_final_image = cv2.resize(trimmed_image, (int(trimmed_size[1]*64/trimmed_size[0]), 64), interpolation=cv2.INTER_AREA)

    
    if semi_final_image.shape[1]<=280:
      final_image = cv2.copyMakeBorder(semi_final_image, 0, 0, 0, 280 - semi_final_image.shape[1], cv2.BORDER_CONSTANT, 0)
      gauss = np.random.normal(0,1,final_image.size)
      gauss = gauss.reshape(final_image.shape[0],final_image.shape[1]).astype('uint8')
      noise = final_image + 0.2* gauss
    else:
      final_image = semi_final_image[:,0:280]
      gauss = np.random.normal(0,1,final_image.size)
      gauss = gauss.reshape(final_image.shape[0],final_image.shape[1]).astype('uint8')
      noise = final_image + 0.2* gauss
    cv2.imwrite("/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/Old_Data/All_Sentences_Irregular/"+'%d' %(i+1) + '.jpg', noise)
    print(i)