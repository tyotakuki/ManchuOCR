import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

MAX_NUMBER = 3000
f = open("/Users/zhuohuizhang/Downloads/ManchuOCR/Data/AllWords.txt")
wordlist = f.readlines()
wordlist = list(map(lambda x:x[:-1], [word for word in wordlist if word != '\n']))
f.close()


#generate sentences
f = open("/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/All_Sentences_Test.txt",'a')
list_fonts = ['Abkai_Xanyan', 'Daicing_White', 'Sunggar_Paka', 'Daicing_Xiaokai']

for i in range(0,MAX_NUMBER):
#    wordcount = random.randint(1,3)
    wordcount = 1
    current_words_index = random.sample(range(0,15782),wordcount)
    current_font = random.sample(list_fonts,1)
    current_words = list(map(wordlist.__getitem__, current_words_index))
    random_spacing = random.sample(range(20, 60), wordcount - 1)
    f.write(' '.join(current_words)+'\n')

    current_image_dimension = []
    base_dir = "/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/"+current_font[0]+"/Centerline_Nopadding/"
    image_name = "%d"%(current_words_index[0]+1)+".jpg"
    path_base_image = os.path.join(base_dir, image_name)
    current_image = cv2.imread(path_base_image)
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    current_image_dimension.append((current_image.shape[0], current_image.shape[1]))

    if wordcount > 1:
        for word in range(1,wordcount):
            image_name = "%d"%(current_words_index[word]+1)+".jpg"
            path_processing_image = os.path.join(base_dir, image_name)
            processing_image = cv2.imread(path_processing_image)
            processing_image = cv2.cvtColor(processing_image, cv2.COLOR_BGR2GRAY)
            processing_image_padded = cv2.copyMakeBorder(processing_image, 0, 0, random_spacing[word-1], 0, cv2.BORDER_CONSTANT, 0)
            current_image = cv2.hconcat([current_image, processing_image_padded])
    else:
        pass
    current_image_dimension.append((current_image.shape[0], current_image.shape[1]))


    cv2.imwrite("/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/All_Sentences_Test/"+'%d' %(i+1) + '.jpg', current_image)
    print(i)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

for i in range(0,MAX_NUMBER):
    img = cv2.imread("/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/All_Sentences_Test/"+'%d' %(i+1) + '.jpg')
    current_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    random_rotate = 0
    rotated_image = rotate_image(current_image, random_rotate)
    rotated_size = (rotated_image.shape[0], rotated_image.shape[1])

    random_scale_x = random.uniform(1, 1.02)
    random_scale_y = random.uniform(1, 1.02)
    new_size = (int(rotated_size[1]*random_scale_x), int(rotated_size[0]*random_scale_y))
    resized_image = cv2.resize(rotated_image, new_size, interpolation=cv2.INTER_AREA)

    random_padding_top = random.randint(0,20)
    random_padding_bottom = random.randint(0,20)
    random_padding_left = random.randint(0,20)
    random_padding_right = random.randint(0,20)

    padded_image = cv2.copyMakeBorder(resized_image, random_padding_top, random_padding_bottom, random_padding_left, random_padding_right, cv2.BORDER_CONSTANT, 0)
    padded_size = (padded_image.shape[0], padded_image.shape[1])

    random_trim_top = int(random.uniform(0, 10))
    random_trim_bottom = int(random.uniform(0, 10))

    trimmed_image = padded_image[int(random_trim_top):padded_size[0]-int(random_trim_bottom),0:padded_size[1]]

    trimmed_size = (trimmed_image.shape[0], trimmed_image.shape[1])
    semi_final_image = cv2.resize(trimmed_image, (int(trimmed_size[1]*32/trimmed_size[0]), 32), interpolation=cv2.INTER_AREA)
    if semi_final_image.shape[1]<=280:
      final_image = cv2.copyMakeBorder(semi_final_image, 0, 0, 0, 280 - semi_final_image.shape[1], cv2.BORDER_CONSTANT, 0)
    else:
      final_image = padded_image[:,0:280]
    cv2.imwrite("/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/All_Sentences_Test_Irregular/"+'%d' %(i+1) + '.jpg', final_image)
    print(i)