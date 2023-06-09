import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


MAX_NUMBER = 10000
f = open("/Users/zhuohuizhang/Downloads/ManchuOCR/Data/AllWords.txt")
wordlist = f.readlines()
wordlist = list(map(lambda x:x[:-1], [word for word in wordlist if word != '\n']))
f.close()


#generate sentences
f = open("/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/Old_Data/All_Sentences_Real.txt",'a')
list_fonts = ['Abkai_Xanyan', 'Sunggar_Ginggulere', 'Daicing_White', 'Sunggar_Paka', 'Daicing_Xiaokai', 'Daicing_Buleku']

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


    cv2.imwrite("/Users/zhuohuizhang/Downloads/ManchuOCR/DataSet/Old_Data/All_Sentences_Real/"+'%d' %(i+1) + '.jpg', current_image)
    print(i)