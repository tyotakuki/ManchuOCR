import os
import sys
import cv2
from cv2 import resize
import numpy as np
import matplotlib.pyplot as plt

import argparse
from PIL import Image

import torch

import src.utils as utils
import src.dataset as dataset

import crnn.seq2seq as crnn

def seq2seq_decode(encoder_out, decoder, decoder_input, decoder_hidden, max_length):
    decoded_words = []
    alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZŽŠŪ-\'"
    converter = utils.ConvertBetweenStringAndLabel(alph)
    prob = 1.0
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_out)
        probs = torch.exp(decoder_output)
        _, topi = decoder_output.data.topk(1)
        ni = topi.squeeze(1)
        decoder_input = ni
        prob *= probs[:, ni]
        if ni == utils.EOS_TOKEN:
            break
        else:
            decoded_words.append(converter.decode(ni))

    words = ''.join(decoded_words)
    prob = prob.item()

    return words, prob

def find_median(array_vals):
        array_vals.sort()
        mid = len(array_vals) // 2
        return array_vals[mid]

def detect_centerline(array_vals):
    max_val = max(array_vals)
    index_list = [index for index in range(len(array_vals)) if array_vals[index] == max_val]
    return find_median(index_list)


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def extract_peak_ranges_from_array(array_vals, minimum_val=100, minimum_range=2):
    start_i = None
    end_i = None
    peak_ranges = []
    for i, val in enumerate(array_vals):
        if val >= minimum_val and start_i is None:
            start_i = i
        elif val >= minimum_val and start_i is not None:
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

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='', help='the path of the input image')
parser.add_argument('--rot_angle', type=int, default=0, help='the global rotation image')
parser.add_argument('--padding', type=int, default=10, help='paddings at the head of the image')
parser.add_argument('--threshold', type=int, default=33, help='threshold for binarizing image, odd number only')
parser.add_argument('--thresholding_radius', type=int, default=32, help='radius to calculate the average for thresholding, even number only')
parser.add_argument('--vertical_minimum', type=int, default=800, help='minimal brightness of each VERTICAL line')
parser.add_argument('--word_minimum', type=int, default=200, help='minimal brightness of each WORD')
parser.add_argument('--blur', type=bool, default=False, help='apply blur to words?')
parser.add_argument('--pretrained', type=int, default=1, help='which pretrained model to use')
cfg = parser.parse_args()

def main():

    global_rot_angle = cfg.rot_angle
    global_padding = cfg.padding

    imagename = cfg.img_path

    if cfg.pretrained == 0:
        my_encoder = "/Users/zhuohuizhang/Downloads/ManchuOCR/crnn_seq2seq_ocr_pytorch-master/model/encoder_0.pth"
        my_decoder = "/Users/zhuohuizhang/Downloads/ManchuOCR/crnn_seq2seq_ocr_pytorch-master/model/decoder_0.pth"
    elif cfg.pretrained == 1:
        my_encoder = "/Users/zhuohuizhang/Downloads/ManchuOCR/crnn_seq2seq_ocr_pytorch-master/model/encoder_1.pth"
        my_decoder = "/Users/zhuohuizhang/Downloads/ManchuOCR/crnn_seq2seq_ocr_pytorch-master/model/decoder_1.pth"
    else:
        sys.exit("Unknown Pretrained Model!")

    print("Analyzing: "+imagename)

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZŽŠŪ-\'"
    print("Using Möllendorff Alphabet List: " + alphabet + "\n")

    # len(alphabet) + SOS_TOKEN + EOS_TOKEN
    num_classes = len(alphabet) + 2

    transformer = dataset.ResizeNormalize(img_width=480, img_height=64)

    image_color = cv2.imread(imagename)
    image_shape = (image_color.shape[0], image_color.shape[1])
    image_binary = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    image = cv2.rotate(image_binary, cv2.ROTATE_90_COUNTERCLOCKWISE)
    adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, cfg.threshold, cfg.thresholding_radius)
    adaptive_threshold = rotate_image(adaptive_threshold, global_rot_angle)
    adaptive_threshold = cv2.copyMakeBorder(adaptive_threshold, 20, 20, 20, 20, cv2.BORDER_CONSTANT, 0)
    adaptive_threshold = adaptive_threshold[10:adaptive_threshold.shape[0]-10, 10:adaptive_threshold.shape[1]-10]
    image_blur = cv2.GaussianBlur(adaptive_threshold,(3,3),cv2.BORDER_DEFAULT)

    cv2.imshow('Binary Image', cv2.rotate(adaptive_threshold, cv2.ROTATE_90_CLOCKWISE))
    cv2.waitKey(1)

    vertical_sum = np.sum(image_blur, axis=1)

    peak_ranges = extract_peak_ranges_from_array(vertical_sum,minimum_val=cfg.vertical_minimum,minimum_range=5)

    img_display = np.copy(adaptive_threshold)

    #peak_ranges.append((peak_ranges[-1][1],adaptive_threshold.shape[0]))
    peak_ranges.reverse()
    horizontal_peak_ranges2d = []

    for peak_range in peak_ranges:
        start_y = 0
        end_y = img_display.shape[1]
        image_x = image_blur[peak_range[0]:peak_range[1], start_y:end_y]
        horizontal_sum = np.sum(image_x,axis = 0)
        # plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))
        # plt.gca().invert_yaxis()
        # plt.show()
        horizontal_peak_ranges = extract_peak_ranges_from_array(horizontal_sum,minimum_val=cfg.word_minimum,minimum_range=5)
        horizontal_peak_ranges2d.append(horizontal_peak_ranges)
        for hor in horizontal_peak_ranges:
            cv2.rectangle(img_display, (hor[0], peak_range[0]), (hor[1], peak_range[1]), 140, 1)

            word_piece = adaptive_threshold[peak_range[0]:peak_range[1],hor[0]:hor[1]]
            if cfg.blur:
                word_piece = cv2.GaussianBlur(word_piece,(1,1),cv2.BORDER_DEFAULT)
            else:
                pass
            image_dimension = (word_piece.shape[0], word_piece.shape[1])
            #cv2.imshow('Words', word_piece)
            #print(word_piece.shape)
            if image_dimension[0] < 30 or image_dimension[1] < 20:
                pass
            else:
                factor = 1
                image_resized = cv2.resize(word_piece, (int(image_dimension[1]*factor),int(image_dimension[0]*factor)), interpolation = cv2.INTER_AREA)
                hor_sum = np.sum(image_resized, axis=1)
                ctr_line = detect_centerline(hor_sum)
                image_dimension_new = (image_resized.shape[0], image_resized.shape[1])
                add_padding = max([ctr_line, image_dimension_new[0]-ctr_line])

                # cv2.imshow('current Image', image_resized)
                # cv2.waitKey(0)

                if image_dimension_new[1]<=500:
                    padded = cv2.copyMakeBorder(image_resized, add_padding-ctr_line, add_padding-image_dimension_new[0]+ctr_line, 0, 0, cv2.BORDER_CONSTANT, 0)
                else:
                    padded = image_resized
                
                factor = 64/padded.shape[0]
                padded = cv2.resize(padded, (int(padded.shape[1]*factor),int(padded.shape[0]*factor)), interpolation = cv2.INTER_AREA)
                padded = cv2.copyMakeBorder(padded, 0, 0, global_padding, 480 - global_padding - padded.shape[0], cv2.BORDER_CONSTANT, 0)
                padded = Image.fromarray(np.uint8(padded)).convert('L')
                padded = transformer(padded)
                padded = padded.view(1, *padded.size())
                padded = torch.autograd.Variable(padded)

                encoder = crnn.Encoder(1, 1024)
                # no dropout during inference
                decoder = crnn.Decoder(1024, num_classes, dropout_p=0.0, max_length=121)

                map_location = 'cpu'

                encoder.load_state_dict(torch.load(my_encoder, map_location=map_location))
                decoder.load_state_dict(torch.load(my_decoder, map_location=map_location))

                encoder.eval()
                decoder.eval()

                encoder_out = encoder(padded)    

                max_length = 121
                decoder_input = torch.zeros(1).long()
                decoder_hidden = decoder.initHidden(1)

                words, prob = seq2seq_decode(encoder_out, decoder, decoder_input, decoder_hidden, max_length)
                print(words+" ", end = '')
        print("\n")
        cv2.destroyAllWindows()
        cv2.imshow('Current Line', cv2.rotate(img_display, cv2.ROTATE_90_CLOCKWISE))
        cv2.waitKey(1)

    input("Reading Completed, Press Any Key to Exit. Ambula Baniha.")
    # color = (0, 0, 255)
    # for i, peak_range in enumerate(peak_ranges):
    #     for horizontal_range in horizontal_peak_ranges2d[i]:
    #         x = peak_range[0]
    #         y = horizontal_range[0]
    #         w = peak_range[1]
    #         h = horizontal_range[1]
    #         patch = adaptive_threshold[x:w,y:h]
    #         cv2.rectangle(img_display, (y,x), (h,w), 255, 2)
    # #        print(cnt)
    # #        cv2.imwrite("/Users/zhuohuizhang/Downloads/ManchuOCR/Data/"+fontname+"/Result/"+'%d' %cnt + '.jpg', patch)
    #         cnt += 1
    # # cv2.imshow('Vertical Segmented Image', line_seg_blur)
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()
