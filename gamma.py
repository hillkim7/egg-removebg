import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
import time

# https://ngost.tistory.com/56

restrPath = 'ispdataset/validation'
batch_size = 8
img_width, img_height = 180, 200
train_data_dir = 'predictdata/validation'


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def incContImg(image_RGB):
#     start = time.time()  # 시작 시간 저장
    # 명암 조절 https://stackoverrun.com/ko/q/10829873
#     cv2.imshow('image_RGB',image_RGB)

    #rgb to lab
#     print("===========labimg")
    image_LAB = cv2.cvtColor(image_RGB,cv2.COLOR_BGR2LAB);
#     cv2.imshow('labimg',image_LAB)

    # lightness channel 분리 후, CLAHE 채널 한후 다시 합치기
#     print("===========CLAHE")
    l, a, b = cv2.split(image_LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #Applying CLAHE to L-channel (Contrast Limited Adaptive Histogram Equalization)
    cl = clahe.apply(l)
#     cv2.imshow('CLAHE',cl)

#     print("===========REMERG_LAB")
    limg = cv2.merge((cl,a,b))
#     cv2.imshow('REMERG_LAB',limg)

#     print("===========LABTOBGR")
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#     cv2.imshow('LABTOBGR',final)
#     print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
# #     cv2.imwrite(restrPath+'.png', final)
    return final





j = 0
arr = ['0', '1', '4', '5']
for i in range(4):
       
    train_data_dir_sub = train_data_dir + '/' + arr[j]
    file_list = os.listdir(train_data_dir_sub)
    print("j::" + str(j))
    print(train_data_dir_sub)
    #     print(file_list)
    for  tmp in file_list:
#         tmp = random.choice(file_list)
#         tmp = file_list[g]
        if 'jpg' in tmp:
            print(tmp)
            tmpdir = train_data_dir_sub + '/' + tmp 
            ldimg = cv2.imread(tmpdir)
#             final = incContImg(ldimg)
            final = adjust_gamma(ldimg,gamma=0.8)
            
            cv2.imwrite(restrPath + '/' + arr[j] + '/' + 'trns_' + tmp, final)
    
    j += 1
    