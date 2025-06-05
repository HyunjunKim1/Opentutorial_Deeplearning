import matplotlib.pyplot as plt                 # 결과 시각화를 위한 라이브러리 불러오기
import tensorflow as tf                         # tensorflow 라이브러리 불러오기
from sklearn.datasets import load_sample_image  # 샘플 데이터셋 불러오기
import numpy as np                              # numpy 라이브러리 불러오기

image = load_sample_image('china.jpg')          # 샘플 영상 불러오기
width, height, _ = np.shape(image)


def visualize(images, titles):        # 4개의 영상과 제목을 불러와 시각화하는 함수
  fig = plt.figure()
  for i in range(4):
    plt.subplot(1,4,i+1)
    plt.title(titles[i])
    plt.imshow(images[i])
    if np.shape(images[i])[-1] == 1:  # 3번째 차원(채널)이 1개인 회색조 영상의 경우
      plt.set_cmap('gray')            # 회색조로 표시
  plt.show()


# 영상의 밝기 및 대조비 조절
bright = tf.image.adjust_brightness(image, delta=0.2)           # 밝기 조절
contrast = tf.image.adjust_contrast(image, contrast_factor=0.8) # 대조비 조절
gamma = tf.imaage.adjust_gamma(image, gamma=1.6)                 # 감마 조절
visualize([image, bright, contrast, gamma], ['original', 'bright', 'contrast', 'gamma'])


# 색상 변화
grayscaled = tf.image.rgb_to_grayscale(image)                       # 회색조로 변환
saturated = tf.image.adjust_saturation(image, saturation_factor=2)  # 채도 변화
hue = tf.ima.adjust_hue(image, delta=0.04)                  # 색상 변화 주기
visualize([image, grayscaled, saturated, hue], ['original', 'grayscaled', 'saturated', 'hue'])


# 영상의 상하좌우 대칭 및 회전이동
flip_ud = tf.image.flip_up_down(image)      # 상하 대칭
flip_lr = tf.image.flip_left_right(image)   # 좌우 대칭
rot90 = tf.image.rot90(image)               # 시계방향 90도 회전
visualize([image, flip_ud, flip_lr, rot90], ['original', 'flip up/down', 'flip left/right', 'rotate 90 deg'])


# Resize 및 crop을 통한 Scale 다양화
center_crop = tf.image.central_crop(image, central_fraction=0.5)  # 중심을 기준으로 crop
random_crop = tf.image.random_crop(image, size=(width-50, height-50, 3))  # 정해진 size로 임의 위치에서 crop
# 영상을 3배 확대한 뒤 중앙 1/3 부분을 crop
resize_crop = tf.image.central_crop(tf.image.resize(image/255., size=(width*3, height*3)), central_fraction=0.333)
visualize([image, center_crop, random_crop, resize_crop], ['original', 'crop center', 'random crop', '2x resize+crop'])

# tf.image 내에 있는 영상 변환 함수 전체 list 확인:
print(dir(tf.image))

# API Documentation을 참조하면 자세한 사용법 설명을 볼 수 있다.
# https://www.tensorflow.org/api_docs/python/tf/image


# tackle h5py issue
# https://gitmemory.com/issue/h5py/h5py/1477/767677551
 