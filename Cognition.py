import numpy as np
import cv2
import tensorflow as tf
import quantize_weights
from tensorflow.examples.tutorials.mnist import input_data
import math


def process(img_input):
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    # resize(원본이미지, 결과이미지크기 , 가로사이즈의 배수 fx, 세로사이즈의배수 fy , 보간법)  보간법:확대와 축소로 인한 이미지 소실 및 블록화 문제 해결
    #  결과이미지크기를 None이나 안 써넣었을 경우 fx,fy로 조절할 수 있다.

    # INTER_LINEAR : 화질 하 . 디폴트 값
    # INTER_AREA : 화질개선이 중간정도. 이미지 축소시에 권장
    # INTER_CUBIC : 화질이 좋지만 처리속도가 느리다. 이미지 확대시  + INTER_LINEAR를 사용함

    (thresh, img_binary) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # threshold( 원본이미지, 임계값, 임계값 이상일 때 바꿀 값, 바꾸는 타입)
    # 이미지를 바이너리화 Otsu는 이미지 히스토그램을 분석한 후 중간값을 취한다

    h, w = img_binary.shape
    # (height, width,channel 순)  img.shape[:1] == height,  img.shape[:2] == width

    ratio = 100 / h
    new_h = 100
    new_w = w * ratio  # 100w/h  == 가로/세로 x 100

    img_empty = np.zeros((110, 110), dtype=img_binary.dtype)  # 여유롭게 크게만든듯
    # 0으로 초기화된 array생성 원소의 dtype = img_binary.dtype
    #  dtype은 그냥 데이터타입 x = 32.341 이면 x.dtype = float32

    img_binary = cv2.resize(img_binary, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
    img_empty[:img_binary.shape[0], : img_binary.shape[1]] = img_binary  # 크게 만든 img_empty에 binary 크기만큼 copy
    # [y,x] 순이므로 [height,width]
    # [:a]시작인덱스를 생략하면 처음부터 인덱스 a-1 까지

    # https://dojang.io/mod/page/view.php?id=2208

    img_binary = img_empty

    cnts = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 영상내에서 같은 값을 가진 값들을 하나의 선으로 연결하는 것이 contour. 그것을 추출하는 findcontours

    # findContours(원본(바이너리여야함), contour추출모드, contour 근사 방법)
    # EXTERNAL : 가장 바깥쪽 contour 추출
    # CHAIN_APPROX_NONE : contour를 구성하는 모든 점 저장
    # CHAIN_APPROX_SIMPLE : contour의 수평,수직,대각선 방향의 점 다 버리고 끝점만 남겨둠 (직사각형의 4개 모서리점만 남기고 버림)

    # 컨투어의 무게중심 좌표를 구한다
    M = cv2.moments(cnts[0][0])

    center_x = (M['m10'] / M['m00'])  # m00 : 폐곡선의 면적
    center_y = (M["m01"] / M['m00'])

    # 무게 중심이 이미지 중심으로 오도록 이동시킨다.

    height, width = img_binary.shape[:2]
    shiftx = width / 2 - center_x
    shifty = height / 2 - center_y

    Translation_Matrix = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    img_binary = cv2.warpAffine(img_binary, Translation_Matrix, (width, height))
    # cv2.warpAffine( input_img, output, (w 열의 수,h 행의 수)) 이미지를 [1,0,x], [0,[1,y]에서 x,y 만큼 이동시키는 것

    img_binary = cv2.resize(img_binary, (28, 28), interpolation=cv2.INTER_AREA)
    flatten = img_binary.flatten() / 255.0
    # 28*28로 resize된 img_binary를 평평하게한다  ????근데 왜 255로 나누지

    return flatten


num_input = 784
num_classes = 10
method = 'quantize'
nb = 3
H = 1

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(quantize_weights.quantize_weights(tf.truncated_normal([5, 5, 1, 32]), method=method, H=H, nb=nb)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(quantize_weights.quantize_weights(tf.truncated_normal([5, 5, 32, 64]), method=method, H=H, nb=nb)),
    # 5x5 conv, 64 inputs, 1024 outputs
    'wd1': tf.Variable(quantize_weights.quantize_weights(tf.truncated_normal([7 * 7 * 64, 1024]), method=method, H=H, nb=nb)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(quantize_weights.quantize_weights(tf.truncated_normal([1024, num_classes]), method=method, H=H, nb=nb))
}


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv1(x, w):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    # conv = tf.nn.bias_add(conv, b)
    conv = maxpool2d(conv, k=2)
    return conv


def conv2(x, w):
    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    # conv = tf.nn.bias_add(conv, b)
    conv = maxpool2d(conv, k=2)
    return conv


def fully1(x, w):
    fc1 = tf.reshape(x, [-1, w.get_shape().as_list()[0]]) # w의 모양대로 reshape
    fc1 = tf.matmul(fc1, w)
    return fc1

conv1 = conv1(X, weights['wc1'])
conv1 = quantize_weights.quantize_weights(conv1, method='radix_ReLU', H=H, nb=nb)
conv2 = conv2(conv1, weights['wc2'])
conv2 = quantize_weights.quantize_weights(conv2, method='radix_ReLU', H=H, nb=nb)
fc1 = fully1(conv2, weights['wd1'])
max = tf.reduce_max(fc1)
fc1 = quantize_weights.quantize_weights(fc1, method='radix_ReLU_fixed', H=H, nb=nb, fixed_max=max)
# fc1 = quantize_weights.quantize_weights(fc1, method='radix_ReLU', H=H, nb=nb)
output = tf.matmul(fc1, weights['out'])

logits = output
prediction = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
# correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

save_file = 'C:/Users/ybrot/OneDrive/바탕 화면/QCNN./train_model'
saver = tf.train.Saver()

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with tf.Session() as sess:
    saver.restore(sess, save_file)
    sess.run(tf.assign(weights['wc1'], quantize_weights.quantize_weights(weights['wc1'], method=method, H=H, nb=nb)))
    sess.run(tf.assign(weights['wc2'], quantize_weights.quantize_weights(weights['wc2'], method=method, H=H, nb=nb)))
    sess.run(tf.assign(weights['wd1'], quantize_weights.quantize_weights(weights['wd1'], method=method, H=H, nb=nb)))
    sess.run(tf.assign(weights['out'], quantize_weights.quantize_weights(weights['out'], method=method, H=H, nb=nb)))
    test_accuracy = sess.run(
        accuracy,
        feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    # print('Test Accuracy: {}'.format(test_accuracy))

    while (True):

        ret, img_color = cap.read()

        if ret == False:
            break

        img_input = img_color.copy()
        cv2.rectangle(img_color, (250, 150), (width - 250, height - 150), (0, 0, 255), 3)
        cv2.imshow('bgr', img_color)

        img_roi = img_input[150:height - 150, 250:width - 250]

        key = cv2.waitKey(1)

        if key == 27: # esc 로 종료
            break
        elif key == 32:  # spacebar 로 CNN에 넘기기
            flatten = process(img_roi)

            # predictions = model.predict(flatten[np.newaxis, :])
            predictions = sess.run(prediction, feed_dict={X: flatten[np.newaxis, :], keep_prob: 1.0})

            # with tf.compat.v1.Session() as sess:
            with tf.compat.v1.Session():
                print(tf.argmax(predictions, 1).eval())

            cv2.imshow('img_roi', img_roi)
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
