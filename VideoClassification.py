import os
import sys
import argparse
import time
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from statistics import mean, median

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

currentDT = time.localtime()
start_datetime = time.strftime("-%m-%d-%H-%M-%S", currentDT)

def crop(image, w, f):
    return image[:, int(w * f): int(w * (1 - f))]

def find_stats(y_list, begin, end):
    temp = []
    for i in np.arange(begin, end, 1):
        temp.append(y_list[i])
    return [min(temp), mean(temp), median(temp), max(temp)]

def extract_feat(image, begin, end):
    x_list, y_list = [], [0]   # boundary padding add '0'
    for x in np.arange(begin, end, 1):
        x_list.append(x-begin)
        for y in np.arange(0, 750, 1):
            if np.all(image[y][x] == (0, 0, 0)):
                y_list.append(750-y)
                break
            if y==749:
                y_list.append(y_list[x-begin])
    y_list.pop(0)   # remove boundary padding '0'
    show_graph(x_list, y_list, 2, 2)
    y_list.extend(find_stats(y_list, 0, 40))  # quardrant 1
    y_list.extend(find_stats(y_list, 40, 80))  # quardrant 2
    y_list.extend(find_stats(y_list, 80, 120))  # quardrant 3
    y_list.extend(find_stats(y_list, 120, 160))  # quardrant 4
    y_list.extend(find_stats(y_list, 0, 80))  # segment 1
    y_list.extend(find_stats(y_list, 40, 120))  # segment 2
    y_list.extend(find_stats(y_list, 80, 160))  # segment 3
    print(len(y_list))
    return y_list

def locate_pos(image, color):
    position_list = []
    y_level = 42
    while len(position_list) == 0 and y_level < 100:
        x=100
        while x<700:
            if np.all(image[y_level][x] == color):
                position_list.append(x)
                x += 25
            x += 1
        y_level+=2
    return position_list

def show_graph(x_list, y_list, width, height):
    plt.figure(figsize = [width, height])
    plt.scatter(x_list, y_list, marker='.', s=5)
    plt.show()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='ID of the device to open')
    parser.add_argument('--model', type=str, default='model/model.h5', help='path to the weights file')
    parser.add_argument('--frame_ratio', type=int, default=7, help='analyze every [n] frames')
    # --process_speed changes at how many times the model analyzes each frame at a different scale
    parser.add_argument('--process_speed', type=int, default=1,
                        help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('--out_name', type=str, default=None, help='name of the output file to write')
    parser.add_argument('--mirror', type=bool, default=True, help='whether to mirror the camera')

    args = parser.parse_args()
    device = args.device
    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    mirror = args.mirror


    # Video reader
    cam = cv2.VideoCapture(device)
    # CV_CAP_PROP_FPS
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    print("Running at {} fps.".format(input_fps))

    ret_val, orig_image = cam.read()

    width = orig_image.shape[1]
    height = orig_image.shape[0]
    factor = 0.3

    i = 0  # default is 0
    resize_fac = 1
    # while(cam.isOpened()) and ret_val is True:

    while True:

        cv2.waitKey(10)

        if cam.isOpened() is False or ret_val is False:
            break

        if mirror:
            orig_image = cv2.flip(orig_image, 1)

        tic = time.time()

        cropped = crop(orig_image, width, factor)

        input_image = cv2.resize(cropped, (0, 0), fx=1 / resize_fac, fy=1 / resize_fac, interpolation=cv2.INTER_CUBIC)

        print('Processing frame: ', i)
        toc = time.time()
        print('processing time is %.5f' % (toc - tic))

        toc = time.time()
        print('processing time is %.5f' % (toc - tic))

        canvas = cv2.resize(input_image, (0, 0), fx=4, fy=2, interpolation=cv2.INTER_CUBIC)

        cv2.imshow('frame', canvas)

        train_X, train_y = [], []  # initialise for features extraction

        image = cv2.resize(input_image, (7522, 750), interpolation=cv2.INTER_CUBIC)

        position_list = locate_pos(image, (255, 0, 0))  # N
        count = len(position_list)
        print(count)
        for i in range(count):
            train_X.append(extract_feat(image,
                                        position_list[i] - 70, position_list[i] + 90))
            train_y.append(1)

        for i in range(0, len(train_X)):
            train_new = np.reshape(train_X[i], (188, 1))

            scaler = MinMaxScaler(feature_range=(0, 1))

            train_new1 = scaler.fit_transform(train_new)

            print(np.reshape(train_new1, (1, 188)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret_val, orig_image = cam.read()

        i += 1



