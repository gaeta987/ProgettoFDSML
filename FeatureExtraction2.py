import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.engine.saving import model_from_json
import os
from statistics import mean, median
from PIL import Image
import scipy.signal

def find_stats(y_list, begin, end):
    temp = []
    for i in np.arange(begin, end, 1):
        temp.append(y_list[i])
    return [min(temp), mean(temp), median(temp), max(temp)]

'''def find_stats1(y_list, begin, end):
    temp = []
    for i in np.arange(begin, end, 1):
        temp.append(y_list[i])
    return [min(temp), mean(temp)]'''

def extract_feat(image, begin, end):
    x_list, y_list = [], [0]   # boundary padding add '0'
    for x in np.arange(begin, end, 1):
        x_list.append(x-begin)
        for y in np.arange(0, 300, 1):
            if np.all(image[y][x] == (255, 255, 255)):
                y_list.append(300-y)
                break
            if y==299:
                y_list.append(y_list[x-begin])
    y_list.pop(0)   # remove boundary padding '0'

    return y_list

def show_graph(x_list, y_list, width, height):
    plt.figure(figsize = [width, height])
    plt.scatter(x_list, y_list, marker='.', s=5)
    plt.show()
    return

image = cv2.imread('provetta.jpg')
image = cv2.resize(image, (300,300), interpolation=cv2.INTER_CUBIC)

x_list, y_list = [], []
for x in np.arange(0, 300, 1):
    for y in np.arange(0, 300, 1):
        #if np.all(image[y][x] == (0, 0, 0)):
        if np.all(image[y][x] == (255,255,255)):
            x_list.append(x)
            y_list.append(300-y)

print('Detect peaks without any filters.')
peaks, _ = scipy.signal.find_peaks(y_list, height = image.shape[0] - 100)

print(peaks)

for i in range(len(peaks)-1):
    print(y_list[peaks[i]])

extrac = extract_feat(image,
                peaks[0]-90, peaks[0]+96)

plt.plot(extrac)
plt.show()


show_graph(x_list, y_list, 18, 3)



















