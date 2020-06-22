import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame

from statistics import mean, median
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

'''
original = cv2.imread('ECG2.jpg',0)

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    print(a.size)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()



A = pywt.idwt(LL, None, 'db2', 'smooth')
print(A.size)
'''

image = cv2.imread('b1_00.png')
image = cv2.resize(image, (7522,750), interpolation=cv2.INTER_CUBIC)
'''
x_list, y_list = [], []
for x in np.arange(0, 7522, 1):
    for y in np.arange(0, 750, 1):
        if np.all(image[y][x] == (0, 0, 0)):
            x_list.append(x)
            y_list.append(750-y)
show_graph(x_list, y_list, 18, 3)

print(x_list)
print(y_list)

'''

train_X, train_y = [], []   # initialise for features extraction

position_list = locate_pos(image, (255, 0, 0)) # N
count = len(position_list)
print(count)
for i in range(count):
    train_X.append(extract_feat(image,
            position_list[i]-70, position_list[i]+90))
    train_y.append(1)

df = DataFrame(train_X)

print(df)

scaler = MinMaxScaler(feature_range=(0,1))

print(df.max)










