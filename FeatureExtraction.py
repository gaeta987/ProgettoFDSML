import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.engine.saving import model_from_json
import os
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
    #show_graph(x_list, y_list, 5, 5)

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

json_file = open('best_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("best-model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

list = os.listdir('images')

train_X, train_y = [], []   # initialise for features extraction

for i in range(len(list)):
    path = 'images/' + list[i]
    image = cv2.imread(path)

    print(list[i])

    position_list = locate_pos(image, (255, 0, 0))
    print(position_list)
    count = len(position_list)
    print(count)

    for j in range(count):
        extrac = extract_feat(image,
                position_list[j]-90, position_list[j]+96)
        train_X.append(extrac)

        train_new = np.reshape(extrac, (186, 1))

        scaler = MinMaxScaler(feature_range=(0, 1))

        train_new1 = scaler.fit_transform(train_new)

        p = np.reshape(train_new1, (1, 186))

        predictions = loaded_model.predict(np.reshape(p, (1, 186, 1)))

        print(np.argmax(predictions, axis=1))
        plt.plot(np.reshape(p,(186,1)))
        plt.show()

        train_y.append(1)