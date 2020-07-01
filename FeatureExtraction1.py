import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.engine.saving import model_from_json
import scipy.signal
import plotly.graph_objects as go

def extract_feat(image, begin, end):
    x_list, y_list = [], [0]   # boundary padding add '0'
    for x in np.arange(begin, end, 1):
        x_list.append(x-begin)
        for y in np.arange(0, 700, 1):
            #if np.all(image[y][x] == (0, 0, 0)):
            if np.all(image[y][x] == (255, 255, 255)):
                y_list.append(700-y)
                break
            if y==699:
                y_list.append(y_list[x-begin])
    y_list.pop(0)   # remove boundary padding '0'
    #show_graph(x_list, y_list, 5, 5)

    return y_list

def show_graph(x_list, y_list, width, height):
    plt.figure(figsize = [width, height])
    plt.scatter(x_list, y_list, marker='.', s=5)
    plt.show()
    return

image = cv2.imread('provetta.jpg')
print(image)
image = cv2.resize(image, (700,700), interpolation=cv2.INTER_CUBIC)

x_list, y_list = [], []
for x in np.arange(0, 700, 1):
    for y in np.arange(0, 700, 1):
        #if np.all(image[y][x] == (0, 0, 0)):
        if np.all(image[y][x] == (255,255,255)):
            x_list.append(x)
            y_list.append(700-y)

show_graph(x_list,y_list,18,3)

print('Detect peaks without any filters.')
peaks, _ = scipy.signal.find_peaks(y_list, height = 500)

print(peaks)

cv2.imshow('prova',image)
cv2.waitKey()

for i in range(len(peaks)-1):
    print(y_list[peaks[i]])

extrac = extract_feat(image,
                peaks[0]-90, peaks[0]+96)

json_file = open('best_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("best-model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_new = np.reshape(extrac, (186, 1))

scaler = MinMaxScaler(feature_range=(0, 1))

train_new1 = scaler.fit_transform(train_new)

p = np.reshape(train_new1, (1, 186))

predictions = loaded_model.predict(np.reshape(p, (1, 186, 1)))

print(np.argmax(predictions, axis=1))
plt.plot(np.reshape(p, (186, 1)))
plt.show()















