import sys
import argparse
import time
import cv2
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.engine.saving import model_from_json
import scipy.signal
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

currentDT = time.localtime()
start_datetime = time.strftime("-%m-%d-%H-%M-%S", currentDT)

def crop(image, w, f):
    return image[:, int(w * f): int(w * (1 - f))]

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

def start():
    device = 0
    frame_rate_ratio = 7
    process_speed = 1
    mirror = True


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

        imgForText = cv2.resize(canvas, (700, 700), interpolation=cv2.INTER_CUBIC)

        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV).astype("float32")

        (h, s, v) = cv2.split(img)
        s = s * 0
        s = np.clip(s, 0, 255)
        img = cv2.merge([h, s, v])

        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_HSV2BGR)

        image = cv2.resize(img, (700, 700), interpolation=cv2.INTER_CUBIC)

        x_list, y_list = [], []
        for x in np.arange(0, 700, 1):
            for y in np.arange(0, 700, 1):
                # if np.all(image[y][x] == (0, 0, 0)):
                if np.all(image[y][x] == (255, 255, 255)):
                    x_list.append(x)
                    y_list.append(700 - y)

        peaks, _ = scipy.signal.find_peaks(y_list, height=400)

        if(len(peaks) != 0):


            extrac = extract_feat(image,
                                  peaks[1] - 90, peaks[1] + 96)

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

            if str(np.argmax(predictions, axis=1)[0]) == '0':
                output = 'N'
            elif str(np.argmax(predictions, axis=1)[0]) == '1':
                output = 'S'
            elif str(np.argmax(predictions, axis=1)[0]) == '2':
                output = 'V'
            elif str(np.argmax(predictions, axis=1)[0]) == '3':
                output = 'F'
            elif str(np.argmax(predictions, axis=1)[0]) == '4':
                output = 'U'

            cv2.putText(imgForText, output, (x_list[peaks[0]], 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0)
            cv2.putText(imgForText, 'prob: ' + str(round(predictions[0][np.argmax(predictions, axis=1)[0]], 2)),
                        (x_list[peaks[0]], 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.imshow('frame', imgForText)
        ret_val, orig_image = cam.read()

        i += 1



