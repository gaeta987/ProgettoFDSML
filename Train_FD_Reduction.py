import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from keras.utils.np_utils import to_categorical
from keras.layers import Convolution1D, MaxPool1D, Dense, Input, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from itertools import cycle, islice
import time
from sklearn.model_selection import train_test_split

def train_model(X_train, y_train, X_test, y_test):
    # input signal image shape
    im_shape = (X_train.shape[1], 1)

    # Input layer
    inputs_cnn = Input(shape=(im_shape),
                       name='inputs_cnn')

    # Block 1
    conv1_1 = Convolution1D(64, (6), activation='relu',
                            input_shape=im_shape)(inputs_cnn)
    conv1_1 = BatchNormalization()(conv1_1)

    pool1 = MaxPool1D(pool_size=(3), strides=(2),
                      padding='same')(conv1_1)

    # Block 2
    conv2_1 = Convolution1D(64, (3), activation='relu',
                            input_shape=im_shape)(pool1)
    conv2_1 = BatchNormalization()(conv2_1)

    pool2 = MaxPool1D(pool_size=(3), strides=(2),
                      padding='same')(conv2_1)

    # Block 3
    conv3_1 = Convolution1D(64, (3), activation='relu',
                            input_shape=im_shape)(pool2)
    conv3_1 = BatchNormalization()(conv3_1)

    pool3 = MaxPool1D(pool_size=(3), strides=(2),
                      padding='same')(conv3_1)
    # Flatten
    flatten = Flatten()(pool3)

    # Dense Block
    dense1 = Dense(64, activation='relu')(flatten)
    dense2 = Dense(32, activation='relu')(dense1)

    # Output Block
    output = Dense(5, activation='softmax', name='output')(dense2)

    # compile model
    model = Model(inputs=inputs_cnn, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
                 ModelCheckpoint(filepath='best-model.h5',
                                 monitor='val_loss',
                                 save_best_only=True)]
    # training
    print('Training...')
    history = model.fit(X_train, y_train, epochs=40, batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)

    model.load_weights('best-model.h5')

    return (model, history)

def evaluate_model(history, X_test, y_test, model):
    scores = model.evaluate((X_test), y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

    fig2, ax_loss = plt.subplots()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model - Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()
    target_names = [str(i) for i in range(5)]

    y_true = []
    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba = model.predict(X_test)
    prediction = np.argmax(prediction_proba, axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)

def add_gaussian_noise(signal):
    noise = np.random.normal(0, 0.05, 179)
    return signal+noise

train_df = pd.read_csv('mitbih_train.csv', header=None)
#test_df = pd.read_csv('mitbih_test.csv', header=None)

print(train_df.head())

print(train_df.info())

#train_df = train_df.drop([1,4,5,6,8,9,10],axis=1)
#train_df.columns = range(train_df.shape[1])

#test_df = test_df.drop([1,4,5,6,8,9,10],axis=1)
#test_df.columns = range(test_df.shape[1])

class_dist = train_df[180].astype(int).value_counts()
print(class_dist)

print(class_dist.mean())

my_colors = list(islice(cycle(['orange', 'r', 'g', 'y', 'k']), None, len(train_df)))

p = train_df[180].astype(int).value_counts().plot(kind='bar', title='Count (target)', color=my_colors);
plt.title('Class Distribution: Pre Random-Sampling')
plt.show()

#train_df_new
df_0 = train_df[train_df[180] == 0].sample(n=20000, random_state=13)
df_1 = resample(train_df[train_df[180] == 1], n_samples=20000,replace=True,
                                           random_state=13)
df_2 = resample(train_df[train_df[180] == 2], n_samples=20000,replace=True,
                                           random_state=13)
df_3 = resample(train_df[train_df[180] == 3], n_samples=20000,replace=True,
                                           random_state=13)
df_4 = resample(train_df[train_df[180] == 4], n_samples=20000,replace=True,
                                           random_state=13)

train_df_new = pd.concat([df_0, df_1, df_2, df_3, df_4])

train_df = train_df_new.drop([1,4,5,6,8,9,10],axis=1)
train_df.columns = range(train_df.shape[1])

p = train_df_new[180].astype(int).value_counts().plot(kind='bar', title='Count (target)', color=my_colors);
plt.title('Class Distribution: Post Random-Sampling')
plt.show()

c = train_df_new.groupby(180, group_keys=False)\
        .apply(lambda train_df_new: train_df_new.sample(1))
print(c)

fig, axes = plt.subplots(5, 1, figsize=(16, 15))

leg = iter(['N', 'S', 'V', 'F', 'U'])
colors = iter(['skyblue', 'red', 'lightgreen', 'orange', 'black'])
for i, ax in enumerate(axes.flatten()):
    ax.plot(c.iloc[i, :179].T, color=next(colors))
    ax.legend(next(leg))
plt.show()

plt.figure(figsize=(14, 7))
tempo = c.iloc[0, :179]
bruiter = add_gaussian_noise(tempo)

# tempo
plt.subplot(2,1,1)
plt.plot(tempo)

plt.title('ECG: BEFORE Gaussion noise additon')

#bruiter
plt.subplot(2,1,2)
plt.plot(bruiter)

plt.title('ECG: AFTER Gaussion noise additon')

plt.show()

# data prepapration : Labels
target_train = train_df_new[180]
target_test = test_df[180]

y_train = to_categorical(target_train)
#y_test = to_categorical(target_test)
# data preparation : Features
X_train = train_df_new.iloc[:,:179].values[:,:, np.newaxis]
#X_test = test_df.iloc[:,:179].values[:,:, np.newaxis]

#start to pick timing
start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size = 0.2, random_state = 42)

model, history = train_model(X_train, y_train, X_test, y_test)

evaluate_model(history, X_test, y_test, model)
y_pred = model.predict(X_test)

y_pred_clean = np.zeros_like(y_pred)
for idx, i in enumerate(np.argmax(y_pred,axis=1)):
    y_pred_clean[idx][i] = 1

print(classification_report(y_test, y_pred_clean))

conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred_clean, axis=1))
print(conf_matrix)

plt.figure(figsize=(10, 7))
sns.heatmap(np.corrcoef(conf_matrix))
plt.title('Confusion Matrix Corrleation-Coefficient')

#end time
print("--- %s seconds ---" % (time.time() - start_time))

