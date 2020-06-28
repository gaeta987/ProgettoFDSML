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

    model_json = model.to_json()
    with open("best_model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("best-model.h5")
    print("Saved model to disk")

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
    noise = np.random.normal(0, 0.05, 186)
    return signal+noise

def dsGaussian(train):
    train_new = pd.DataFrame()
    for i in range(len(train)):
        tempo = train.iloc[i, :186]
        bruiter = add_gaussian_noise(tempo)

        plt.figure(figsize=(14, 7))

        # tempo
        plt.subplot(5, 1, 1)
        plt.plot(bruiter)

        plt.title('ECG: AFTER Gaussion noise additon')

        for j in range(len(bruiter)):
            if bruiter[j] < 0:
                bruiter[j] = bruiter[j] * -1
        print(bruiter)

        # tempo
        plt.subplot(5, 1, 3)
        plt.plot(tempo)

        plt.title('ECG: BEFORE Gaussion noise additon')

        # bruiter
        plt.subplot(5, 1, 5)
        plt.plot(bruiter)

        plt.title('ECG: AFTER Gaussion noise additon and AFTER Normalization')

        plt.show()

        train_new.append(bruiter)

    print(train_new)

train_df = pd.read_csv('mitbih_train.csv', header=None)
#test_df = pd.read_csv('mitbih_test.csv', header=None)

#dsGaussian(train_df)

print(train_df.head())

print(train_df.info())

class_dist = train_df[187].astype(int).value_counts()
print(class_dist)

print(class_dist.mean())

my_colors = list(islice(cycle(['orange', 'r', 'g', 'y', 'k']), None, len(train_df)))

p = train_df[187].astype(int).value_counts().plot(kind='bar', title='Count (target)', color=my_colors);
plt.title('Class Distribution: Pre Random-Sampling')
plt.show()

#train_df_new
df_0 = train_df[train_df[187] == 0].sample(n=20000, random_state=13)
df_1 = resample(train_df[train_df[187] == 1], n_samples=20000,replace=True,
                                           random_state=13)
df_2 = resample(train_df[train_df[187] == 2], n_samples=20000,replace=True,
                                           random_state=13)
df_3 = resample(train_df[train_df[187] == 3], n_samples=20000,replace=True,
                                           random_state=13)
df_4 = resample(train_df[train_df[187] == 4], n_samples=20000,replace=True,
                                           random_state=13)

train_df_new = pd.concat([df_0, df_1, df_2, df_3, df_4])

p = train_df_new[187].astype(int).value_counts().plot(kind='bar', title='Count (target)', color=my_colors);
plt.title('Class Distribution: Post Random-Sampling')
plt.show()

c = train_df_new.groupby(187, group_keys=False)\
        .apply(lambda train_df_new: train_df_new.sample(1))
print(c)

fig, axes = plt.subplots(5, 1, figsize=(16, 15))

leg = iter(['N', 'S', 'V', 'F', 'U'])
colors = iter(['skyblue', 'red', 'lightgreen', 'orange', 'black'])
for i, ax in enumerate(axes.flatten()):
    ax.plot(c.iloc[i, :186].T, color=next(colors))
    ax.legend(next(leg))
plt.show()

plt.figure(figsize=(14, 7))
tempo = c.iloc[0, :186]
bruiter = add_gaussian_noise(tempo)

'''for i in range(len(bruiter)):
    if bruiter[i] < 0:
        bruiter[i] = bruiter[i] * -1
print(bruiter)'''

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
target_train = train_df_new[187]

#target_test = test_df[187]
y_train = to_categorical(target_train)

# data preparation : Features
X_train = train_df_new.iloc[:,:186].values[:,:, np.newaxis]

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
plt.show()



