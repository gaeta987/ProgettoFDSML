import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from keras.utils.np_utils import to_categorical
from keras.layers import Convolution1D, MaxPool1D, Dense, Input, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model

train_df = pd.read_csv('mitbih_train.csv', header=None)
test_df = pd.read_csv('mitbih_test.csv', header=None)

train_df.head()

train_df.info()

class_dist = train_df[187].astype(int).value_counts()
class_dist

class_dist.mean()

plt.figure(figsize=(10, 7))
p = class_dist.plot(kind='pie',
                    labels=['N','S','V','F','Q'],
                    autopct='%1.1f%%')
p.add_artist(plt.Circle((0,0), 0.7, color='white'))
plt.title('Class Distribution')
plt.legend()
plt.show()
plt.savefig('Origial Class Distribution.PNG')

#train_df_new
df_0 = train_df[train_df[187] == 0].sample(n=20000, random_state=8)
df_1 = resample(train_df[train_df[187] == 1], n_samples=20000,replace=True,
                                           random_state=8)
df_2 = resample(train_df[train_df[187] == 2], n_samples=20000,replace=True,
                                           random_state=8)
df_3 = resample(train_df[train_df[187] == 3], n_samples=20000,replace=True,
                                           random_state=8)
df_4 = resample(train_df[train_df[187] == 4], n_samples=20000,replace=True,
                                           random_state=8)
train_df[train_df[187]==4]

train_df_new = pd.concat([df_0, df_1, df_2, df_3, df_4])

plt.figure(figsize=(10, 7))
p = train_df_new[187].value_counts().plot(kind='pie',
                    labels=['N','S','V','F','Q'],
                    autopct='%1.1f%%')
p.add_artist(plt.Circle((0,0), 0.7, color='white'))
plt.title('Class Distribution: Post Random-Sampling')
plt.legend()
plt.show()
plt.savefig('post random sampling class dist.PNG')
