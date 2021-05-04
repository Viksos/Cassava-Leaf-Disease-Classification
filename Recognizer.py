
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB5
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import seaborn as sns
from sklearn.utils import class_weight
import cv2

###########################################3


train = pd.read_csv('train.csv')
train.label = train.label.astype('str')
names_of_disease = pd.read_json('label_num_to_disease_map.json', typ='series')

sns.countplot(train["label"])
plt.show()

print(train.label.value_counts())

img_size = 128

data = ImageDataGenerator(validation_split=0.2)                            

train_data = data.flow_from_dataframe(
    dataframe=train,
    directory='train_images',
    x_col='image_id',
    y_col='label',
    target_size=(img_size, img_size),
    batch_size=32,
    subset='training',
    shuffle = True,
    class_mode='categorical'
)

valid_data = data.flow_from_dataframe(
    dataframe=train,
    directory='train_images',
    x_col='image_id',
    y_col='label',
    target_size=(img_size, img_size),
    batch_size=32,
    subset='validation',
    class_mode = 'categorical',
    shuffle = True
)

filepath = "best.model.hdf5"
checkpoint = ModelCheckpoint(filepath, 
                             save_best_only = True, 
                             save_weights_only = True,
                             monitor = 'val_loss', 
                             mode = 'min', verbose = 1)

early_stopping = EarlyStopping(monitor='val_loss', mode="min", verbose=1, patience=15)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                              patience = 3,
                              mode = 'min', verbose = 1)

model = Sequential()
model.add(EfficientNetB5(include_top = False, weights = "imagenet",
                        input_shape=(img_size, img_size, 3)))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3)))
model.add(tf.keras.layers.Flatten())
model.add(Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation = "relu"))
model.add(tf.keras.layers.Dense(64, activation = "relu"))
model.add(tf.keras.layers.Dense(5, activation = "softmax"))
model.compile(optimizer = 'adam',
            loss = "categorical_crossentropy",
            metrics = ["accuracy"])

model.summary()

#class_weights = class_weight.compute_class_weight('balanced', np.unique(train_data.classes),train_data.classes)
class_weights = {0:3.85975197,1: 1.95299487,2: 1.81047065,3: 0.32543726,4: 1.6563135}

history = model.fit(train_data,
                            epochs = 50,
                            steps_per_epoch = len(train_data)/8,
                            validation_data = valid_data,
                            validation_steps = len(valid_data)/8,
                            class_weight=class_weights,
                            callbacks = [checkpoint, reduce_lr,early_stopping])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_weights.hdf5")

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy vs Val-Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train acc', 'val acc'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss vs Val-Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train loss', 'val loss'])
plt.show()

submission = pd.read_csv("sample_submission.csv")
prediction = []

for image_id in submission.image_id:
    image = cv2.cvtColor(cv2.imread('test_images/'+image_id),cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(img_size,img_size))
    image = np.expand_dims(image, axis = 0)
    prediction.append(np.argmax(model.predict(image)))

submission['label'] = prediction
submission.to_csv('submission.csv', index = False)

print(submission)
