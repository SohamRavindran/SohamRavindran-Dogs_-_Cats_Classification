import pandas
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Data preprocessing

train_datagen = ImageDataGenerator(rescale=1 / 255 , shear_range=0.2 , zoom_range=0.2 , horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    'D:/Data Science/Dogs_and_ cats_classification/training set' ,
    target_size=(64 , 64) ,
    batch_size=32 ,
    class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    'D:/Data Science/Dogs_and_ cats_classification/test set' ,
    target_size=(64 , 64) ,
    batch_size=32 ,
    class_mode='binary')
# Initialise the CNN model
cnn = tf.keras.models.Sequential()

# Convulutional
cnn.add(tf.keras.layers.Conv2D(filters=32 , kernel_size=3 , activation='relu' , input_shape=[64 , 64 , 3]))

# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2 , strides=2))

# Additional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32 , kernel_size=3 , activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2 , strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128 , activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1 , activation='sigmoid'))

# Compile the model
cnn.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])

# train the fit
history = cnn.fit(x=train_generator , validation_data=validation_generator , epochs=5)
# Prediction with the model using a test image
test_image = image.load_img("D:/Data Science/Dogs_and_ cats_classification/test image.jpg" , target_size=(64 , 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image , axis=0)
results = cnn.predict(test_image)
train_generator.class_indices
if results[0][0]:
    prediction = "Dog"
else:
    prediction = "Cat"

print(prediction)
