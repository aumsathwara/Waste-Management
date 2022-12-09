import tensorflow as tf
import keras
from keras.layers import Dense,Dropout,Activation,Add,MaxPooling2D,Conv2D,Flatten,BatchNormalization,MaxPool2D
from keras.models import Sequential
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing import image
import numpy as np

#import train data
train_datagen = ImageDataGenerator(rescale=1/255,shear_range = 0.1,
                                   zoom_range = 0.3,horizontal_flip = True,
                                   vertical_flip =  True ,
                                   rotation_range=50,

                                   brightness_range = (0.25, 1.3))


`train_data = train_datagen.flow_from_directory`('C:/Users/Aum Sathwara/Documents/Aum/Workplace/Python/Waste Management/DATASET/TRAIN',target_size = (250, 250),class_mode='sparse',shuffle=True,seed=1)
#import test data

test_datagen = ImageDataGenerator(rescale = 1/255)
test_data = test_datagen.flow_from_directory("C:/Users/Aum Sathwara/Documents/Aum/Workplace/Python/Waste Management/DATASET/TEST",batch_size=32,target_size=(250,250),class_mode='sparse',shuffle=True,seed=1)

classes= ["Oragnic","Recycle"]
model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = (250,250,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation("softmax"))
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])
early =  tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="auto")
#fit our model
his = model.fit( train_data,
        epochs=1,
        validation_data = test_data, callbacks = [early])
#evulate model
model.evaluate(test_data)

#plotting training values
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

acc = his.history['accuracy']
val_acc = his.history['val_accuracy']
loss = his.history['loss']
val_loss = his.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
image_path = "C:/Users/Aum Sathwara/Documents/Aum/Workplace/Python/Waste Management/DATASET/TEST/O/O_13798.jpg"
new_img = image.load_img(image_path, target_size=(250, 250))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
prediction = np.argmax(prediction,axis=1)
print(prediction)
print(classes[prediction[0]])
plt.imshow(new_img)