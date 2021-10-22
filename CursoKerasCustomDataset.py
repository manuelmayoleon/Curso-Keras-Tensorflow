import keras
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('C:\\Users\\LDuran\\Downloads\\dataset_cats_and_dogs\\train', target_size = (128,128), batch_size = 32)
val_generator = val_datagen.flow_from_directory('C:\\Users\\LDuran\\Downloads\\dataset_cats_and_dogs\\val', target_size = (128,128), batch_size = 32)
test_generator = test_datagen.flow_from_directory('C:\\Users\\LDuran\\Downloads\\dataset_cats_and_dogs\\test', target_size = (128,128), batch_size = 32)

model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), input_shape=(128,128,3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), input_shape=(128,128,3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(units=512))
model.add(layers.Activation('relu'))
model.add(layers.Dense(units=2))
model.add(layers.Activation('softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

train = model.fit_generator(train_generator, epochs=20, validation_data = val_generator)


ent_acc = train.history['accuracy']
val_acc = train.history['val_accuracy']
ent_loss = train.history['loss']
val_loss = train.history['val_loss']
epochs = range(len(ent_acc))
plt.plot(epochs, ent_acc, 'bo', label='Entrenamiento')
plt.plot(epochs, val_acc, 'b', label='Validacion')
plt.title('Accuracy Entrenamiento y Validacion')
plt.legend()
plt.figure()
plt.plot(epochs, ent_loss, 'bo', label='Entrenamiento')
plt.plot(epochs, val_loss, 'b', label='Validacion')
plt.title('Loss Entrenamiento y Validacion')
plt.legend()
plt.show()

"""
test_loss, test_acc = model.evaluate_generator(test_generator)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
"""