import keras
from keras.datasets import mnist
from keras import models
from keras import layers


# 1. Definir dataset:
# Descarga:
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Adecuamiento de los datos:
# Adaptar las dimensiones:
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# Normalizar los valores entre 0 y 1:
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# Convertir las etiquetas en formato one-hot:
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)




# 2. Diseñar arquitectura de red:
model = models.Sequential() # modelo secuencial => tipos de modelos vienen dados por keras: https://keras.io/api/

#añadimos capas al modelo 
model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), input_shape=(28,28,1)))  #convolucion
model.add(layers.Activation('relu')) #activacion 
model.add(layers.MaxPooling2D()) #pulling 

#para hacer el modelo de arquitectura LeNet
# model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), input_shape=(10,10,1))) #convolucion
# model.add(layers.Activation('relu')) #activacion 
# model.add(layers.MaxPooling2D()) #pulling 

model.add(layers.Flatten())
model.add(layers.Dense(units=84))
model.add(layers.Activation('relu'))
model.add(layers.Dense(units=10))
model.add(layers.Activation('softmax'))




# 3. Compilar: https://keras.io/api/optimizers/ 
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])




# 4. Entrenar y validar:
train = model.fit(train_images, 
	train_labels, 
	epochs=5, 
	batch_size=32, 	
	validation_data=(test_images, test_labels))

# batch size => la cantidad de imagenes que cargan. 

# 5. Test:
score = model.evaluate(test_images, test_labels)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
