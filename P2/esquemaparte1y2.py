# -*- coding: utf-8 -*-

#########################################################################
############ CARGAR LAS LIBRERÍAS NECESARIAS ############################
#########################################################################
#Libraries
import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.utils as np_utils
#Optimizer
from keras.optimizers import SGD
#Dataset
from keras.datasets import cifar100


# A completar: esquema disponible en las diapositivas

#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

#Loading Cifar100. ImgSize = (32,32,3)
def loadImgs():
    (x_train,y_train), (x_test,y_test) = cifar100.load_data(label_mode = 'fine')
    # Change type to float32
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    #Normalize to 255
    x_train /= 255
    x_test  /= 255

    #
    train_idx = np.isin(y_train,np.arange(25))
    train_idx = np.reshape(train_idx,-1)
    x_train   = x_train[train_idx]
    y_train   = y_train[train_idx]

    # Transform class vectors into matrix (0,..,1,..,0) in i-th position
    y_train = np_utils.to_categorical(y_train,25)
    y_test  = np.utils.to_categorical(y_test,25)

    return x_train , y_train , x_test , y_test


#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

def calculateAccuracy(labels,pred):
    labels = np.argmax(labels,axis = 1)
    preds  = np.argmax(preds ,axis = 1)

    accuracy = sum(labbels == preds)/len(labels)

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

def showEvolution(hist):

    #Loss
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Trainig loss', 'Validation loss'])
    plt.show()

    #Accuracy
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training accuracy', 'Validation accuracy'])
    plt.show()


#########################################################################
################## DEFINICIÓN DEL MODELO BASENET ########################
#########################################################################

# A completar

model = Sequential()

#Layer 1 and 2
model.add(Conv2d(32,kernel_size = (5,5), activation='relu'))
#Layer 3
model.add(MaxPooling2D(pool_size = (2,2)))
#Layer 4 and 5
model.add(Conv2d(32,kernel_size = (5,5), activation='relu'))
#Layer 6
model.add(MaxPooling2D(pool_size = (2,2)))
#Layer 7

#Layer 8

#Layer 9


#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

# A completar


# Una vez tenemos el modelo base, y antes de entrenar, vamos a guardar los
# pesos aleatorios con los que empieza la red, para poder reestablecerlos
# después y comparar resultados entre no usar mejoras y sí usarlas.
weights = model.get_weights()

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

# A completar

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

# A completar

#########################################################################
########################## MEJORA DEL MODELO ############################
#########################################################################

# A completar. Tanto la normalización de los datos como el data
# augmentation debe hacerse con la clase ImageDataGenerator.
# Se recomienda ir entrenando con cada paso para comprobar
# en qué grado mejora cada uno de ellos.
