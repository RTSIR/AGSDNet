# import packages

from conf import myConfig as config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D,Activation,Input,Add,Subtract,AveragePooling2D,Multiply,Concatenate,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import numpy as np
from numpy import array
from numpy.linalg import norm
import tensorflow as tf
from numpy import *
import random
from skimage.util import random_noise

# custom filter
def my_Hfilter(shape, dtype=None):

    f = np.array([
            [[[-1]], [[0]], [[1]]],
            [[[-2]], [[0]], [[2]]],
            [[[-1]], [[0]], [[1]]]
        ])
    assert f.shape == shape
    return K.variable(f, dtype='float32')
    
def my_Vfilter(shape, dtype=None):

    f = np.array([
            [[[-1]], [[-2]], [[-1]]],
            [[[0]], [[0]], [[0]]],
            [[[1]], [[2]], [[1]]]
        ])
    assert f.shape == shape
    return K.variable(f, dtype='float32')

# create CNN model
input_img=Input(shape=(None,None,1))
Gh=Conv2D(filters=1, kernel_size = 3, kernel_initializer=my_Hfilter, padding='same')(input_img)
Gv=Conv2D(filters=1, kernel_size = 3, kernel_initializer=my_Vfilter, padding='same')(input_img)
Gx=K.sqrt(Gh*Gh + Gv*Gv)

x=Activation('relu')(input_img)
x=Conv2D(64,(3,3), dilation_rate=1,padding="same")(x)

x1 = Concatenate()([x,Gx])

x=Activation('relu')(x1)
x=Conv2D(65,(3,3), dilation_rate=2,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(65,(3,3), dilation_rate=3,padding="same")(x)

x=Conv2D(65,(3,3), padding="same")(x)
x_temp2=Activation('relu')(x)
x=AveragePooling2D(pool_size=(1, 1))(x_temp2)
x=Conv2D(65,(1,1), padding="same")(x)
x_temp3 = Add()([x, x_temp2])

x=Activation('relu')(x_temp3)
x=Conv2D(65,(3,3), dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(65,(3,3), dilation_rate=2,padding="same")(x)

x=Activation('relu')(x)
x2=Conv2D(65,(3,3), dilation_rate=1,padding="same")(x)

x = Add()([x2, x1])

x = Concatenate()([x,x1])

x3 = Conv2D(65,(3,3), padding="same")(x)
# CAB
x=AveragePooling2D(pool_size=(1, 1))(x3)
x=Activation('relu')(x)
x = Conv2D(65,(3,3), dilation_rate=1,padding="same")(x)
x = Conv2D(65,(3,3), dilation_rate=3,padding="same")(x)
x = Conv2D(65,(3,3), dilation_rate=1,padding="same")(x)
x=Activation('sigmoid')(x)
x4 = Multiply()([x,x3])

# PAB
x=Activation('relu')(x3)
x = Conv2D(65,(3,3), dilation_rate=1,padding="same")(x)
x = Conv2D(65,(3,3), dilation_rate=3,padding="same")(x)
x = Conv2D(65,(3,3), dilation_rate=1,padding="same")(x)
x=Activation('sigmoid')(x)
x5 = Multiply()([x,x3])

x = Concatenate()([x4, x5])
x = Conv2D(65,(3,3), padding="same")(x)
x = Add()([x, x1])

x = Conv2D(65,(3,3), padding="same")(x)
x = Add()([x, x1])

x=Conv2D(1,(3,3),padding="same")(x)
x6 = Add()([x, input_img])
model = Model(inputs=input_img, outputs=x6)

# load the data and normalize it
cleanImages=np.load(config.data)
print(cleanImages.dtype)
cleanImages=cleanImages/255.0
cleanImages=cleanImages.astype('float32')

# define augmentor and create custom flow
aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest", validation_split=0.2)

def myFlow(generator,X):
    for batch in generator.flow(x=X,batch_size=config.batch_size,seed=0):
        m,n,o,p=batch.shape
        looks=random.randint(1,21)
        stack=np.zeros((m,n,o,p,looks)) 
        for j in range(0,looks):
            stack[:,:,:,:,j] = random_noise(batch, mode='speckle') 
        noisyImagesBatch=np.mean(stack,axis=4)
        yield(noisyImagesBatch,batch)

# create custom learning rate scheduler
def lr_decay(epoch):
    initAlpha=0.0001
    factor=0.5
    dropEvery=25
    alpha=initAlpha*(factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)
callbacks=[LearningRateScheduler(lr_decay)]

# create custom loss, compile the model
print("[INFO] compilingTheModel")
opt=optimizers.Adam(learning_rate=0.0001)
def custom_loss(y_true,y_pred):
    diff=abs(y_true-y_pred)
    #l1=K.sum(diff)/(config.batch_size)
    l1=(diff)/(config.batch_size)
    return l1
model.compile(loss=custom_loss,optimizer=opt)

# train
model.fit_generator(myFlow(aug,cleanImages),
epochs=config.epochs,steps_per_epoch=len(cleanImages)//config.batch_size,callbacks=callbacks,verbose=1)

# save the model
model.save('./Pretrained_models/AGSDNet.h5')
