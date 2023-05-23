# cerber NN for semantic segmentation
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Add, Conv2D, MaxPooling2D,Conv2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import os, glob
from PIL import Image

seed=777
np.random.seed(seed) 
tf.random.set_seed(seed)

# data preparation for semantic segmentation purpose (image as label)

train_peak_dir=os.chdir(r"C:/Users/TK/Desktop/spider_nn/data_generator/datanew/train/images/")
extension="*.bmp"
train_peaks=list()
for file in glob.glob(extension):
   image=Image.open(file)
   train_peaks.append(np.asarray(np.stack((image,)*3,axis=-1),dtype=np.float32))
train_peaks=np.asarray(train_peaks)/255.
train_peaks=tf.convert_to_tensor(train_peaks.astype(np.float32))   
# train_peaks=tf.cast(train_peaks, dtype=tf.float32)
print(tf.shape(train_peaks))

train_mask_dir=os.chdir(r"C:/Users/TK/Desktop/spider_nn/data_generator/datanew/train/masks/")
extension="*.bmp"
train_masks=[]
for file in glob.glob(extension):
   mask=Image.open(file)
   train_masks.append(np.asarray(np.stack((mask,)*1,axis=-1),dtype=np.float32))
train_masks=np.asarray(train_masks)/255.
train_masks=tf.convert_to_tensor(train_masks.astype(np.float32)) 
# train_masks=tf.cast(train_masks, dtype=tf.float32)
print(tf.shape(train_masks))

test_peak_dir=os.chdir(r"C:/Users/TK/Desktop/spider_nn/data_generator/datanew/test/images/")
extension="*.bmp"
test_peaks=[]
for file in glob.glob(extension):
   image=Image.open(file)
   test_peaks.append(np.asarray(np.stack((image,)*3,axis=-1),dtype=np.float32))
test_peaks=np.asarray(test_peaks)/255.
test_peaks=tf.convert_to_tensor(test_peaks.astype(np.float32))
# test_peaks=tf.cast(test_peaks, tf.float32)
print(tf.shape(test_peaks))

test_mask_dir=os.chdir(r"C:/Users/TK/Desktop/spider_nn/data_generator/datanew/test/masks/")
extension="*.bmp"
test_masks=[]
for file in glob.glob(extension):
   image=Image.open(file)
   test_masks.append(np.asarray(np.stack((image,)*1,axis=-1),dtype=np.float32))
test_masks=np.asarray(test_masks)/255.
test_masks=tf.convert_to_tensor(test_masks.astype(np.float32))
# test_masks=tf.cast(test_masks, tf.float32)
print(tf.shape(test_masks))

# data preparation end

input_shape=tf.shape(train_peaks)
print(input_shape)

# data augmentation

datagen = ImageDataGenerator(
    rotation_range=350,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True, vertical_flip=True,zoom_range=2.5,brightness_range=(0.0,80.0))

# ++++++++network below++++++++

def cerber(batch_size,n):
    inputs=tf.keras.layers.Input(shape=(1608,1608,3), batch_size=batch_size)
    # 1st conv block
    # print(str(tf.shape(inputs))+"xc1")
   
    x=Conv2D(1,kernel_size=(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Activation("elu")(x)
    shortcut1c=x

    y=Conv2D(1,kernel_size=(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(inputs)
    y = tf.keras.layers.BatchNormalization()(y)
    y = Activation("elu")(y)
    shortcut1cy=y
    # print(str(y.shape)+"sc1")
    # dense block 0


    z=Conv2D(1,kernel_size=(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(inputs)
    z = tf.keras.layers.BatchNormalization()(z)
    z = Activation("elu")(z)
    shortcut1cz=z
    z=tf.keras.layers.Concatenate()([shortcut1c,shortcut1cy,z])
    y=tf.keras.layers.Concatenate()([shortcut1c,shortcut1cz,y])
    x=tf.keras.layers.Concatenate()([shortcut1cy,shortcut1cz,x])
    # print(str(x.shape)+"sc1")
    # dense block 0

    x=tf.keras.layers.Conv2D(2,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(x)
    shorcut0d=x
    z=tf.keras.layers.Conv2D(2,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(z)
    shorcut0dz=z
    y=tf.keras.layers.Conv2D(2,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(y)
    shorcut0dy=y
    y=tf.keras.layers.Concatenate()([shorcut0d,shorcut0dz,y])


    z=tf.keras.layers.Concatenate()([shorcut0dy,shorcut0d,z])


    x=tf.keras.layers.Concatenate()([shorcut0dy,shorcut0dz,x])
    # print(str(x.shape)+"sd0")
    x=tf.keras.layers.Conv2D(2*n,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+123),activation="softmax")(x)
    shortcut1d=x

    z=tf.keras.layers.Conv2D(2*n,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+123),activation="softmax")(z)
    shortcut1dz=z


    y=tf.keras.layers.Conv2D(2*n,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+123),activation="softmax")(y)
    shortcut1dy=y
    y=tf.keras.layers.Concatenate()([shortcut1d,shortcut1dz,y])
    z=tf.keras.layers.Concatenate()([shortcut1d,shortcut1dy,z])    
    x=tf.keras.layers.Concatenate()([shortcut1dy,shortcut1dz,x])
    # print(str(x.shape)+"xd0")
    # conv block 1 continuation
    z = Conv2D(2*n, kernel_size=(3,3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(z)
    shortcut2cz=z

    y = Conv2D(2*n, kernel_size=(3,3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(y)
    shortcut2cy=y

    x = Conv2D(2*n, kernel_size=(3,3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(x)
    shortcut2c=x

    x=tf.keras.layers.Concatenate()([shortcut2cy,shortcut2cz,x])
    y=tf.keras.layers.Concatenate()([shortcut2c,shortcut2cz,y])
    z=tf.keras.layers.Concatenate()([shortcut2cy,shortcut2c,z])
    # print(str(x.shape)+"xc2")
    x = tf.keras.layers.BatchNormalization()(x)
    x = Activation("elu")(x)
    x=Conv2D(2*n, kernel_size=(3,3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(x)
    shortcut3c=x

    z = tf.keras.layers.BatchNormalization()(z)
    z = Activation("elu")(z)
    z=Conv2D(2*n, kernel_size=(3,3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(z)
    shortcut3cz=z


    y = tf.keras.layers.BatchNormalization()(y)
    y = Activation("elu")(y)
    y=Conv2D(2*n, kernel_size=(3,3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(y)
    shortcut3cy=y
    y=tf.keras.layers.Concatenate()([shortcut3c,shortcut3cz,y])
    z=tf.keras.layers.Concatenate()([shortcut3c,shortcut3cy,z])
    x=tf.keras.layers.Concatenate()([shortcut3cy,shortcut3cz,x])
    # print(str(x.shape)+"xc3")
    x=MaxPooling2D((2, 2))(x)
    # print(str(x.shape)+"xcpool1")
    x=tf.keras.layers.BatchNormalization()(x)
    shortcutp1=x

    z=MaxPooling2D((2, 2))(z)
    # print(str(z.shape)+"zcpool1")
    z=tf.keras.layers.BatchNormalization()(z)
    shortcutp1z=z


    y=MaxPooling2D((2, 2))(y)
    # print(str(y.shape)+"ycpool1")
    y=tf.keras.layers.BatchNormalization()(y)
    shortcutp1y=y
    y=tf.keras.layers.Concatenate()([shortcutp1,shortcutp1z,y])
    z=tf.keras.layers.Concatenate()([shortcutp1,shortcutp1y,z])
    x=tf.keras.layers.Concatenate()([shortcutp1y,shortcutp1z,x])
    # dense block1
    x=tf.keras.layers.Conv2D(2,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(x)
    shorcut1d=x

    z=tf.keras.layers.Conv2D(2,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(z)
    shorcut1dz=z
    y=tf.keras.layers.Conv2D(2,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(y)
    shorcut1dy=y

    z=tf.keras.layers.Concatenate()([shorcut1d,shorcut1dy,z])
    x=tf.keras.layers.Concatenate()([shorcut1dy,shorcut1dz,x])
    y=tf.keras.layers.Concatenate()([shorcut1d,shorcut1dz,y])

    # print(str(z.shape)+"sd1")
    z=tf.keras.layers.Conv2D(2*n,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+123),activation="softmax")(z)
    y=tf.keras.layers.Conv2D(2*n,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+123),activation="softmax")(y)
    # print(str(x.shape)+"sd1")
    x=tf.keras.layers.Conv2D(2*n,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+123),activation="softmax")(x)



    # print(str(y.shape)+"sd1")

    # print(str(x.shape)+"xd1")
    # conv block2

    x=tf.keras.layers.BatchNormalization()(x)
    x=Activation("elu")(x)
    x = Conv2D(8*n, kernel_size=(3,3), padding='same',  kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(x)
    shortcut31c=x

    z=tf.keras.layers.BatchNormalization()(z)
    z=Activation("elu")(z)
    z = Conv2D(8*n, kernel_size=(3,3), padding='same',  kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(z)
    shortcut31cz=z


    y=tf.keras.layers.BatchNormalization()(y)
    y=Activation("elu")(y)
    y = Conv2D(8*n, kernel_size=(3,3), padding='same',  kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(y)
    shortcut31cy=y

    y=tf.keras.layers.Concatenate()([shortcut31c,shortcut31cz,y])
    z=tf.keras.layers.Concatenate()([shortcut31c,shortcut31cy,z])
    x=tf.keras.layers.Concatenate()([shortcut31cy,shortcut31cz,x])
    # print(str(x.shape)+"xc31")
    x = tf.keras.layers.BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2D(8*n, kernel_size=(3,3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(x)
    shortcut4c=x

    z = tf.keras.layers.BatchNormalization()(z)
    z = Activation("elu")(z)
    z = Conv2D(8*n, kernel_size=(3,3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(z)
    shortcut4cz=z


    y = tf.keras.layers.BatchNormalization()(y)
    y = Activation("elu")(y)
    y = Conv2D(8*n, kernel_size=(3,3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(y)
    shortcut4cy=y
    y=tf.keras.layers.Concatenate()([shortcut4c,shortcut4cz,y])
    z=tf.keras.layers.Concatenate()([shortcut4c,shortcut4cy,z])
    x=tf.keras.layers.Concatenate()([shortcut4cy,shortcut4cz,x])
    # print(str(x.shape)+"xc4")
    x = MaxPooling2D((2, 2))(x)
    # print(str(x.shape)+"pool2")
    # dense block2

    x=tf.keras.layers.Conv2D(2,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(x)

    x=tf.keras.layers.Conv2D(8*n,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+123),activation="softmax")(x)
    shortd8n=x

    z = MaxPooling2D((2, 2))(z)
    z=tf.keras.layers.Conv2D(2,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(z)

    z=tf.keras.layers.Conv2D(8*n,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+123),activation="softmax")(z)
    shortd8nz=z


    y = MaxPooling2D((2, 2))(y)

    y=tf.keras.layers.Conv2D(2,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(y)

    y=tf.keras.layers.Conv2D(8*n,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+123),activation="softmax")(y)
    shortd8ny=y
    y=tf.keras.layers.Concatenate()([shortd8n,shortd8nz,y])
    z=tf.keras.layers.Concatenate()([shortd8n,shortd8ny,z])
    x=tf.keras.layers.Concatenate()([shortd8ny,shortd8nz,x])

# dense layers bridge

    # input dense block - preparation
    d11=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+321))(x)
    d21=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+321))(y)
    d31=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+321))(z)

    # processing dense block
    d12=tf.keras.layers.Conv2D(n*n,(1,1),(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+321))(d11)
    d22=tf.keras.layers.Conv2D(n*n,(1,1),(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+321))(d21)
    d32=tf.keras.layers.Conv2D(n*n,(1,1),(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+321))(d31)

    # interconnections now
    ic13=tf.keras.layers.Concatenate()([d12,d22])
    ic12=tf.keras.layers.Concatenate()([d12,d32])
    ic11=tf.keras.layers.Concatenate()([d32,d22])

    # and now mixing
    m11=tf.keras.layers.Concatenate()([ic11,d12])
    m12=tf.keras.layers.Concatenate()([ic12,d22])
    m13=tf.keras.layers.Concatenate()([ic13,d32])

    # dense continuation
    d13=tf.keras.layers.Conv2D(n*n,(1,1),(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+321))(m11)
    d23=tf.keras.layers.Conv2D(n*n,(1,1),(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+321))(m12)
    d33=tf.keras.layers.Conv2D(n*n,(1,1),(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+321))(m13)

    # interconnections now
    ic23=tf.keras.layers.Concatenate()([d13,d23])
    ic22=tf.keras.layers.Concatenate()([d13,d33])
    ic21=tf.keras.layers.Concatenate()([d33,d23])

    # and now mixing
    m21=tf.keras.layers.Concatenate()([ic21,d13])
    m22=tf.keras.layers.Concatenate()([ic22,d23])
    m23=tf.keras.layers.Concatenate()([ic23,d33])

    # dense block 2 with interconnections
    d14=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+321))(m21)
    d24=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+321))(m22)
    d34=tf.keras.layers.Conv2D(n,(1,1),(1,1),activation="elu",kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+321))(m23)

    # interconnections now
    ic33=tf.keras.layers.Concatenate()([d14,d24])
    ic32=tf.keras.layers.Concatenate()([d14,d34])
    ic31=tf.keras.layers.Concatenate()([d34,d24])

    # and now mixing
    m31=tf.keras.layers.Concatenate()([ic31,d14])
    m32=tf.keras.layers.Concatenate()([ic32,d24])
    m33=tf.keras.layers.Concatenate()([ic33,d34])

    x=tf.keras.layers.Concatenate()([m31,m32,m33])

# dense layers bridge end
# dense block 3
    x=tf.keras.layers.Conv2D(2,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(x)
    x=tf.keras.layers.Conv2D(8*n,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+123),activation="softmax")(x)
    x=tf.keras.layers.Concatenate()([shortd8n,shortd8ny,shortd8nz,x])
    # print(str(x.shape)+"xd3")
    # deconv block 1
    x = tf.keras.layers.BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2DTranspose(8*n, (3, 3), strides=(2, 2), padding='same')(x)
    x=tf.keras.layers.Concatenate()([shortcut31c,shortcut4c,shortcut31cy,shortcut4cy,shortcut31cz,shortcut4cz,x])
    # print(str(x.shape)+"xdc1")

    x = tf.keras.layers.BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2D(8*n, (3, 3), padding='same',kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(x)
    x=tf.keras.layers.Concatenate()([shortcut31c,shortcut4c,shortcut31cy,shortcut4cy,shortcut31cz,shortcut4cz,x])
    # print(str(x.shape)+"xdc1.1")
    x = tf.keras.layers.BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2D(8*n, (3, 3), padding='same',kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(x)
    x=tf.keras.layers.Concatenate()([shortcut31c,shortcut4c,shortcut31cy,shortcut4cy,shortcut31cz,shortcut4cz,x])
  # print(str(x.shape)+"xdc1.2")  


# dense block 4
    x=tf.keras.layers.Conv2D(2,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(x)
    x=tf.keras.layers.Concatenate()([shorcut1d,shorcut1dy,shorcut1dz,x])
    # print(str(x.shape)+"sd4")
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Conv2D(2*n,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+123),activation="softmax")(x)
    x=tf.keras.layers.Concatenate()([shortcutp1,shortcutp1y,shortcutp1z,x])
    # print(str(x.shape)+"xd4")
    # deconv block 2
    x = tf.keras.layers.BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2DTranspose(2*n, (3, 3), strides=(2, 2), padding='same')(x)
    x=tf.keras.layers.Concatenate()([shortcut2c,shortcut2cy,shortcut2cz,x])
    # print(str(x.shape)+"xdc3")

    x = tf.keras.layers.BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2D(2*n, (3, 3), padding='same',kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(x)
    x=tf.keras.layers.Concatenate()([shortcut2c,shortcut2cy,shortcut2cz,x])
    # print(str(x.shape)+"xdc3.1")
    x = tf.keras.layers.BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2D(2*n, (3, 3), padding='same',kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+333))(x)
    x=tf.keras.layers.Concatenate()([shortcut1d,shortcut1dy,shortcut1dz,x])
    # print(str(x.shape)+"xdc3.2")
    # dense block 5
    x=tf.keras.layers.Conv2D(2,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(x)
    x=tf.keras.layers.Concatenate()([shorcut0d,shorcut0dy,shorcut0dz,x])
    x=tf.keras.layers.BatchNormalization()(x)

    x=tf.keras.layers.Conv2D(4*n,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(x)
    
    # print(str(x.shape)+"xd5")
    # final conv block
    x=tf.keras.layers.Conv2D(2*n,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(x)
    x=tf.keras.layers.Concatenate()([shortcut3c,shortcut3cy,shortcut3cz,x])
    x=tf.keras.layers.Conv2D(1,(1,1),(1,1),kernel_initializer=tf.keras.initializers.HeNormal(seed=seed+334),activation="relu")(x)
    x=tf.keras.layers.Concatenate()([shortcut1c,shortcut1cy,shortcut1cz,x])
    x=tf.keras.layers.BatchNormalization()(x)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    return tf.keras.Model(inputs,outputs)

model = cerber(1,5)
model.summary()
plot_model(model, to_file=r"C:/Users/TK/Desktop/spider_nn/fitterer/models/cerberus_noNeck_conv.png", show_shapes=True, show_layer_names=True)
opti=tf.keras.optimizers.AdamW(0.01,0.001)
print("kompilacja")
model.compile(optimizer=opti, loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=3.5,alpha=0.01), metrics=[tf.keras.metrics.BinaryIoU(),tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC()])

#training, callbacks

my_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=r'C:/Users/TK/Desktop/spider_nn/fitterer/models/cerberConvSigmoid.binaryfcross.g3.5.al0.5.1b.12n.Epoka-{epoch:02d}_loss-{val_loss:.4f}_IoU-{val_binary_io_u:.4f}_binAcc-{val_binary_accuracy:.4f}.h5',verbose=1,monitor="val_loss",save_weights_only=False,mode="min",save_best_only=True)
print("fitowanie")
model.fit(datagen.flow(train_peaks, train_masks,batch_size=1), batch_size=1, epochs=50, verbose=1,callbacks=my_callbacks,validation_data=(test_peaks, test_masks),validation_batch_size=1)