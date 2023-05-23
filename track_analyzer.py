# T. Kowalski project 
# made for Machine Learning, KISD/IFJ PAN, Krakow
# trained network output visualisation

import numpy as np
import tensorflow as tf
import os, glob
from PIL import Image
import matplotlib.pyplot as plt

# data preparation

def input_tensor_prepare(input):
    output=(np.asarray(np.stack((input,)*3,axis=-1),dtype=np.float32)/255.)
    output=tf.convert_to_tensor(output.astype(float),dtype=tf.float32)   
    output=tf.cast(output, dtype=tf.float32)

    return output
    
obraz=1
os.chdir(r"C:/Users/User/Desktop/Protony LiF 1/hresting_data") # raw data
# os.chdir(r"C:/Users/User/Desktop/Machine Learning/data_generator/datagen/train/images") # generated data

# data processing part...

extension="*.bmp"
for file in glob.glob(extension):
    input_image_im=Image.open(file)

    preprocessed_tensor=[]
    imm=np.asarray(input_image_im)
    
    preprocessed_tensor.append(imm)
    preprocessed_tensor=input_tensor_prepare(preprocessed_tensor[:][:][:])

# loading trained model...

    reconstructed_model_1 = tf.keras.models.load_model(r"C:/Users/User/Desktop/spider_nn/fitterer/models_new/convolver.tverski.1b.2n.Epoka-01_loss-0.1721_IoU-0.4994_binAcc-0.9987.h5")

# predict...

    weights1=reconstructed_model_1.predict(preprocessed_tensor)

# predict again...

    # bridge=[]
    # bridge.append(input_tensor_prepare(weights1[0,:,:,0]*255.))
    # bridge=tf.cast(bridge,dtype=np.float32)

    # weights1=reconstructed_model_1.predict(bridge)
    weights1=np.asanyarray(weights1,dtype=np.float32)
# normalise...

    # weigths1_max=np.max(weights1[0,:,:,0])
    # weigths1_min=np.min(weights1[0,:,:,0])
    # weigths1_mean=np.mean(weights1[0,:,:,0])
    norm=((255.0)*(weights1[0,:,:,0]-np.min(weights1[0,:,:,0]))/np.ptp(weights1[0,:,:,0]))
    # norm=(weights1[0,:,:,0]-weigths1_min)/(weigths1_max-weigths1_min)#-((weigths1_mean-weigths1_min)/(weigths1_max-weigths1_min))
    # mod1=(norm)

# threshold globally (not very good, some data are lost)

    masked=norm>=29

# plot the results

    fig, axs = plt.subplots(nrows=1, ncols=3,  figsize=(3, 5))
    axs[0].set_title('Rescaled probability of track')
    axs[0].imshow(norm,cmap='twilight',norm="log")

    axs[1].set_title('Raw data image - 8 bit BMP')
    axs[1].imshow(input_image_im,cmap='twilight',norm="log")

    axs[2].set_title('masks')
    axs[2].imshow(masked,cmap='binary')
    plt.show()
