# T. Kowalski - author... will never sign with a first name. This code is so crude.
# part of Machine Learning project - training data generator
# 
# this code makes machine life harder
# "tracks" - 2d gaussians - are so hard to detect. For human merely noticeable
# such method allowed NN to train on trash data and work with 'true' raw data with good results

# for 16 GB of RAM (laptop, and so on) do not make more than 70 1608x1608 images

import numpy as np
import random as r
from datetime import datetime
from PIL import Image
import multiprocessing
from multiprocessing import Process

r.seed(datetime.now().timestamp())

# uniform - just from 0 to 1-1/rand_max, with rand_max**(-1) resolution

def uniform_0to1_gen():
    rand_max = 1e30

    uniform = r.randint(0,rand_max)/(1+rand_max)
    return uniform

# simple box-mueller transform in 1 dimension

def gaussian_gen(uniform1,uniform2,st_dev,mean):
    gaussian = st_dev*np.sqrt(-2*np.log(uniform1))*np.cos(2*np.pi*uniform2)+mean
    return gaussian

# linear transformation of uniform_0to1_gen, giving uniform distr. from x to y

def uniform_XtoY_gen(x,y):
    rand_max = 1e30

    uniform_xy = (y-x)*(r.randint(0,rand_max)/(1+rand_max))+x
    return uniform_xy

# below: background gen., offset - bckgr. level, next - adding gaussian noise with mean = 0

def background_gaussian(std_dev,offset):
    random1=uniform_0to1_gen()
    random2=uniform_0to1_gen()
    bckg=gaussian_gen(random1,random2,std_dev,0)+offset
    if bckg<0.:
        bckg=0.
    if bckg>255.:
        bckg=255.
    return bckg

# gaussian 2d function with[out] built-in rotation

def gaussian2d_rotation(x,y,x0,y0,std_dev_x,std_dev_y,amplitude,angle_rotation):
    rot_x_block=(np.cos(angle_rotation)**2/(2*std_dev_x**2)+np.sin(angle_rotation)**2/(2*std_dev_y**2))
    rot_y_block=(np.sin(angle_rotation)**2/(2*std_dev_x**2)+np.cos(angle_rotation)**2/(2*std_dev_y**2))
    rot_xy_block=(np.sin(2*angle_rotation)/(2*std_dev_x**2)-np.sin(2*angle_rotation)/(2*std_dev_y**2))
    x_block=-((x-x0)**2)
    xy_block=((x-x0))*((y-y0))
    y_block=-((y-y0)**2)
    signal=amplitude*np.exp(rot_x_block*x_block+rot_xy_block*xy_block+rot_y_block*y_block)
    if signal>255:
        signal=255
    if signal<0:
        signal=0
    return signal
# +)()()()()()()((((((((((((((((((((((((((((((((((((((((((((((((((((((((((()))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

# ------------------------------------------------------------------------------------------------
# for the need of short_dense_cresunet network: mask and image generator
# ------------------------------------------------------------------------------------------------
def cResUnet_big_image_gen1(no,dir_image,dir_mask):

    for imagge in range(0,no,1):
        print(no)
        picture = np.ndarray(shape=(1608,1608),dtype="float32")
        picture.fill(0)
        mask = np.ndarray(shape=(1608,1608),dtype="float32")
        mask.fill(0)

        no_gaussians=r.randint(0,20)
        gaussians=np.ndarray((no_gaussians,6))

        no_inhom=r.randint(6,9)
        inhom=np.ndarray((no_inhom,6))

        offset=uniform_XtoY_gen(10,190/2)
        std_dev=uniform_XtoY_gen(5,offset*3)

        for index in range(0,len(gaussians),1):
            gaussians[index,0] = uniform_XtoY_gen(200,1400)
            gaussians[index,1] = uniform_XtoY_gen(200,1400)
            gaussians[index,2] = uniform_XtoY_gen(2,6)
            gaussians[index,3] = uniform_XtoY_gen(2,6)
            gaussians[index,4] = uniform_XtoY_gen(0.35*offset,offset*0.74)
            gaussians[index,5] = uniform_XtoY_gen(0,np.pi)

        for element in range(0,len(inhom),1):
            inhom[element,0] = uniform_XtoY_gen(200,1400)
            inhom[element,1] = uniform_XtoY_gen(200,1400)
            inhom[element,2] = uniform_XtoY_gen(20,450)
            inhom[element,3] = uniform_XtoY_gen(20,450)
            inhom[element,4] = uniform_XtoY_gen(20,95)
            inhom[element,5] = uniform_XtoY_gen(0,np.pi)

        bckg_sdx=uniform_XtoY_gen(700,1000)
        bckg_sdy=bckg_sdx
        bckg_x0=804
        bckg_y0=804
        bckg_amp=uniform_XtoY_gen(120,190)


        for i in range(0,len(picture),1):
            print(i)
                    
            for j in range(0,len(picture[0]),1):
                for indexd in range(0,len(gaussians),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,gaussians[indexd,0],gaussians[indexd,1],gaussians[indexd,2],gaussians[indexd,3],gaussians[indexd,4],gaussians[indexd,5])
                    if ((((i-gaussians[indexd][0]))*np.cos(gaussians[indexd][5])/(gaussians[indexd][2])) + ((j-gaussians[indexd][1])*np.sin(gaussians[indexd][5])/(gaussians[indexd][2])))**2+((((i-gaussians[indexd][0]))*np.sin(gaussians[indexd][5])/(gaussians[indexd][3])) - ((j-gaussians[indexd][1])*np.cos(gaussians[indexd][5])/(gaussians[indexd][3])))**2<=1.0:
                        mask[i,j]+=255
                    else:
                        mask[i,j]+=0
                for elementd in range(0,len(inhom),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,inhom[elementd,0],inhom[elementd,1],inhom[elementd,2],inhom[elementd,3],inhom[elementd,4],inhom[elementd,5])
                if mask[i,j]>=255:
                    mask[i,j]=255
                if mask[i,j]<=0:
                    mask[i,j]=0
                picture[i,j] += gaussian2d_rotation(i,j,bckg_x0,bckg_y0,bckg_sdx,bckg_sdy,bckg_amp,0)
                picture[i,j] += background_gaussian(std_dev,offset)
        picture=(255*(picture-np.min(picture))/np.ptp(picture)).astype(int)
        mask=(255*(mask-np.min(mask))/np.ptp(mask)).astype(int)

        check=Image.fromarray(mask.astype('uint8'),mode='L')
        check.save(dir_mask+str(imagge)+"szisza1.bmp",format="bmp")
        image=Image.fromarray(picture.astype('uint8'),mode='L')       
        image.save(dir_image+str(imagge)+"szisza1.bmp",format="bmp")
        print(imagge)
def cResUnet_big_image_gen2(no,dir_image,dir_mask):

    for imagge in range(0,no,1):
        print(no)
        picture = np.ndarray(shape=(1608,1608),dtype="float32")
        picture.fill(0)
        mask = np.ndarray(shape=(1608,1608),dtype="float32")
        mask.fill(0)

        no_gaussians=r.randint(0,20)
        gaussians=np.ndarray((no_gaussians,6))

        no_inhom=r.randint(6,9)
        inhom=np.ndarray((no_inhom,6))

        offset=uniform_XtoY_gen(10,190/2)
        std_dev=uniform_XtoY_gen(5,offset*3)

        for index in range(0,len(gaussians),1):
            gaussians[index,0] = uniform_XtoY_gen(200,1400)
            gaussians[index,1] = uniform_XtoY_gen(200,1400)
            gaussians[index,2] = uniform_XtoY_gen(2,6)
            gaussians[index,3] = uniform_XtoY_gen(2,6)
            gaussians[index,4] = uniform_XtoY_gen(0.35*offset,offset*0.74)
            gaussians[index,5] = uniform_XtoY_gen(0,np.pi)

        for element in range(0,len(inhom),1):
            inhom[element,0] = uniform_XtoY_gen(200,1400)
            inhom[element,1] = uniform_XtoY_gen(200,1400)
            inhom[element,2] = uniform_XtoY_gen(20,450)
            inhom[element,3] = uniform_XtoY_gen(20,450)
            inhom[element,4] = uniform_XtoY_gen(20,95)
            inhom[element,5] = uniform_XtoY_gen(0,np.pi)

        bckg_sdx=uniform_XtoY_gen(700,1000)
        bckg_sdy=bckg_sdx
        bckg_x0=804
        bckg_y0=804
        bckg_amp=uniform_XtoY_gen(120,190)


        for i in range(0,len(picture),1):
            print(i)
                    
            for j in range(0,len(picture[0]),1):
                for indexd in range(0,len(gaussians),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,gaussians[indexd,0],gaussians[indexd,1],gaussians[indexd,2],gaussians[indexd,3],gaussians[indexd,4],gaussians[indexd,5])
                    if ((((i-gaussians[indexd][0]))*np.cos(gaussians[indexd][5])/(gaussians[indexd][2])) + ((j-gaussians[indexd][1])*np.sin(gaussians[indexd][5])/(gaussians[indexd][2])))**2+((((i-gaussians[indexd][0]))*np.sin(gaussians[indexd][5])/(gaussians[indexd][3])) - ((j-gaussians[indexd][1])*np.cos(gaussians[indexd][5])/(gaussians[indexd][3])))**2<=1.0:
                        mask[i,j]+=255
                    else:
                        mask[i,j]+=0
                for elementd in range(0,len(inhom),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,inhom[elementd,0],inhom[elementd,1],inhom[elementd,2],inhom[elementd,3],inhom[elementd,4],inhom[elementd,5])
                if mask[i,j]>=255:
                    mask[i,j]=255
                if mask[i,j]<=0:
                    mask[i,j]=0
                picture[i,j] += gaussian2d_rotation(i,j,bckg_x0,bckg_y0,bckg_sdx,bckg_sdy,bckg_amp,0)
                picture[i,j] += background_gaussian(std_dev,offset)
        picture=(255*(picture-np.min(picture))/np.ptp(picture)).astype(int)
        mask=(255*(mask-np.min(mask))/np.ptp(mask)).astype(int)

        check=Image.fromarray(mask.astype('uint8'),mode='L')
        check.save(dir_mask+str(imagge)+"szisza2.bmp",format="bmp")
        image=Image.fromarray(picture.astype('uint8'),mode='L')       
        image.save(dir_image+str(imagge)+"szisza2.bmp",format="bmp")
        print(imagge)
def cResUnet_big_image_gen3(no,dir_image,dir_mask):

    for imagge in range(0,no,1):
        print(no)
        picture = np.ndarray(shape=(1608,1608),dtype="float32")
        picture.fill(0)
        mask = np.ndarray(shape=(1608,1608),dtype="float32")
        mask.fill(0)

        no_gaussians=r.randint(0,20)
        gaussians=np.ndarray((no_gaussians,6))

        no_inhom=r.randint(6,9)
        inhom=np.ndarray((no_inhom,6))

        offset=uniform_XtoY_gen(10,190/2)
        std_dev=uniform_XtoY_gen(5,offset*3)

        for index in range(0,len(gaussians),1):
            gaussians[index,0] = uniform_XtoY_gen(200,1400)
            gaussians[index,1] = uniform_XtoY_gen(200,1400)
            gaussians[index,2] = uniform_XtoY_gen(2,6)
            gaussians[index,3] = uniform_XtoY_gen(2,6)
            gaussians[index,4] = uniform_XtoY_gen(0.35*offset,offset*0.74)
            gaussians[index,5] = uniform_XtoY_gen(0,np.pi)

        for element in range(0,len(inhom),1):
            inhom[element,0] = uniform_XtoY_gen(200,1400)
            inhom[element,1] = uniform_XtoY_gen(200,1400)
            inhom[element,2] = uniform_XtoY_gen(20,450)
            inhom[element,3] = uniform_XtoY_gen(20,450)
            inhom[element,4] = uniform_XtoY_gen(20,95)
            inhom[element,5] = uniform_XtoY_gen(0,np.pi)

        bckg_sdx=uniform_XtoY_gen(700,1000)
        bckg_sdy=bckg_sdx
        bckg_x0=804
        bckg_y0=804
        bckg_amp=uniform_XtoY_gen(120,190)


        for i in range(0,len(picture),1):
            print(i)
                    
            for j in range(0,len(picture[0]),1):
                for indexd in range(0,len(gaussians),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,gaussians[indexd,0],gaussians[indexd,1],gaussians[indexd,2],gaussians[indexd,3],gaussians[indexd,4],gaussians[indexd,5])
                    if ((((i-gaussians[indexd][0]))*np.cos(gaussians[indexd][5])/(gaussians[indexd][2])) + ((j-gaussians[indexd][1])*np.sin(gaussians[indexd][5])/(gaussians[indexd][2])))**2+((((i-gaussians[indexd][0]))*np.sin(gaussians[indexd][5])/(gaussians[indexd][3])) - ((j-gaussians[indexd][1])*np.cos(gaussians[indexd][5])/(gaussians[indexd][3])))**2<=1.0:
                        mask[i,j]+=255
                    else:
                        mask[i,j]+=0
                for elementd in range(0,len(inhom),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,inhom[elementd,0],inhom[elementd,1],inhom[elementd,2],inhom[elementd,3],inhom[elementd,4],inhom[elementd,5])
                if mask[i,j]>=255:
                    mask[i,j]=255
                if mask[i,j]<=0:
                    mask[i,j]=0
                picture[i,j] += gaussian2d_rotation(i,j,bckg_x0,bckg_y0,bckg_sdx,bckg_sdy,bckg_amp,0)
                picture[i,j] += background_gaussian(std_dev,offset)
        picture=(255*(picture-np.min(picture))/np.ptp(picture)).astype(int)
        mask=(255*(mask-np.min(mask))/np.ptp(mask)).astype(int)

        check=Image.fromarray(mask.astype('uint8'),mode='L')
        check.save(dir_mask+str(imagge)+"szisza3.bmp",format="bmp")
        image=Image.fromarray(picture.astype('uint8'),mode='L')       
        image.save(dir_image+str(imagge)+"szisza3.bmp",format="bmp")
        print(imagge)
def cResUnet_big_image_gen4(no,dir_image,dir_mask):

    for imagge in range(0,no,1):
        print(no)
        picture = np.ndarray(shape=(1608,1608),dtype="float32")
        picture.fill(0)
        mask = np.ndarray(shape=(1608,1608),dtype="float32")
        mask.fill(0)

        no_gaussians=r.randint(0,20)
        gaussians=np.ndarray((no_gaussians,6))

        no_inhom=r.randint(6,9)
        inhom=np.ndarray((no_inhom,6))

        offset=uniform_XtoY_gen(10,190/2)
        std_dev=uniform_XtoY_gen(5,offset*3)

        for index in range(0,len(gaussians),1):
            gaussians[index,0] = uniform_XtoY_gen(200,1400)
            gaussians[index,1] = uniform_XtoY_gen(200,1400)
            gaussians[index,2] = uniform_XtoY_gen(2,6)
            gaussians[index,3] = uniform_XtoY_gen(2,6)
            gaussians[index,4] = uniform_XtoY_gen(0.35*offset,offset*0.74)
            gaussians[index,5] = uniform_XtoY_gen(0,np.pi)

        for element in range(0,len(inhom),1):
            inhom[element,0] = uniform_XtoY_gen(200,1400)
            inhom[element,1] = uniform_XtoY_gen(200,1400)
            inhom[element,2] = uniform_XtoY_gen(20,450)
            inhom[element,3] = uniform_XtoY_gen(20,450)
            inhom[element,4] = uniform_XtoY_gen(20,95)
            inhom[element,5] = uniform_XtoY_gen(0,np.pi)

        bckg_sdx=uniform_XtoY_gen(700,1000)
        bckg_sdy=bckg_sdx
        bckg_x0=804
        bckg_y0=804
        bckg_amp=uniform_XtoY_gen(120,190)


        for i in range(0,len(picture),1):
            print(i)
                    
            for j in range(0,len(picture[0]),1):
                for indexd in range(0,len(gaussians),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,gaussians[indexd,0],gaussians[indexd,1],gaussians[indexd,2],gaussians[indexd,3],gaussians[indexd,4],gaussians[indexd,5])
                    if ((((i-gaussians[indexd][0]))*np.cos(gaussians[indexd][5])/(gaussians[indexd][2])) + ((j-gaussians[indexd][1])*np.sin(gaussians[indexd][5])/(gaussians[indexd][2])))**2+((((i-gaussians[indexd][0]))*np.sin(gaussians[indexd][5])/(gaussians[indexd][3])) - ((j-gaussians[indexd][1])*np.cos(gaussians[indexd][5])/(gaussians[indexd][3])))**2<=1.0:
                        mask[i,j]+=255
                    else:
                        mask[i,j]+=0
                for elementd in range(0,len(inhom),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,inhom[elementd,0],inhom[elementd,1],inhom[elementd,2],inhom[elementd,3],inhom[elementd,4],inhom[elementd,5])
                if mask[i,j]>=255:
                    mask[i,j]=255
                if mask[i,j]<=0:
                    mask[i,j]=0
                picture[i,j] += gaussian2d_rotation(i,j,bckg_x0,bckg_y0,bckg_sdx,bckg_sdy,bckg_amp,0)
                picture[i,j] += background_gaussian(std_dev,offset)
        picture=(255*(picture-np.min(picture))/np.ptp(picture)).astype(int)
        mask=(255*(mask-np.min(mask))/np.ptp(mask)).astype(int)

        check=Image.fromarray(mask.astype('uint8'),mode='L')
        check.save(dir_mask+str(imagge)+"szisza4.bmp",format="bmp")
        image=Image.fromarray(picture.astype('uint8'),mode='L')       
        image.save(dir_image+str(imagge)+"szisza4.bmp",format="bmp")
        print(imagge)
def cResUnet_big_image_gen5(no,dir_image,dir_mask):

    for imagge in range(0,no,1):
        print(no)
        picture = np.ndarray(shape=(1608,1608),dtype="float32")
        picture.fill(0)
        mask = np.ndarray(shape=(1608,1608),dtype="float32")
        mask.fill(0)

        no_gaussians=r.randint(0,20)
        gaussians=np.ndarray((no_gaussians,6))

        no_inhom=r.randint(6,9)
        inhom=np.ndarray((no_inhom,6))

        offset=uniform_XtoY_gen(10,190/2)
        std_dev=uniform_XtoY_gen(5,offset*3)

        for index in range(0,len(gaussians),1):
            gaussians[index,0] = uniform_XtoY_gen(200,1400)
            gaussians[index,1] = uniform_XtoY_gen(200,1400)
            gaussians[index,2] = uniform_XtoY_gen(2,6)
            gaussians[index,3] = uniform_XtoY_gen(2,6)
            gaussians[index,4] = uniform_XtoY_gen(0.35*offset,offset*0.74)
            gaussians[index,5] = uniform_XtoY_gen(0,np.pi)

        for element in range(0,len(inhom),1):
            inhom[element,0] = uniform_XtoY_gen(200,1400)
            inhom[element,1] = uniform_XtoY_gen(200,1400)
            inhom[element,2] = uniform_XtoY_gen(20,450)
            inhom[element,3] = uniform_XtoY_gen(20,450)
            inhom[element,4] = uniform_XtoY_gen(20,95)
            inhom[element,5] = uniform_XtoY_gen(0,np.pi)

        bckg_sdx=uniform_XtoY_gen(700,1000)
        bckg_sdy=bckg_sdx
        bckg_x0=804
        bckg_y0=804
        bckg_amp=uniform_XtoY_gen(120,190)


        for i in range(0,len(picture),1):
            print(i)
                    
            for j in range(0,len(picture[0]),1):
                for indexd in range(0,len(gaussians),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,gaussians[indexd,0],gaussians[indexd,1],gaussians[indexd,2],gaussians[indexd,3],gaussians[indexd,4],gaussians[indexd,5])
                    if ((((i-gaussians[indexd][0]))*np.cos(gaussians[indexd][5])/(gaussians[indexd][2])) + ((j-gaussians[indexd][1])*np.sin(gaussians[indexd][5])/(gaussians[indexd][2])))**2+((((i-gaussians[indexd][0]))*np.sin(gaussians[indexd][5])/(gaussians[indexd][3])) - ((j-gaussians[indexd][1])*np.cos(gaussians[indexd][5])/(gaussians[indexd][3])))**2<=1.0:
                        mask[i,j]+=255
                    else:
                        mask[i,j]+=0
                for elementd in range(0,len(inhom),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,inhom[elementd,0],inhom[elementd,1],inhom[elementd,2],inhom[elementd,3],inhom[elementd,4],inhom[elementd,5])
                if mask[i,j]>=255:
                    mask[i,j]=255
                if mask[i,j]<=0:
                    mask[i,j]=0
                picture[i,j] += gaussian2d_rotation(i,j,bckg_x0,bckg_y0,bckg_sdx,bckg_sdy,bckg_amp,0)
                picture[i,j] += background_gaussian(std_dev,offset)
        picture=(255*(picture-np.min(picture))/np.ptp(picture)).astype(int)
        mask=(255*(mask-np.min(mask))/np.ptp(mask)).astype(int)

        check=Image.fromarray(mask.astype('uint8'),mode='L')
        check.save(dir_mask+str(imagge)+"szisza5.bmp",format="bmp")
        image=Image.fromarray(picture.astype('uint8'),mode='L')       
        image.save(dir_image+str(imagge)+"szisza5.bmp",format="bmp")
        print(imagge)
def cResUnet_big_image_gen6(no,dir_image,dir_mask):

    for imagge in range(0,no,1):
        print(no)
        picture = np.ndarray(shape=(1608,1608),dtype="float32")
        picture.fill(0)
        mask = np.ndarray(shape=(1608,1608),dtype="float32")
        mask.fill(0)

        no_gaussians=r.randint(0,20)
        gaussians=np.ndarray((no_gaussians,6))

        no_inhom=r.randint(6,9)
        inhom=np.ndarray((no_inhom,6))

        offset=uniform_XtoY_gen(10,190/2)
        std_dev=uniform_XtoY_gen(5,offset*3)

        for index in range(0,len(gaussians),1):
            gaussians[index,0] = uniform_XtoY_gen(200,1400)
            gaussians[index,1] = uniform_XtoY_gen(200,1400)
            gaussians[index,2] = uniform_XtoY_gen(2,6)
            gaussians[index,3] = uniform_XtoY_gen(2,6)
            gaussians[index,4] = uniform_XtoY_gen(0.35*offset,offset*0.74)
            gaussians[index,5] = uniform_XtoY_gen(0,np.pi)

        for element in range(0,len(inhom),1):
            inhom[element,0] = uniform_XtoY_gen(200,1400)
            inhom[element,1] = uniform_XtoY_gen(200,1400)
            inhom[element,2] = uniform_XtoY_gen(20,450)
            inhom[element,3] = uniform_XtoY_gen(20,450)
            inhom[element,4] = uniform_XtoY_gen(20,95)
            inhom[element,5] = uniform_XtoY_gen(0,np.pi)

        bckg_sdx=uniform_XtoY_gen(700,1000)
        bckg_sdy=bckg_sdx
        bckg_x0=804
        bckg_y0=804
        bckg_amp=uniform_XtoY_gen(120,190)


        for i in range(0,len(picture),1):
            print(i)
                    
            for j in range(0,len(picture[0]),1):
                for indexd in range(0,len(gaussians),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,gaussians[indexd,0],gaussians[indexd,1],gaussians[indexd,2],gaussians[indexd,3],gaussians[indexd,4],gaussians[indexd,5])
                    if ((((i-gaussians[indexd][0]))*np.cos(gaussians[indexd][5])/(gaussians[indexd][2])) + ((j-gaussians[indexd][1])*np.sin(gaussians[indexd][5])/(gaussians[indexd][2])))**2+((((i-gaussians[indexd][0]))*np.sin(gaussians[indexd][5])/(gaussians[indexd][3])) - ((j-gaussians[indexd][1])*np.cos(gaussians[indexd][5])/(gaussians[indexd][3])))**2<=1.0:
                        mask[i,j]+=255
                    else:
                        mask[i,j]+=0
                for elementd in range(0,len(inhom),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,inhom[elementd,0],inhom[elementd,1],inhom[elementd,2],inhom[elementd,3],inhom[elementd,4],inhom[elementd,5])
                if mask[i,j]>=255:
                    mask[i,j]=255
                if mask[i,j]<=0:
                    mask[i,j]=0
                picture[i,j] += gaussian2d_rotation(i,j,bckg_x0,bckg_y0,bckg_sdx,bckg_sdy,bckg_amp,0)
                picture[i,j] += background_gaussian(std_dev,offset)
        picture=(255*(picture-np.min(picture))/np.ptp(picture)).astype(int)
        mask=(255*(mask-np.min(mask))/np.ptp(mask)).astype(int)

        check=Image.fromarray(mask.astype('uint8'),mode='L')
        check.save(dir_mask+str(imagge)+"szisza6.bmp",format="bmp")
        image=Image.fromarray(picture.astype('uint8'),mode='L')       
        image.save(dir_image+str(imagge)+"szisza6.bmp",format="bmp")
        print(imagge)
def cResUnet_big_image_gen7(no,dir_image,dir_mask):

    for imagge in range(0,no,1):
        print(no)
        picture = np.ndarray(shape=(1608,1608),dtype="float32")
        picture.fill(0)
        mask = np.ndarray(shape=(1608,1608),dtype="float32")
        mask.fill(0)

        no_gaussians=r.randint(0,20)
        gaussians=np.ndarray((no_gaussians,6))

        no_inhom=r.randint(6,9)
        inhom=np.ndarray((no_inhom,6))

        offset=uniform_XtoY_gen(10,190/2)
        std_dev=uniform_XtoY_gen(5,offset*3)

        for index in range(0,len(gaussians),1):
            gaussians[index,0] = uniform_XtoY_gen(200,1400)
            gaussians[index,1] = uniform_XtoY_gen(200,1400)
            gaussians[index,2] = uniform_XtoY_gen(2,6)
            gaussians[index,3] = uniform_XtoY_gen(2,6)
            gaussians[index,4] = uniform_XtoY_gen(0.35*offset,offset*0.74)
            gaussians[index,5] = uniform_XtoY_gen(0,np.pi)

        for element in range(0,len(inhom),1):
            inhom[element,0] = uniform_XtoY_gen(200,1400)
            inhom[element,1] = uniform_XtoY_gen(200,1400)
            inhom[element,2] = uniform_XtoY_gen(20,450)
            inhom[element,3] = uniform_XtoY_gen(20,450)
            inhom[element,4] = uniform_XtoY_gen(20,95)
            inhom[element,5] = uniform_XtoY_gen(0,np.pi)

        bckg_sdx=uniform_XtoY_gen(700,1000)
        bckg_sdy=bckg_sdx
        bckg_x0=804
        bckg_y0=804
        bckg_amp=uniform_XtoY_gen(120,190)


        for i in range(0,len(picture),1):
            print(i)
                    
            for j in range(0,len(picture[0]),1):
                for indexd in range(0,len(gaussians),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,gaussians[indexd,0],gaussians[indexd,1],gaussians[indexd,2],gaussians[indexd,3],gaussians[indexd,4],gaussians[indexd,5])
                    if ((((i-gaussians[indexd][0]))*np.cos(gaussians[indexd][5])/(gaussians[indexd][2])) + ((j-gaussians[indexd][1])*np.sin(gaussians[indexd][5])/(gaussians[indexd][2])))**2+((((i-gaussians[indexd][0]))*np.sin(gaussians[indexd][5])/(gaussians[indexd][3])) - ((j-gaussians[indexd][1])*np.cos(gaussians[indexd][5])/(gaussians[indexd][3])))**2<=1.0:
                        mask[i,j]+=255
                    else:
                        mask[i,j]+=0
                for elementd in range(0,len(inhom),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,inhom[elementd,0],inhom[elementd,1],inhom[elementd,2],inhom[elementd,3],inhom[elementd,4],inhom[elementd,5])
                if mask[i,j]>=255:
                    mask[i,j]=255
                if mask[i,j]<=0:
                    mask[i,j]=0
                picture[i,j] += gaussian2d_rotation(i,j,bckg_x0,bckg_y0,bckg_sdx,bckg_sdy,bckg_amp,0)
                picture[i,j] += background_gaussian(std_dev,offset)
        picture=(255*(picture-np.min(picture))/np.ptp(picture)).astype(int)
        mask=(255*(mask-np.min(mask))/np.ptp(mask)).astype(int)

        check=Image.fromarray(mask.astype('uint8'),mode='L')
        check.save(dir_mask+str(imagge)+"szisza7.bmp",format="bmp")
        image=Image.fromarray(picture.astype('uint8'),mode='L')       
        image.save(dir_image+str(imagge)+"szisza7.bmp",format="bmp")
        print(imagge)
def cResUnet_big_image_gen8(no,dir_image,dir_mask):

    for imagge in range(0,no,1):
        print(no)
        picture = np.ndarray(shape=(1608,1608),dtype="float32")
        picture.fill(0)
        mask = np.ndarray(shape=(1608,1608),dtype="float32")
        mask.fill(0)

        no_gaussians=r.randint(0,20)
        gaussians=np.ndarray((no_gaussians,6))

        no_inhom=r.randint(6,9)
        inhom=np.ndarray((no_inhom,6))

        offset=uniform_XtoY_gen(10,190/2)
        std_dev=uniform_XtoY_gen(5,offset*3)

        for index in range(0,len(gaussians),1):
            gaussians[index,0] = uniform_XtoY_gen(200,1400)
            gaussians[index,1] = uniform_XtoY_gen(200,1400)
            gaussians[index,2] = uniform_XtoY_gen(2,6)
            gaussians[index,3] = uniform_XtoY_gen(2,6)
            gaussians[index,4] = uniform_XtoY_gen(0.35*offset,offset*0.74)
            gaussians[index,5] = uniform_XtoY_gen(0,np.pi)

        for element in range(0,len(inhom),1):
            inhom[element,0] = uniform_XtoY_gen(200,1400)
            inhom[element,1] = uniform_XtoY_gen(200,1400)
            inhom[element,2] = uniform_XtoY_gen(20,450)
            inhom[element,3] = uniform_XtoY_gen(20,450)
            inhom[element,4] = uniform_XtoY_gen(20,95)
            inhom[element,5] = uniform_XtoY_gen(0,np.pi)

        bckg_sdx=uniform_XtoY_gen(700,1000)
        bckg_sdy=bckg_sdx
        bckg_x0=804
        bckg_y0=804
        bckg_amp=uniform_XtoY_gen(120,190)


        for i in range(0,len(picture),1):
            print(i)
                    
            for j in range(0,len(picture[0]),1):
                for indexd in range(0,len(gaussians),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,gaussians[indexd,0],gaussians[indexd,1],gaussians[indexd,2],gaussians[indexd,3],gaussians[indexd,4],gaussians[indexd,5])
                    if ((((i-gaussians[indexd][0]))*np.cos(gaussians[indexd][5])/(gaussians[indexd][2])) + ((j-gaussians[indexd][1])*np.sin(gaussians[indexd][5])/(gaussians[indexd][2])))**2+((((i-gaussians[indexd][0]))*np.sin(gaussians[indexd][5])/(gaussians[indexd][3])) - ((j-gaussians[indexd][1])*np.cos(gaussians[indexd][5])/(gaussians[indexd][3])))**2<=1.0:
                        mask[i,j]+=255
                    else:
                        mask[i,j]+=0
                for elementd in range(0,len(inhom),1):
                    picture[i,j]+=gaussian2d_rotation(i,j,inhom[elementd,0],inhom[elementd,1],inhom[elementd,2],inhom[elementd,3],inhom[elementd,4],inhom[elementd,5])
                if mask[i,j]>=255:
                    mask[i,j]=255
                if mask[i,j]<=0:
                    mask[i,j]=0
                picture[i,j] += gaussian2d_rotation(i,j,bckg_x0,bckg_y0,bckg_sdx,bckg_sdy,bckg_amp,0)
                picture[i,j] += background_gaussian(std_dev,offset)
        picture=(255*(picture-np.min(picture))/np.ptp(picture)).astype(int)
        mask=(255*(mask-np.min(mask))/np.ptp(mask)).astype(int)

        check=Image.fromarray(mask.astype('uint8'),mode='L')
        check.save(dir_mask+str(imagge)+"szisza8.bmp",format="bmp")
        image=Image.fromarray(picture.astype('uint8'),mode='L')       
        image.save(dir_image+str(imagge)+"szisza8.bmp",format="bmp")
        print(imagge)

# training/validation directories

# dir_image=f"C:/Users/User/Desktop/Machine Learning/data_generator/datagen/train/images/"
# dir_mask=f"C:/Users/User/Desktop/Machine Learning/data_generator/datagen/train/masks/"

# dir_image_1=f"C:/Users/User/Desktop/Machine Learning/data_generator/datagen/test/images/"
# dir_mask_1=f"C:/Users/User/Desktop/Machine Learning/data_generator/datagen/test/masks/"

dir_image_new=f"C:/Users/User/Desktop/spider_nn/data_generator/datanew/test/images/"
dir_mask_new=f"C:/Users/User/Desktop/spider_nn/data_generator/datanew/test/masks/"

dir_image_new1=f"C:/Users/User/Desktop/spider_nn/data_generator/datanew/train/images/"
dir_mask_new1=f"C:/Users/User/Desktop/spider_nn/data_generator/datanew/train/masks/"
# below number of images to produce... let it be high enough

no = 100

# below initialisation of multiprocessing... make it all several times faster

def grinder():

    images = []

    images.append(Process(target= cResUnet_big_image_gen1,args=(no,dir_image_new,dir_mask_new),name="1"))
    images.append(Process(target= cResUnet_big_image_gen2,args=(no,dir_image_new,dir_mask_new),name="2"))
    images.append(Process(target= cResUnet_big_image_gen3,args=(no,dir_image_new,dir_mask_new),name="3"))
    # images.append(Process(target= cResUnet_big_image_gen4,args=(no,dir_image_new,dir_mask_new),name="4"))
    images.append(Process(target= cResUnet_big_image_gen5,args=(no,dir_image_new1,dir_mask_new1),name="5"))
    images.append(Process(target= cResUnet_big_image_gen6,args=(no,dir_image_new1,dir_mask_new1),name="6"))
    images.append(Process(target= cResUnet_big_image_gen7,args=(no,dir_image_new1,dir_mask_new1),name="7"))
    # images.append(Process(target= cResUnet_big_image_gen8,args=(no,dir_image_new1,dir_mask_new1),name="8"))


    multiprocessing.freeze_support()

    for el in images:
        el.start()
    
    for el in images:
        el.join()

# now make pictures

if __name__ == "__main__":
    grinder()