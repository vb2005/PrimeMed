# -*- coding: utf-8 -*-
"""
Clean Prime Project

Created on Wed Nov  2 14:00:53 2022

@author: User
"""

#%% Check CUDA
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices()) # list of DeviceAttributes

#%% LIBS
import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import nibabel as nib
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose
from tensorflow.keras.optimizers import RMSprop, Adam
import skimage
from skimage.util import montage
from skimage.transform import resize
from scipy.ndimage import zoom
import cv2

#%% Test Reading NII
sample_filename="D:/PrimeMed/Prime/nii/37_40000965_Leshch.nii"
sample_img = nib.load(sample_filename)
sample_img = np.asanyarray(sample_img.dataobj)
print(sample_img.shape)
img_min = np.min(sample_img)
sample_img=(sample_img - img_min)*255 / (np.max(sample_img) - img_min)
print(img_min,np.max(sample_img))
cv2.imwrite("2.png",sample_img[sample_img.shape[0]//2])
a=montage(sample_img)
print(a.shape)

#%% UNET 3D. Kvantron Edition
def get_model():
  inputs = Input((64,64,64,1))
  conv1 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(inputs)
  conv1 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(conv1)
  
  #pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
  pool1 = Conv3D(8, (3, 3, 3),(2, 2, 2), activation='relu', padding='same')(conv1)
  

  conv2 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(pool1)
  conv2 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv2)
  
  #pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
  pool2 = Conv3D(16, (3, 3, 3),(2, 2, 2), activation='relu', padding='same')(conv2)

  conv3 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool2)
  conv3 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv3)
  #pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
  pool3 = Conv3D(32, (3, 3, 3),(2, 2, 2), activation='relu', padding='same')(conv3)

  conv4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool3)
  conv4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv4)
  
  pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

  conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
  conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
  
  conv5_1=Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)

  up6 = concatenate([conv5_1, conv4], axis=4)

  #print(conv4)
  
  conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
  conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

  up7 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
  conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up7)
  conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv7)

  up8 = concatenate([Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
  conv8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up8)
  conv8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv8)

  up9 = concatenate([Conv3DTranspose(8, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
  conv9 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(up9)
  conv9 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(conv9)

  #conv10 = Conv3D(1, (1, 1, 1), activation='softsign')(conv9)
  conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)
  #conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv10)

  model = Model(inputs=[inputs], outputs=[conv10])
  #model.summary()

  model.compile(optimizer=Adam(), 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
#(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199)
  return model
model=get_model()

#%% Read Nii Files
X_train=[]
Y_train=[]

data_root="D:\\PrimeMed\\Prime\\nii\\src"
images = os.listdir(data_root)
for image in images:
  im=nib.load(data_root+"\\"+image)
  im = np.array(im.dataobj)
  im = im[0:200,200:400,200:400]
  im = zoom(im, (0.32, 0.32, 0.32))
  
  im_min = np.min(im)
  im_max = np.max(im)

  im=((im - im_min)*255.0) / (im_max - im_min)
  im=im.reshape(64,64,64,1)
  print(im.shape)
  X_train.append(im)


data_root="D:\\PrimeMed\\Prime\\nii\\dst"
images = os.listdir(data_root)
for image in images:
  im=nib.load(data_root+"\\"+image)
  im = np.array(im.dataobj)
  im = im[0:200,200:400,200:400]
  im = zoom(im, (0.32, 0.32, 0.32))
  #print(np.unique(im.ravel()))  
  im_min = np.min(im)
  im_max = np.max(im)
  im=((im - im_min)*255.0) / (im_max - im_min)
  im[im > 0]=1
  im=im.reshape(64,64,64,1)
  print(im.shape)
  Y_train.append(im)

X_train=np.array(X_train)
Y_train=np.array(Y_train)
print(X_train.shape)
print(Y_train.shape)

#%% Start Training
def cb():
  a55 = model.predict(im)*255
  aa=a55.reshape(64,64,64)
  cv2.imwrite("result.png", montage(aa))

for i in range(1000):
    history = model.fit(X_train, Y_train, batch_size=2, epochs=1)
    cb()
    
#%% Test
im2=nib.load('D:/PrimeMed/Prime/nii/src/990056070.nii')
a55 = model.predict(im)*255
aa=a55.reshape(64,64,64)
aa = zoom(aa, (1.0/0.32, 1.0/0.32, 1.0/0.32))
bb = np.zeros((512,512,512))
print("FIN")

for a in range(1,199):
    for b in range(1,199):
        for c in range(1,199):
             #[a,b+200,c+200] = aa[a,b,c]
             val = []
             for x in range (-1,2):
                 for y in range (-1,2):
                     for z in range(-1,2):
                         val.append(aa[a+x, b+y, c+z])
                        
             val.sort()
             bb[c,b+200,a+200] = val[13]
                    

print("FIN")

bb[bb>128] = 255
bb[bb<=128] = 0

cv2.imwrite("result.png", montage(bb))
final_img = nib.Nifti1Image(bb, im2.affine)
nib.save(final_img, 'D:/PrimeMed/Prime/4.nii')