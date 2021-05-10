#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.compat.v1.keras.applications import MobileNet
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras.optimizers import Adam, RMSprop
from tensorflow.compat.v1.keras.metrics import binary_crossentropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import io
import os
import yaml
from datetime import datetime
import boto3
import s3fs

AUTOTUNE = tf.data.experimental.AUTOTUNE


# you will need to create your own aws credentials to access S3
creds = yaml.safe_load(open('aws_credentials.yml'))
AWS_ACCESS_KEY = creds['aws_access_key_id']
AWS_SECRET_ACCESS_KEY = creds['aws_secret_key_id']
BUCKET_NAME = 'cs205-project-xray'
S3_DIRECTORY = 's3://' + BUCKET_NAME

HEIGHT = 32 #256
WIDTH = 32 #256
BATCH_SIZE = 32
BUFFER_SIZE = 64

# connect to AWS S3
s3_resource = boto3.resource('s3',
                             aws_access_key_id=AWS_ACCESS_KEY,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

my_bucket = s3_resource.Bucket(name=BUCKET_NAME)



###
# Helper functions to create data set
# - modified from Ashref Maiza's code in the link below 
# - link: https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72
###

def format_img_and_label(im_file, lab):
    img_dir = S3_DIRECTORY + '/images/' + im_file
    # read in image tensor
    img_str = tf.io.read_file(img_dir)   
    # decode image
    img_decoded = tf.image.decode_png(img_str, channels=3)
    # resize image to fixed shape
    resized_img = tf.image.resize(img_decoded, [HEIGHT, WIDTH])
    # normalize image
    normalized_img = resized_img / 255.0
    return normalized_img, lab


def create_dataset(filenames, labels):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(format_img_and_label, num_parallel_calls=AUTOTUNE)
    return dataset


# fetch links to image files from S3
loaded_imgs = []

for image_file in my_bucket.objects.filter(Prefix='images').limit(8840):
    loaded_imgs.append(image_file.key[7:])

    
# load image labels
df = pd.read_csv('s3://cs205-project-xray/Data_Entry_2017_v2020.csv')
df = df[df['Image Index'].isin(loaded_imgs)].copy()
img_and_label = df[["Image Index", "Finding Labels"]].copy()
img_and_label['Finding Labels'] = img_and_label['Finding Labels'].str.split('|')

# create data sets
X_train, X_test, y_train, y_test = train_test_split(img_and_label['Image Index'], 
                                                  img_and_label['Finding Labels'], 
                                                  test_size=0.2)

X_train = list(X_train)
X_test = list(X_test)
y_train = list(y_train)
y_test = list(y_test)


MLB = MultiLabelBinarizer()
MLB.fit(y_test)

# multi-label binary transform our labels
y_train = MLB.transform(y_train)
y_test = MLB.transform(y_test)

train_data = create_dataset(X_train, y_train)
test_data = create_dataset(X_test, y_test)


# model building and training
cnn_base = MobileNet(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

model = tf.keras.models.Sequential([
  cnn_base,
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dropout(0.1),
  layers.Dense(64, activation='relu'),
  # 15 labels to classify
  layers.Dense(15, activation='sigmoid')
])

# final number of parameters after adding on head layer to base MobileNet
model.summary()

model.compile(optimizer=RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_data.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE),
                    validation_data=test_data.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE),
                    epochs=5)

