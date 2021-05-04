# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

base/null.j2 [markdown]
# # 0. Setup

# %%
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.metrics import binary_crossentropy
from keras import backend as K 

tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads = 8, inter_op_parallelism_threads=8)))

#tf.compat.v1.keras.backend.set_session\
#(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement = True, device_count = {'CPU' : 1})))


#config = tf.compat.v1.ConfigProto(allow_soft_placement=True, device_count = {'CPU': 1})
#session = tf.compat.v1.Session(config=config)

# NUM_THREADS = …
# sess = tf.Session(config=tf.ConfigProto(
#     intra_op_parallelism_threads=NUM_THREADS))


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import io
import os
import PIL
import yaml

import boto3
import s3fs


# %%



# %%



# %%
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print(f"tensorflow version {tf.__version__}")
print(f"keras version {tf.keras.__version__}")
print(f"Eager Execution Enabled: {tf.executing_eagerly()}\n")

# Get the number of replicas 
strategy = tf.distribute.MirroredStrategy()
print(f"\nNumber of replicas: {strategy.num_replicas_in_sync}\n")

devices = tf.config.experimental.get_visible_devices()
print(f"Devices: {devices}\n")
print(f"{tf.config.experimental.list_logical_devices('GPU')}\n")

print(f"Num CPU Available: {len(tf.config.list_physical_devices('CPU'))}")
print(f"Num GPU Available: {len(tf.config.list_physical_devices('GPU'))}\n")

print(f"All Pysical Devices: {tf.config.list_physical_devices()}")

# Better performance with the tf.data API
# Reference: https://www.tensorflow.org/guide/datac_performance
AUTOTUNE = tf.data.experimental.AUTOTUNE


# %%
len(tf.config.list_physical_devices('CPU'))


# %%
# you will need to create your own aws credentials to access S3
creds = yaml.safe_load(open('aws_credentials.yml'))
AWS_ACCESS_KEY = creds['aws_access_key_id']
AWS_SECRET_ACCESS_KEY = creds['aws_secret_key_id']
BUCKET_NAME = 'cs205-project-xray'
S3_DIRECTORY = 's3://' + BUCKET_NAME

base/null.j2 [markdown]
# # 1. Data loading
base/null.j2 [markdown]
# ### 1.1 Configurations and helper functions

# %%
HEIGHT = 256
WIDTH = 256
BATCH_SIZE = 32
BUFFER_SIZE = 64

# connect to AWS S3
s3_resource = boto3.resource('s3',
                             aws_access_key_id=AWS_ACCESS_KEY,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

my_bucket = s3_resource.Bucket(name=BUCKET_NAME)


# %%
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
    dataset = dataset.map(format_img_and_label)
    return dataset


# %%
# fetch image data (unused function)
#def fetch_input(path):     
#  img_object = s3_resource.Object(BUCKET_NAME,path)
#  img = PIL.Image.open(img_object.get()['Body'])
#  return(img)

base/null.j2 [markdown]
# ### 1.2 Load and  create data set

# %%
# fetch links to image files from S3
loaded_imgs = []

for image_file in my_bucket.objects.filter(Prefix='images').limit(5000):
  # record which images were downloaded (since this is only a subet of full dataset)
    loaded_imgs.append(image_file.key[7:])
    # make images folder if it doesnt exist already
    #if not os.path.exists(os.path.dirname(image_file.key)):
    #    os.makedirs(os.path.dirname(image_file.key)) 
    # download into local
    #my_bucket.download_file(image_file.key, image_file.key)


# %%
# load image labels
df = pd.read_csv('s3://cs205-project-xray/Data_Entry_2017_v2020.csv')
df = df[df['Image Index'].isin(loaded_imgs)]
img_and_label = df[["Image Index", "Finding Labels"]]
img_and_label['Finding Labels'] = img_and_label['Finding Labels'].str.split('|')


# %%
X_train, X_val, y_train, y_val = train_test_split(img_and_label['Image Index'], 
                                                  img_and_label['Finding Labels'], 
                                                  test_size=0.2)

X_train = list(X_train)
X_val = list(X_val)
y_train = list(y_train)
y_val = list(y_val)

base/null.j2 [markdown]
# Since we have multi-label classification problem, we must encode our labels as a binary vector (15 by 1 vector)

# %%
MLB = MultiLabelBinarizer()
MLB.fit(y_train)

vec2label_dict = {}

# 15 label classes
for (encode_num, label) in enumerate(MLB.classes_):
    vec2label_dict[encode_num] = label
    print(encode_num, label)


# %%
# multi-label binary transform our labels
y_train = MLB.transform(y_train)
y_val = MLB.transform(y_val)


# %%
train_data = create_dataset(X_train, y_train)
test_data = create_dataset(X_val, y_val)

for f, l in test_data.take(1):
    print("Shape of features array:", f.numpy().shape)
    print("Shape of labels array:", l.numpy().shape, '\n')

print(f'Number of images in train set: {tf.data.experimental.cardinality(train_data).numpy()}' )
print(f'Number of images in test set: {tf.data.experimental.cardinality(test_data).numpy()}' )

base/null.j2 [markdown]
# Let's take a look at a small batch of the X-ray images:

# %%
sample_imgs = train_data.take(4)
fig, ax = plt.subplots(2, 2, figsize=(16,16))

# loop over tge four images using numpy iterator
for ax, (img, label) in zip(ax.ravel(), sample_imgs.as_numpy_iterator()):
    # break when no more axes left
    if ax is None:
        break
    indices = [i for i, x in enumerate(label) if x == 1]
    diseases = [vec2label_dict[i] for i in indices]
    ax.imshow(img)
    ax.set_title(f"Diagnoses: {diseases}")
    ax.axis('off')

base/null.j2 [markdown]
# # 2. Model building and training
base/null.j2 [markdown]
# We will use MobileNet, and transfer learning to save time on training.

# %%
cnn_base = MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

cnn_base.summary()


cnn_base.trainable


# %%
cnn_base.trainable


# %%
cnn_base.trainable = True
set_trainable = False
for layer in cnn_base.layers:
    layer.trainable = True
    
    """
    if layer.name == 'conv_dw_10':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
        print(layer.name)
    else:
        layer.trainable = False
        """


# %%
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


# %%
model.compile(optimizer=RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# %%
get_ipython().run_cell_magic('time', '', 'history = model.fit(train_data.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE),\n                    validation_data=test_data.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE),\n                    epochs=20)')


# %%



