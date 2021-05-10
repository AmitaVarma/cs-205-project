import tensorflow as tf

from tensorflow.compat.v1.keras.applications import MobileNet
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras.optimizers import Adam, RMSprop
from tensorflow.compat.v1.keras.metrics import binary_crossentropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import pandas as pd
import pyspark

import io
import yaml
from datetime import datetime
import boto3
import s3fs

AUTOTUNE = tf.data.experimental.AUTOTUNE

from pyspark import SparkContext, SparkConf
from elephas.spark_model import SparkModel

conf = SparkConf().setAppName('Elephas_App')
sc = SparkContext(conf=conf)

print(tf.__version__)

image_dim = (128, 128, 3)
cnn_base = MobileNet(weights='imagenet', include_top=False, input_shape=image_dim)

model = tf.keras.models.Sequential([
  cnn_base,
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dropout(0.1),
  layers.Dense(64, activation='relu'),
  # 15 labels to classify
  layers.Dense(15, activation='sigmoid')
])


model.compile(optimizer=RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Create the datasets

# S3 credentials
creds = yaml.safe_load(open('aws_credentials.yml'))
AWS_ACCESS_KEY = creds['aws_access_key_id']
AWS_SECRET_ACCESS_KEY = creds['aws_secret_key_id']
BUCKET_NAME = 'cs205-project-xray'
S3_DIRECTORY = 's3://' + BUCKET_NAME

s3_resource = boto3.resource('s3',
                             aws_access_key_id=AWS_ACCESS_KEY,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
my_bucket = s3_resource.Bucket(name=BUCKET_NAME)

# Get the data from S3
loaded_imgs = []
for image_file in my_bucket.objects.filter(Prefix='images'):
    # record which images were downloaded (since this is not the full dataset)
    loaded_imgs.append(image_file.key[7:])
df = pd.read_csv('s3://cs205-project-xray/Data_Entry_2017_v2020.csv')
df = df[df['Image Index'].isin(loaded_imgs)]
img_and_label = df[["Image Index", "Finding Labels"]]
img_and_label['Finding Labels'] = img_and_label['Finding Labels'].str.split('|')

X_train, X_test, y_train, y_test = train_test_split(img_and_label['Image Index'],
                                                  img_and_label['Finding Labels'],
                                                  test_size=0.2)

X_train = list(X_train)
X_test = list(X_test)
y_train = list(y_train)
y_test = list(y_test)

with open('test_index', 'w') as test_images:
    test_images.write(str(X_test)+'\n')

MLB = MultiLabelBinarizer()
MLB.fit(y_test)

# multi-label binary transform our labels
y_train = MLB.transform(y_train)
y_test = MLB.transform(y_test)

def read_image(bucket_name, file_name):
    img_str = tf.io.read_file(bucket_name+file_name)
    img_decoded = tf.image.decode_png(img_str, channels=3)
    return img_decoded

def format_img_and_label(tup):
    im_file, lab = str(tup[0].numpy().decode('utf-8')), tup[1].numpy()
    img_dir = S3_DIRECTORY + '/images/' + im_file
    img_decoded = read_image(S3_DIRECTORY, '/images/'+im_file)
    resized_img = tf.image.resize(img_decoded, [image_dim[0], image_dim[1]])
    normalized_img = resized_img / 255.0
    return normalized_img, lab

def create_dataset(filenames, labels):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    rdd = sc.parallelize(dataset).map(format_img_and_label).persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    print('Num. of partitions: ',rdd.getNumPartitions())
    return rdd

train_data = create_dataset(X_train, y_train)
test_data = create_dataset(X_test, y_test)

spark_model = SparkModel(model, mode='asynchronous')
spark_model.fit(train_data, epochs=5, batch_size=32, verbose=1, validation_split=0.2)

spark_model.save('spark_model.h5')

X_test = np.array(test_data.map(lambda x: x[0]).collect())
y_test = np.array(test_data.map(lambda x: x[1]).collect())

test_results = spark_model.evaluate(X_test, y_test, verbose=2)

with open('test_results', "w") as results:
    results.write(str(test_results)+'\n')
    results.write(str(spark_model.training_histories)+'\n')
