#!/usr/bin/env python
# coding: utf-8

# # 0. Setup

# In[1]:


get_ipython().run_cell_magic('time', '', '\nimport tensorflow as tf\nfrom tensorflow.compat.v1.keras.applications import MobileNet\nfrom tensorflow.compat.v1.keras import layers\nfrom tensorflow.compat.v1.keras.optimizers import Adam, RMSprop\nfrom tensorflow.compat.v1.keras.metrics import binary_crossentropy\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import MultiLabelBinarizer\n\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nimport io\nimport os\nimport yaml\nfrom datetime import datetime\nimport boto3\nimport s3fs\n\nAUTOTUNE = tf.data.experimental.AUTOTUNE\n\n\n# you will need to create your own aws credentials to access S3\ncreds = yaml.safe_load(open(\'aws_credentials.yml\'))\nAWS_ACCESS_KEY = creds[\'aws_access_key_id\']\nAWS_SECRET_ACCESS_KEY = creds[\'aws_secret_key_id\']\nBUCKET_NAME = \'cs205-project-xray\'\nS3_DIRECTORY = \'s3://\' + BUCKET_NAME\n\nHEIGHT = 32 #256\nWIDTH = 32 #256\nBATCH_SIZE = 32\nBUFFER_SIZE = 64\n\n# connect to AWS S3\ns3_resource = boto3.resource(\'s3\',\n                             aws_access_key_id=AWS_ACCESS_KEY,\n                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY)\n\nmy_bucket = s3_resource.Bucket(name=BUCKET_NAME)\n\n\n\n###\n# Helper functions to create data set\n# - modified from Ashref Maiza\'s code in the link below \n# - link: https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72\n###\n\ndef format_img_and_label(im_file, lab):\n    img_dir = S3_DIRECTORY + \'/images/\' + im_file\n    # read in image tensor\n    img_str = tf.io.read_file(img_dir)   \n    # decode image\n    img_decoded = tf.image.decode_png(img_str, channels=3)\n    # resize image to fixed shape\n    resized_img = tf.image.resize(img_decoded, [HEIGHT, WIDTH])\n    # normalize image\n    normalized_img = resized_img / 255.0\n    return normalized_img, lab\n\n\ndef create_dataset(filenames, labels):\n    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))\n    dataset = dataset.map(format_img_and_label, num_parallel_calls=AUTOTUNE)\n    return dataset\n\n\n# fetch links to image files from S3\nloaded_imgs = []\n\nfor image_file in my_bucket.objects.filter(Prefix=\'images\').limit(8840):\n    loaded_imgs.append(image_file.key[7:])\n\n    \n# load image labels\ndf = pd.read_csv(\'s3://cs205-project-xray/Data_Entry_2017_v2020.csv\')\ndf = df[df[\'Image Index\'].isin(loaded_imgs)].copy()\nimg_and_label = df[["Image Index", "Finding Labels"]].copy()\nimg_and_label[\'Finding Labels\'] = img_and_label[\'Finding Labels\'].str.split(\'|\')\n\n# create data sets\nX_train, X_test, y_train, y_test = train_test_split(img_and_label[\'Image Index\'], \n                                                  img_and_label[\'Finding Labels\'], \n                                                  test_size=0.2)\n\nX_train = list(X_train)\nX_test = list(X_test)\ny_train = list(y_train)\ny_test = list(y_test)\n\n\nMLB = MultiLabelBinarizer()\nMLB.fit(y_test)\n\n# multi-label binary transform our labels\ny_train = MLB.transform(y_train)\ny_test = MLB.transform(y_test)\n\ntrain_data = create_dataset(X_train, y_train)\ntest_data = create_dataset(X_test, y_test)\n\n\n# model building and training\ncnn_base = MobileNet(weights=\'imagenet\', include_top=False, input_shape=(32, 32, 3))\n\nmodel = tf.keras.models.Sequential([\n  cnn_base,\n  layers.Flatten(),\n  layers.Dense(256, activation=\'relu\'),\n  layers.Dropout(0.1),\n  layers.Dense(64, activation=\'relu\'),\n  # 15 labels to classify\n  layers.Dense(15, activation=\'sigmoid\')\n])\n\n# final number of parameters after adding on head layer to base MobileNet\nmodel.summary()\n\nmodel.compile(optimizer=RMSprop(lr=2e-5),\n              loss=\'binary_crossentropy\',\n              metrics=[\'accuracy\'])\n\n\nhistory = model.fit(train_data.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE),\n                    validation_data=test_data.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE),\n                    epochs=5)')

