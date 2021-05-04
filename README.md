# CS205 Final Project - Group 4
## Parallelizing a chest X-ray deep learning system

## Overview
Chest X-rays are the most common diagnostic tool for used in pratice today in medicine. Integrating deep learning models medical imaging to complement radiologists has the potential to improve people's health, as well as financial costs for individuals and healthcare systems. However, in order for such system to be successful in accuracy, speed, and cost, it must first learn from many X-ray images of various chest-related pathologies, which requires a large amount of data and computing resources to train such a model in a reasonable amount of time. Our project aims to speed-up this traininng process by using big data and big compute tools, namely Spark and GPU accelerated computing.


## Data
The image data is stored in a publicly available AWS S3 bucket (`s3://cs205-project-xray/images`). Each is associated with one or more of 15 labels of disease pathologies. This can be found here: `s3://cs205-project-xray/Data_Entry_2017_v2020.csv`.