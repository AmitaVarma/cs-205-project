# CS205 Final Project - Group 4
## Parallelizing a chest X-ray deep learning system

## Overview
Chest X-rays are the most common diagnostic tool for used in pratice today in medicine. Integrating deep learning models medical imaging to complement radiologists has the potential to improve people's health, as well as financial costs for individuals and healthcare systems. However, in order for such system to be successful in accuracy, speed, and cost, it must first learn from many X-ray images of various chest-related pathologies, which requires a large amount of data and computing resources to train such a model in a reasonable amount of time. Our project aims to speed-up this traininng process by using big data and big compute tools, namely Spark (through Elephas) and GPU accelerated computing.


## Data
The image data is stored in a publicly available AWS S3 bucket (`s3://cs205-project-xray/images`). Each image is associated with one or more of 15 labels of disease pathologies. A csv file of the labels can be found here: `s3://cs205-project-xray/Data_Entry_2017_v2020.csv`.


## Setup
We used a AWS Deep Learning AMI with Conda to provide a virtual environment with most of the tools needed to run our code. More specifically, we used AWS Deep Learning AMI (Ubuntu 18.04), which you can find [here](https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-AMI-Ubuntu-1/B07Y43P7X5). This was installed on an `[INSERT INSTANCE TYPE]` AWS EC2 instance. Once we SSH'ed into our EC2 instance, type the command
```
source activate tensorflow2_p36
```

to use Tensorflow, scikit-learn and other Python data packages.

Because the AMI does not come pre-installed with Elephas, we must install it separately. To install use the command below:
```
pip install --upgrade git+https://github.com/maxpumperla/elephas
```

### Parallel Execution
We launch an EMR Cluster on AWS with the following setup. This gives us tensorflow out of the box, along with all Spark related dependencies.
![AWS EMR Configuration](docs/AWS_EMR_Config.jpg)

Additionally, we can use the script ```./elephas_libs.sh``` on each host to install elephas with its required versions of PySpark. The script also installs a few other dependencies with scikit-learn and pandas.
At this point, our EMR Cluster has spark 2.4.7, elephas 2.1.0, along with the package versions listed above, and the code can now be run on the cluster.
We can do a simple ```spark-submit --driver-memory 5G xray_elephas.py``` to begin the model training.

