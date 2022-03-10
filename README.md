# VGG
Implementation VGG-19 Architecture
<br />
VGG19 architecture is well depicted in the following image
![Alt text](https://www.researchgate.net/profile/Clifford-Yang/publication/325137356/figure/fig2/AS:670371271413777@1536840374533/llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means.jpg "VGG19 arrchitecture")

# Install packages
run !pip install -r requirements.txt

# Train.py
To train model run:
* !python train.py --path ./dataset --bath_size 1 --num_epochs 1 --save_period 1 --name Test_VGG --save_path ./saved_model

# Loader.py
* Load image data from folder according to classes.
* Example:
train

> dog

>> img1.jpg

>> img2.jpg

> cat

>> img1.jpg

>> img2.jpg
