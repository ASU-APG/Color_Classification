# Color_Classification

A demonstration on how to load and ground the attribute can be found at : Demo.ipynb

Original Image -->  Color Classification Result
<p float="center">
  <img src="/images/test1.png" width="200" />
  <img src="/images/result1.png" width="200" />
</p>
<p float="center">
  <img src="/images/test2.png" width="200" />
  <img src="/images/result2.png" width="200" />
</p>

The ResNet 50 is trained on a pixel-wise labled dataset for 11 categories pixel-wise color classification. We use the convolutional features from second or third Res-block.

For a clarification of the file system:

#### train_color_pixel_classifier.py.py:
<pre>Training script for color classification.</pre>


#### /Models/resnet.py:
<pre>ResNet structure script.</pre>

#### /lib/:
<pre>Contains all the neccesary dependencies for our framework</pre>

### /data/:
<pre>Only 400 pixel-labeled images from <a href="http://lear.inrialpes.fr/people/vandeweijer/data.html"> EbayColor Daset</a> are using for training. (18 MB)</pre>
 
### /checkpoint/:
<pre>Pre-trained Model. A pre-trained ResNet 50 can be found <a href="https://drive.google.com/file/d/1eXfOv3pxSNVTDtKPbSH-RI9lDor0UQ3v/view?usp=sharing">here</a>.</pre>
