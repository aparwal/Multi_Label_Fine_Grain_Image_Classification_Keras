# Multi Label Image Classification 
Using Keras for [iMaterialist_challenge_FGVC5](https://sites.google.com/view/fgvc5/hom) at [CVPR 18](http://cvpr2018.thecvf.com/program/workshops)

## Description
As of now, network is fine-tuned with InceptionV3 without the fc layers is used as a feature extractor

Class ImageDataGeneratorMultiLabel extends [ImageDataGenerator](https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L414) from Keras to work with separately loaded (multi)labels

The Dataset can be downloaded from [challenge website](https://www.kaggle.com/c/imaterialist-challenge-furniture-2018)

|Dataset|Number of images|
|:-----|:--------------|
|Train  | 1,014,544|
|Test| 39,706|
|Validation| 9,897|

> Under development

## Usage
For training, run
```
transfer_trian.py
```

To resume training
```
transfer_resume.py
```

For testing
```
test.py
```

### Depenedency
* Python 3.x
* Keras 2.1.x
* OpenCV 3.x
