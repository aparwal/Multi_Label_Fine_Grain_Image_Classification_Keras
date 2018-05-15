# MultiLabel FineGrain Image Classification 
Using Keras for <b>iMaterialist_challenge_FGVC5</b> at <b>CVPR 18</b>

## Description
As of now, network is fine-tuned with InceptionV3 without the fc layers is used as a feature extractor

Class ImageDataGeneratorMultiLabel extends ImageDataGenerator to work with multilabel data

|Dataset|Number of images|
|:-----|:--------------|
|Train  | 1,014,544|
|Test| 39,706|
|Validation| 9,897|

> Under development

## Usage
For training, run
```
transfer_trail.py
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
