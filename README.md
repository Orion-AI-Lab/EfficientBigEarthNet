
# EfficientBigEarthNet

Code and models from the paper [Efficient deep learning models for land cover image classification](https://arxiv.org/abs/2111.09451) .

All models will be uploaded [here](). (TO BE UPDATED.)

## Available pretrained models:
- ### Standard architectures
  - [DenseNet121]()
  - [DenseNet169]()
  - [DenseNet201]()
  - [ResNet50]()
  - [ResNet101]()
  - [ResNet152]()
  - [VGG16]()
  - [VGG19]()
  - [ViT/6](vit)
  - [ViT/12]()
  - [ViT/20]()
  - [ViT/30]()
  - [ViT/40]()
  - [ViTM/20]()
  - [MLPMixer]()
  - [MLPMixer_Tiny]()

-  ### Wide-ResNet based Architectures
    - [WideResNet]()
    - [WideResNet-COORD]()
    - [WideResNet-CBAM]()
    - [WideResNet-CBAM-Ghost]()
    - [WideResNet-COORD]()
    - [WideResNet-COORD-Ghost]()
    - [WideResNet-SE]() 
    - [WideResNet-SE-Ghost]()
    - [WideResNet-ECA-GHOST]()


- ### EfficientNet Family
  - #### Traditional EfficientNet Architecture
    - [EfficientNetB0]()
    - [EfficientNetB1]()
    - [EfficientNetB2]()
    - [EfficientNetB3]()
    - [EfficientNetB4]()
    - [EfficientNetB5]()
    - [EfficientNetB6]()
    - [EfficientNetB7]()
  
  - ### Augmented EfficientNet
    - [EfficientNet-CBAM]()
    - [EfficientNet-CBAM-GHOST]()
    - [EfficientNet-COORD]()
    - [EfficientNet-COORD-GHOST]()
    - [EfficientNet-SE]()
    - [EfficientNet-SE-GHOST]()
    - [EfficientNet-ECA-GHOST]()
    
  - #### ECA EfficientNet Architectures
    -  [EfficientNetB0-ECA]()
    -  [EfficientNetB1-ECA]() 
    -  [EfficientNetB2-ECA]()
    -  [EfficientNetB3-ECA]()
    -  [EfficientNetB4-ECA]()
    -  [EfficientNetB5-ECA]()
    -  [EfficientNetB6-ECA]()
    -  [EfficientNetB7-ECA]()
  
  -  #### Wide-ResNet-ECA based EfficientNet Architectures:
     - [WideResNet-ECA-B0]() 
     - [WideResNet-ECA-B1]()
     - [WideResNet-ECA-B2]()
     - [WideResNet-ECA-B3]()
     - [WideResNet-ECA-B4]()
     - [WideResNet-ECA-B5]()
     - [WideResNet-ECA-B6]()
     - [WideResNet-ECA-B7]()


## Requirements :

```  tensorflow==2.4.1 ```, ``` horovod==0.21.0  ```

## Citation 

If you use the models or code provided in this repo, please consider citing our paper:
```
@misc{papoutsis2021efficient,
      title={Efficient deep learning models for land cover image classification}, 
      author={Ioannis Papoutsis and Nikolaos-Ioannis Bountos and Angelos Zavras and Dimitrios Michail and Christos Tryfonopoulos},
      year={2021},
      eprint={2111.09451},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
