
# EfficientBigEarthNet

Code and models from the paper [Efficient deep learning models for land cover image classification](https://arxiv.org/abs/2111.09451) .

All models will be uploaded [here](). (In progress.)

## Available pretrained models:
- ### Standard architectures
  - #### CNNs  
    - [DenseNet121]()
    - [DenseNet169](https://www.dropbox.com/s/qh6cnryod7uric7/checkpoint_DenseNet169.zip?dl=0)
    - [DenseNet201]()
    - [ResNet50]()
    - [ResNet101]()
    - [ResNet152]()
    - [VGG16]()
    - [VGG19]()
  - #### Vision Transformers
    - [ViT/6](vit)
    - [ViT/12]()
    - [ViT/20]()
    - [ViT/30]()
    - [ViT/40]()
    - [ViTM/20]()
  - #### MLP Mixer
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

## Usage:
  To run an experiment modify the [config file](configs/base.json) and execute train.py. Example for MLPMixer with batch size = 100 and learning rate 1e-4:
  ```

{
    "model_name": "MLPMixer",
    "hparams": {"phi": 1.0, "alpha": 1.0, "beta": 1.0, "gamma": 1.0, "dropout": 0.1},
    "batch_size": 100,
    "nb_epoch": 30,
    "learning_rate": 1e-4,
    "save_checkpoint_after_iteration": 0,
    "save_checkpoint_per_iteration": 1,
    "tr_tf_record_files": ["/work2/pa20/ipapout/gitSpace/TF1.10.1gpu_Py3/NikosTmp/v2/bigearthnet-noa-hua/bigearthnet-tf2/fulldataset/split-10nodes-fulldataset/train*.tfrecord"],
    "val_tf_record_files": ["/work2/pa20/ipapout/gitSpace/TF1.10.1gpu_Py3/NikosTmp/v2/bigearthnet-noa-hua/bigearthnet-tf2/fulldataset/split-10nodes-fulldataset/val*.tfrecord"],
    "test_tf_record_files": ["/work2/pa20/ipapout/gitSpace/TF1.10.1gpu_Py3/NikosTmp/v2/bigearthnet-noa-hua/bigearthnet-tf2/fulldataset/split-10nodes-fulldataset/test*.tfrecord"],
    "label_type": "BigEarthNet-19",
    "fine_tune": false,
    "shuffle_buffer_size": 5000,
    "training_size": 269695,
    "val_size": 125866,
    "test_size": 125866,
    "decay_rate": 0.1,
    "backward_passes": 4,
    "decay_step": 27,
    "label_smoothing": 0,
    "mode": "train",
    "eval_checkpoint": "/work2/pa20/ipapout/gitSpace/TF1.10.1gpu_Py3/NikosTmp/v2/charmbigearth/bigearthnet-tf2/bestTestResNet50/checkpoint_ResNet50",
    "augment": true
}

```

To execute in a single-GPU machine:
```
python3 train.py --parallel=False
```

or for multi node training : 
```
horovodrun --gloo -np $SLURM_NTASKS -H $WORKERS --network-interface ib0 --start-timeout 120 --gloo-timeout-seconds 120 python3 train.py --parallel=True
```

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
