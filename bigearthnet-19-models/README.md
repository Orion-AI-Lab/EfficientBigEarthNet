# BigEarthNet-19 Deep Learning Models
This repository contains code to use the BigEarthNet archive with a new class nomenclature (BigEarthNet-19) for deep learning applications. The new class nomenclature was defined by interpreting and arranging the CORINE Land Cover (CLC) Level-3 nomenclature based on the properties of Sentinel-2 images. The new class nomenclature is the product of a collaboration between the [Direção-Geral do Território](http://www.dgterritorio.pt/) in Lisbon, Portugal and the [Remote Sensing Image Analysis (RSiM)](https://www.rsim.tu-berlin.de/) group at TU Berlin, Germany.

If you use the BigEarthNet-19 or our pre-trained models, please cite the papers given below:

> G. Sumbul, J. Kang, T. Kreuziger, F. Marcelino, H. Costa, P. Benevides, M. Caetano, B. Demir, “[BigEarthNet Deep Learning Models with A New Class-Nomenclature for Remote Sensing Image Understanding](https://arxiv.org/pdf/2001.06372)”, CoRR, abs/2001.06372, 2020

```
@inproceedings{BigEarthNet-19,
    author = {Gencer Sumbul and Jian Kang and Tristan Kreuziger and Filipe Marcelino and Hugo Costa and Pedro Benevides and Mario Caetano and Begüm Demir},
    title = {BigEarthNet Deep Learning Models with A New Class-Nomenclature for Remote Sensing Image Understanding},
    year = {2020},
    month= {January},
    archivePrefix = {arXiv},
    eprint = {2001.06372},
}
```

> G. Sumbul, M. Charfuelan, B. Demir, V. Markl, “[BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding](http://bigearth.net/static/documents/BigEarthNet_IGARSS_2019.pdf)”, IEEE International Geoscience and Remote Sensing Symposium, pp. 5901-5904, Yokohama, Japan, 2019.

```
@inproceedings{BigEarthNet,
    author = {Gencer Sumbul and Marcela Charfuelan and Begüm Demir and Volker Markl},
    title = {BigEarthNet: A Large-Scale Benchmark Archive For Remote Sensing Image Understanding},
    booktitle = {IEEE International Geoscience and Remote Sensing Symposium}, 
    year = {2019},
    pages = {5901--5904}
    doi = {10.1109/IGARSS.2019.8900532}, 
    month = {July}
}
```

If you are interested in BigEarthNet with the original CLC Level-3 class nomenclature, please check [here](https://gitlab.tu-berlin.de/rsim/bigearthnet-models/tree/master).

# Pre-trained Deep Learning Models on BigEarthNet-19
We provide code and model weights for the following deep learning models that have been pre-trained on BigEarthNet with the new class nomenclature (BigEarthNet-19) for scene classification:


| Model Names  | Pre-Trained TensorFlow Models                                | Pre-Trained PyTorch Models                                   |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| K-Branch CNN | [K-BranchCNN.zip](http://bigearth.net/static/pretrained-models/BigEarthNet-19_labels/K-BranchCNN.zip) | Coming soon                                                  |
| VGG16        | [VGG16.zip](http://bigearth.net/static/pretrained-models/BigEarthNet-19_labels/VGG16.zip) | Coming soon                                                  |
| VGG19        | [VGG19.zip](http://bigearth.net/static/pretrained-models/BigEarthNet-19_labels/VGG19.zip) | Coming soon                                                  |
| ResNet50     | [ResNet50.zip](http://bigearth.net/static/pretrained-models/BigEarthNet-19_labels/ResNet50.zip) | [ResNet50.pth.tar](http://bigearth.net/static/pretrained-models-pytorch/BigEarthNet-19_labels/ResNet50.pth.tar) |
| ResNet101    | [ResNet101.zip](http://bigearth.net/static/pretrained-models/BigEarthNet-19_labels/ResNet101.zip) | Coming soon                                                  |
| ResNet152    | [ResNet152.zip](http://bigearth.net/static/pretrained-models/BigEarthNet-19_labels/ResNet152.zip) | Coming soon                                                  |

The TensorFlow code for these models can be found [here](https://gitlab.tu-berlin.de/rsim/bigearthnet-models-tf).

The PyTorch code for these models can be found [here](https://gitlab.tubit.tu-berlin.de/rsim/bigearthnet-models-pytorch).

# Generation of Training/Test/Validation Splits
After downloading the raw images from https://www.bigearth.net, they need to be prepared for your ML application. We provide the script `prep_splits_BigEarthNet-19.py` for this purpose. It generates consumable data files (i.e., TFRecord) for training, validation and test splits which are suitable to use with TensorFlow or PyTorch. Suggested splits can be found with corresponding csv files under `splits` folder. The following command line arguments for `prep_splits_BigEarthNet-19.py` can be specified:

* `-r` or `--root_folder`: The root folder containing the raw images you have previously downloaded.
* `-o` or `--out_folder`: The output folder where the resulting files will be created.
* `-n` or `--splits`: A list of CSV files each of which contains the patch names of corresponding split.
* `-l` or `--library`: A flag to indicate for which ML library data files will be prepared: TensorFlow or PyTorch.
* `--update_json`: A flag to indicate that this script will also change the original json files of the BigEarthNet by updating labels 

To run the script, either the GDAL or the rasterio package should be installed. The TensorFlow package should also be installed. The script is tested with Python 2.7, TensorFlow 1.3, PyTorch 1.2 and Ubuntu 16.04. 

**Note**: BigEarthNet patches with high density snow, cloud and cloud shadow are not included in the training, test and validation sets constructed by the provided scripts (see the list of patches with seasonal snow [here](http://bigearth.net/static/documents/patches_with_seasonal_snow.csv) and that of cloud and cloud shadow [here](http://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv)). 

Authors
-------

**Gencer Sümbül**
http://www.user.tu-berlin.de/gencersumbul/

**Jian Kang**
https://www.rsim.tu-berlin.de/menue/team/dring_jian_kang/

**Tristan Kreuziger**
https://www.rsim.tu-berlin.de/menue/team/tristan_kreuziger/

Maintained by
-------

**Gencer Sümbül** for TensorFlow models

**Jian Kang** for PyTorch models


# License
The BigEarthNet Archive is licensed under the **Community Data License Agreement – Permissive, Version 1.0** ([Text](https://cdla.io/permissive-1-0/)).

The code in this repository to facilitate the use of the archive is licensed under the **MIT License**:

```
MIT License

Copyright (c) 2019 The BigEarthNet Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
