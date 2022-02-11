# Small-Object Sensitive Segmentation Using Across Feature Map Attention

This is the source code for the method as described in our paper:
**Small-Object Sensitive Segmentation Using Across Feature Map Attention**. 

<div align=center><img width="1200" height="420" src="https://github.com/ShengtianSang/AFMA/blob/main/figures/overview.jpg"/></div>
<p align="left"> 
<font size=2>The overview of our method. (a) represents an overview of combining the AFMA method with a general semantic segmentation method. The encoder of the segmentation model is input to the AFMA method, and its output is applied to the output of the segmentation method. (b) presents a detailed illustration of combining the AFMA method with different semantic segmentation models. It can be observed that the AFMA approach is adaptable to different types of architectures of various semantic segmentation models and can work on different  layers of the encoder’s feature maps. </font>
</p>

The lines 23-182 of models/attonimage/encoder_channelatt_img.py are used for calculating AFMA between original image and any layer of feature map. The files with *_decoder* suffix in the **deeplabv3 **,  **fpn **,  **linknet **,  **manet **,  **pan **,  **pspnet **,  **unet **,  **unetplusplus** folders are the steps to combine AFMA to the existing model.  You can apply AFMA on your own models, which is a very easy idea to implement.

<div align=center><img width="1200" height="420" src="https://github.com/ShengtianSang/AFMA/blob/main/figures/method.jpg"/></div>
<p align="left"> 
The framework of our method. (a) Calculate the Across Feature Map Attention. The inputs are the initial image and i-th layer feature maps of the encoder. (b) Output Modification. The generated AFMA in (a) is used to modify the output of the decoder’s predicted masks. (c) The process of generating gold AFMA.
</p>

## Requirements
* scikit-learn
* numpy
* tqdm

## Data

In order to use the code, you have to provide 
* [Theraputic Target Database](http://db.idrblab.net/ttd/full-data-download)  You don't need to download by yourself, I have uploaded all the TTD 2016 version in *<./data/TTD>*. 
* [SemedDB](https://skr3.nlm.nih.gov/SemMedDB/) **You need to download from [here](https://pan.baidu.com/s/1zuOELNGAua6i523_nLK6mw)** with password:1234 to obtain the whole knowledge graph. After downloading the "predications.txt" file, please replace the file *<./data/SemedDB/predications.txt>*. with this **new** downloaded file. 

## Run the codes
Install the environment.
```bash
pip install -r requirements.txt
```

Construct training and test data.
```bash
python experimental_data.py
```

Train and test the model.
```bash
python main.py
```

## Illustration of feature selection
<div align=center><img width="800" height="300" src="https://github.com/ShengtianSang/SemaTyP/blob/main/figures/Illustration_of_Feature_selection.jpg"/></div>
<p align="center">
An illustration of the features constructed in our work.
</p>


## File declaration

**data/SemmedDB**： contains all relations extracted from SemmedDB, which are used for constructing the Knowledge Graph in our experiment. The whole "predications.txt" contains **39,133,975** relations, we just leave a small sample "predications.txt" file here which contain **100** relation. The whole "predications.txt" file coule be downloaded from 
  
**data/TTD**： contains the drug, target and disease relations retrieved from Theraputic Target Database.
    
**experimental_data.py**: constuct the drug-target-disease associations from TTD and Knowledge Graph.

**knowledge_graph.py**: construct the Knowledge Graph used in our experiment.
 
**data_loader.py**：used to load traing and test data.

**main.py**：used to train and test the models


# CELT: Using feature layer interactions to improve semantic segmentation models

<p float="center">
  <img width="185" height="180" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/MANet.gif"/>
  &nbsp;
  &nbsp;
  <img width="185" height="180" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/Unet.gif"/> 
    &nbsp;
    &nbsp;
  <img width="170" height="180" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/PSPnet.gif"/>
    &nbsp;
    &nbsp;
  <img width="205" height="175" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/FeaturePN.gif"/>
    &nbsp;
    &nbsp;
  <img width="185" height="180" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/Unet%2B%2B.gif"/>
     &nbsp;
    &nbsp;
  <img width="190" height="180" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/DeepLabV3.gif"/>
     &nbsp;
    &nbsp;
  <img width="180" height="180" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/Linknet.gif"/>
     &nbsp;
    &nbsp;
  <img width="170" height="200" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/PAN.gif"/>
</p>

<p align="center">
  <img width="500" height="70" src="https://github.com/Temporaryanonymous/CELT/blob/main/figure/Figure%20Legend.jpg">
</p>

This is the source code for the method as described in our paper:
**CELT: Using feature layer interactions to improve semantic segmentation models**. The lines 81-139 of Architecture/encoder/resnet.py are about how to insert CELT into the encoder of the existing segmentation models. You can apply CELT on your own models, which is a very easy idea to implement.

## Data

In order to make it easier for the readers to reproduce and understand the code, we have provided a small amount of example data used in our experiment under the **image_samples** folder, where each dataset (CamVid, Skin Lesion, CUB Birds) provides five training, validation and test images.


## File declaration


**Architecture/encoder**：contains the encoding part of all eight CELT plugged models.

**Architecture/manet**：the decoder and segmentation part of the manet_celt model.

**Architecture/unet**： the decoder and segmentation part of the unet_celt model.

**Architecture/unetplusplus**：the decoder and segmentation part of the unet++_celt model.

**Architecture/deeplabv3**：the decoder and segmentation part of the deeplabv3_celt model.

**Architecture/fpn**：the decoder and segmentation part of the fpn_celt model.  

**Architecture/pan**：the decoder and segmentation part of the pan_celt model.

**Architecture/linknet**：the decoder and segmentation part of the linknet_celt model.

**Architecture/pspnet**：the decoder and segmentation part of the pspnet_celt model.

**CELT_Unet_CamVid.py**: The Unet model with CELT method, which is used for CamVid dataset. You can change this file to test other models and datasets.

## Run the codes
Install the environment.
```bash
pip install -r requirements.txt
```

Train and test the model.
```bash
python CELT_Unet_CamVid.py
```
