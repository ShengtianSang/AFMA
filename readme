<div align=center><img width="800" height="450" src="https://github.com/ShengtianSang/SemaTyP/blob/main/figures/Illustration_of_SemKG.jpg"/></div>

# SemaTyP: a knowledge graph based literature mining method for drug discovery

This is the source code and data for the task of drug discovery as described in our paper:
["SemaTyP: a knowledge graph based literature mining method for drug discovery"](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2167-5)

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


## Cite

Please cite our paper if you use this code in your own work:

```
@article{sang2018sematyp,
  title={SemaTyP: a knowledge graph based literature mining method for drug discovery},
  author={Sang, Shengtian and Yang, Zhihao and Wang, Lei and Liu, Xiaoxia and Lin, Hongfei and Wang, Jian},
  journal={BMC bioinformatics},
  volume={19},
  number={1},
  pages={1--11},
  year={2018},
  publisher={Springer}
}
```
