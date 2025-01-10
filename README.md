# Enhanced U2-Net with Residual Depthwise Separable and Split-Attention Blocks for Precise Medical Image Segmentation 
## Datasets
- 2018 Data Science Bowl is publicly available at [https://www.kaggle.com/competitions/data-science-bowl-2018/data](https://www.kaggle.com/competitions/data-science-bowl-2018/data)
- CVC-ClinicDB is publicly available at [https://www.kaggle.com/datasets/balraj98/cvcclinicdb](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)
- To apply the model on a custom dataset, the data tree should be constructed as:
data/
├── images/
│   ├── image_1.png
│   ├── image_2.png
│   ├── ...
│   └── image_n.png
├── masks/
│   ├── image_1.png
│   ├── image_2.png
│   ├── ...
│   └── image_n.png


复制代码

## requirement
- Python 3.6
- Pytorch 1.10.0
- CUDA 11.3
```bash
pip install -r requirements.txt
```
## Introduction
- put the training images and labels into "data/image" and "data/masks" respectively.
- Begin by utilizing `data_split_csv.py` to preprocess and partition the dataset.  
- Subsequently, execute `train.py` to train the model on the prepared data.
