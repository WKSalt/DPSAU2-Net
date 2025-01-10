# Enhanced U2-Net with Residual Depthwise Separable and Split-Attention Blocks for Precise Medical Image Segmentation 
## DPSAU2-Net Architecture
![image](https://github.com/user-attachments/assets/712bb1c8-3703-4e06-8dda-11c050d8f0eb)
## Datasets
- 2018 Data Science Bowl is publicly available at [https://www.kaggle.com/competitions/data-science-bowl-2018/data](https://www.kaggle.com/competitions/data-science-bowl-2018/data)
- CVC-ClinicDB is publicly available at [https://www.kaggle.com/datasets/balraj98/cvcclinicdb](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)
- To apply the model on a custom dataset, the data tree should be constructed as:
```bash
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
```
## Requirement
- Python 3.6
- Pytorch 1.10.0
- CUDA 11.3
```bash
pip install -r requirements.txt
```
## CSV generation
```bash
python data_split_csv.py --dataset your/data/path --size 0.9
```
## Train
```bash
python train.py --dataset your/data/path --csvfile your/csv/path --loss dice --batch 16 --lr 0.001 --epoch 200
```
## Evaluation
```bash
python eval_binary.py --dataset your/data/path --csvfile your/csv/path --model save_models/epoch_last.pth --debug True
```
