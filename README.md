# Enhanced U2-Net with Residual Depthwise Separable and Split-Attention Blocks for Precise Medical Image Segmentation 
## Network structure of DPSAU2-Net
![image](https://github.com/user-attachments/assets/712bb1c8-3703-4e06-8dda-11c050d8f0eb)
This study introduces a novel approach by integrating residual depthwise separable convolutions into the RSU modules of U2-Net, while also incorporating SA blocks into both the encoder and decoder layers. The RSDPU module effectively aggregates long-range spatial information while preserving critical features from lower-level semantic layers. Meanwhile, the SA block adapts to multi-scale features and enhances feature representation with relatively low computational overhead. In segmentation tasks, features from different regions carry varying levels of importance for semantic categories, and the SA block excels at modeling these relationships.
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
## References
Our implementation is based on [ResNeSt](https://github.com/zhanghang1989/ResNeSt/tree/5fe47e93bd7e098d15bc278d8ab4812b82b49414) and [U2-Net](https://github.com/xuebinqin/U-2-Net). We would like to thank them.
## journal
[The Visual Computer](https://link.springer.com/journal/371?utm_medium=display&utm_source=letpub&utm_content=text_link&utm_term=null&utm_campaign=MPSR_00371_AWA1_CN_CNPL_letpb_mp)
