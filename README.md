# Graph Contrastive Learning for Anomaly Detection 

GraphCAD has been Accepted as the regular paper by IEEE Transactions on Knowledge and Data Engineering (**TKDE 2022**).

**Brief Intro.** There are **THREE** strengths of GraphCAD that need to be recorded:
+ GraphCAD proposes to leverage the supervised graph contrastive learning for contrasting abnormal nodes with normal ones in terms of their distances to the global context. 
+ GraphCAD deceives a new GNN framework that can infer, remove suspicious links, as well as learn the global context of the input graph simultaneously.
+ GraphCAD designs the corrupting strategy to yield synthetic labels for tackling the label scarcity issue.

```
@article{chen2022gccad,
  title={GCCAD: Graph Contrastive Learning for Anomaly Detection},
  author={Chen, Bo and Zhang, Jing and Zhang, Xiaokang and Dong, Yuxiao and Song, Jian and Zhang, Peng and Xu, Kaibo and Kharlamov, Evgeny and Tang, Jie},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022},
  publisher={IEEE}
}
```

### Get Started

#### Setup
+ Hardware: GPUs with memory exceeds 32G are recommended. Original GraphCAD is running at Nvidia V100s.
+ Dependencies: ```pip install -r requirements.txt```
+ Some issues about PyG: We slightly modify the GIN code in the PyG package for adding edge weight. To reproduce the experimental results in Alpha and Yelp that use GIN as the backbone, you need to add the ```gin_conv_weight.py``` to the path ```{your install path}/torch_geometric/nn/conv/```, and also add a statement ```from .gin_conv_weight import GINConv_w``` to the ```__init__.py``` in the same path. 

#### Data Download
We prepared the processed data through Google Drive: https://drive.google.com/drive/folders/1mX6dYcZXZv51F1NquxY_eOw5zwFnsIbT?usp=sharing

### Run
+ Go to the corresponding file,
+ **For AMiner and MAG**
    + CUDA_VISIBLE_DEVICES={Device_Id} python main.py --train_dir {train dir} --test_dir {test_dir}
+ **For Alpha and Yelp**
    + CUDA_VISIBLE_DEVICES={Device_Id} python main.py --data_dir {data_dir}



If you want to process the data by your own self. Here are some instructions:
**For AMiner and MAG**
...
**For Alpha and Yelp**
We directly adopt the BitCoin Alpha from [Srijan et al.](https://www-cs.stanford.edu/~srijan/pubs/rev2-wsdm18.pdf), and also the Yelp data from [Dou et al.](https://arxiv.org/pdf/2008.08692.pdf).
