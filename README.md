# GPT-GNN: Generative Pre-Training of Graph Neural Networks

<p align="center">
  <img src="./gpt-intro.png" width="600">
  <br />
  <br />
</p>


GPT-GNN is a pre-training framework to initialize GNNs by generative pre-training. It can be applied to large-scale and heterogensous graphs.

You can see our KDD 2020 paper [“**Generative Pre-Training of Graph Neural Networks**”](https://arxiv.org/pdf/2006.15437.pdf) for more details.


## Overview
The key package is GPT_GNN, which contains the the high-level GPT-GNN pretraining framework, base GNN models, and base graph structure and data loader.

To illustrate how to apply the GPT_GNN framework for arbitrary graphs, we provide examples of pre-training on both hetergeneous (OAG) and homogeneous graphs (reddit). Both of them are of large-scale.

Within each example_* package, there is a pretrain_* file for pre-training a GNN on the given graph, and also multiple finetune_* files for training and validating on downstream tasks.

## DataSet
For **Open Academic Graph (OAG)**, we provide a heterogeneous graph containing highly-cited CS papers (8.1G) spanning from 1900-2020. You can download the preprocessed graph via this [link](https://drive.google.com/open?id=1a85skqsMBwnJ151QpurLFSa9o2ymc_rq). We split the data by their time: Pre-training ( t < 2014 ); Training ( 2014 <= t < 2017); Validation ( t = 2017 ); Testing ( 2018 <= t ).
If you want to directly process from raw data, you can download via this [link](https://drive.google.com/open?id=1yDdVaartOCOSsQlUZs8cJcAUhmvRiBSz). After downloading it, run `preprocess_OAG.py` to extract features and store them in our data structure. 

For **Reddit**, we simply download the preprocessed graph using pyG.datasets API, and then turn it into our own data structure using `preprocess_reddit.py`. We randomly split the data into different sets.

## Usage

Coming soon.

## Pre-trained Models

Coming soon.


















### Reference

Please consider citing the following paper when using our code for your application.

```bibtex
@inproceedings{gpt_gnn,
  title={GPT-GNN: Generative Pre-Training of Graph Neural Networks},
  author={Ziniu Hu and Yuxiao Dong and Kuansan Wang and Kai-Wei Chang and Yizhou Sun},
  booktitle={Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2020}
}
```


This implementation is mainly based on [pyHGT](https://github.com/acbull/pyHGT) API.
