
# ImGAGN: Imbalanced Networks Embedding via Generative Adversarial Graph Networks

This is our Pytorch implementation for the [paper](https://arxiv.org/abs/2106.02817):

> Liang Qu, Huaisheng Zhu, Ruiqi Zheng, Yuhui Shi, and Hongzhi Yin. 2021. ImGAGN:Imbalanced Network Embedding via Generative Adversarial Graph Networks. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’21), August 14–18, 2021, Virtual Event, Singapore. ACM, New York, NY, USA, 9 pages. https://doi.org/10.11453447548.3467334

## Introduction

This work presents a generative adversarial graph network model, called ImGAGN to address the imbalanced classification problem on graphs. It introduces a novel generator for graph structure data, named **GraphGenerator**, which can simulate both the minority class nodes’ attribute distribution and network topological structure distribution by generating a set of synthetic minority nodes such that the number of nodes in different classes can be balanced. 

## Requirements

+ PyTorch >= 0.4 
+ Python >= 3.6

## Usage

### Dataset

The datasets can be downloaded from [Cora](https://relational.fit.cvut.cz/dataset/CORA), [Citeseer](https://linqs.soe.ucsc.edu/data), [Pubmed](https://linqs.soe.ucsc.edu/data) and [DBLP](https://www.aminer.cn/citation#b541). Take Cora dataset as an example:

- make a folder called **dataset** at root directory.

- make a folder called **cora** in **dataset** directory.

- **Cora** folder contains four files:

  - "edge.cora": each line represents a edge between two nodes with the follwing format:

    ```
    1397 1470
    1397 362
    ... ...
    ```

  - "feature.cora": each line represents the features (e.g., one-hot feature) of nodes with the following format:

    ```
    0 0 0 0 1 0 0 0,...,0
    0 1 0 1 0 0 0 0,...,0
    ...
    ```

  - "train.cora": each line represents the node ID in training set with the following format:

    ```
    0
    2
    ...
    ```

  - "test.cora": each line represents the node ID in test set with the following format:

    ```
    2159
    2160
    ...
    ```

### *Example Usage*

```
cd ImGAGN
python train.py
```

## Citation

```
@misc{qu2021imgagnimbalanced,
      title={ImGAGN:Imbalanced Network Embedding via Generative Adversarial Graph Networks}, 
      author={Liang Qu and Huaisheng Zhu and Ruiqi Zheng and Yuhui Shi and Hongzhi Yin},
      year={2021},
      eprint={2106.02817},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```





