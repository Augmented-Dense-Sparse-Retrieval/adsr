# Augmented Dense-Sparse Retriever

*This project aims to produce an enhanced retriever, both taking advantages of `sparse` AND `dense` passage embeddings*
*by Noah Lee, Ji Hun Keom*<br/>


## Requirements
```
$ cd baseline
$ pip install -r requirements.txt
```

## Dataset
The dataset is

## Training
### 1. Train DPR
```
$ python train_dpr.py --run_name [name of run] --dpr_epochs [# of epoch]
```

## Evaluate
### 1. Evaluate Dense / Sparse Embedding
```
$ python retrieval.py --retrieve_dir [model directory]
```
### 3. Evaluate Augmented Embedding
