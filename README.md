# Augmented Dense-Sparse Retriever

*This project aims to produce an enhanced retriever, both taking advantages of `sparse` AND `dense` passage embeddings*
*by Noah Lee, Ji Hun Keom*<br/>


## Requirements
```
$ cd baseline
$ pip install -r requirements.txt
```

## Dataset
Two separate datasets are utilized for our model. First, for its comprehensive yet representative traits, all the contexts passages are crawled from the 2Korean Wikipedia Dataset, consisting of a total of 60613 context passages, further preprocessed for effective retrieval. Then, we filter the 3 KLUE/MRC dataset in a {question, context} pair format for training. From the 4792 resulting pairs, we fix 15 percent as the test set and use 10 percent of the remaining data for validation.

## Training
### 1. Train DPR
```
$ python train_dpr.py --run_name [name of run] --dpr_epochs [# of epoch]
```

## Retrieve and Evaluate
### 1. Retrieve and Evaluate Dense / Sparse Embedding
```
$ python retrieval.py --retrieve_dir [model directory]
```
### 2. Evaluate Augmented Embedding
You have to run dense / sparse retrieval prior to running to following code.
```
$ python augment_retrieval.py [context path] [data path] [test data path] [passsage emb path] [query emb path]
```
