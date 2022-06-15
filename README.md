# Augmented Dense-Sparse Retriever

*This project aims to produce an enhanced retriever for longer Korean sequences, jointly taking advantages of `sparse` AND `dense` passage embeddings*<br/>
*by Noah Lee, Ji Hun Keom*<br/>


## Requirements
```
$ cd baseline
$ pip install -r requirements.txt
```

## Dataset
Two separate datasets are utilized for our model. First, for its comprehensive yet representative traits, all the contexts passages are crawled from the Korean Wikipedia Dataset, consisting of a total of 60613 context passages, further preprocessed for effective retrieval. Then, we filter the 3 KLUE/MRC dataset in a {question, context} pair format for training. From the 4792 resulting pairs, we fix 15 percent as the test set and use 10 percent of the remaining data for validation.

## Training
### 1. Train DPR_base
```
$ python train_dpr.py --run_name [name of run] --dpr_epochs [# of epoch] \
  --file_suffix [dpr_pickle file] --dpr_train_batch [# train batch] \
  --dpr_eval_batch [# eval batch] --dpr_learning_rate [LR] \
  --dpr_weight_decay [weight_decay rate] --dpr_eval_steps [# steps per evaluation] \
  --dpr_warmup_steps [# warmup steps]
```

### 2. Train DPR_long-sum
```
$ python train_dpr_v1.py --run_name [name of run] --dpr_epochs [# of epoch] \
  --file_suffix [dpr_pickle file] --dpr_train_batch [# train batch] \
  --dpr_eval_batch [# eval batch] --dpr_learning_rate [LR]  \
  --dpr_weight_decay [weight_decay rate] --dpr_eval_steps [# steps per evaluation] \
  --dpr_warmup_steps [# warmup steps]
```


### 3. Train DPR_neg
```
$ python train_dpr_v2.py --num_neg [# of neg samp] --run_name [name of run] \
  --dpr_epochs [# of epoch]  --file_suffix [dpr_pickle file] \
  --dpr_train_batch [# train batch] --dpr_eval_batch [# eval batch] 
  --dpr_learning_rate [LR]  --dpr_weight_decay [weight_decay rate] \
  --dpr_eval_steps [# steps per evaluation] --dpr_warmup_steps [# warmup steps]
```


## Retrieve and Evaluate Single Embedding 
### 1. Single Sparse Embedding (TFIDF)
```
$ python retrieval.py --retriever_type SparseRetrieval_TFIDF --spr_tokenizer [tokenizer] \ 
  --file_suffix [file name] --max_features [TFIDF dim] --top_k_retrieval [# retrieval]
```

### 2. Single Sparse Embedding (BM25)
```
$ python retrieval.py --retriever_type SparseRetrieval_BM25 --spr_tokenizer [tokenizer] \ 
  --file_suffix [file name] --bm25_type [type] --top_k_retrieval [# retrieval]
```


### 3. Single Dense Embedding
```
$ python retrieval.py --retriever_type DenseRetrieval --dpr_model [tokenizer] \ 
  --file_suffix [file name] --dpr_longsum [dpr_type] --top_k_retrieval [# retrieval]
```


## Retrieve and Evaluate ADSR
### 1. Evaluate ADSR-C
You have to run dense / sparse retrieval prior to running to following code.
```
$ python ADSR_C_retrieval.py [context path] [data path] [test data path] [passsage emb path] [query emb path]
```
### 2. Evaluate ADSR-S
You have to run dense / sparse retrieval prior to running to following code.
```
$ python ADSR_S_retrieval.py [context path] [data path] [test data path] [passsage emb path] [query emb path]
```


## Additional Documentation
```
spr_tokenizer: str
    - none
    - klue
    - bigbird
    - kobert_m  (monologg)
    - kobert_s (skt)
    - bert_multi
    
bm25_type: str
    - plus
    - okapi
    - l

retriever_type: str
    - SparseRetrieval_TFIDF
    - SparseRetrieval_BM25
    - DenseRetrieval
```
*refer to arguments.py for further details*
