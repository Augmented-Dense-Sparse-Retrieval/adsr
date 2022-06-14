import numpy as numpy
import torch
import pickle, os, json, sys
from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import transformers
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.sparse import hstack
from utils import *

if __name__ == "__main__:
    tokenize_fn = AutoTokenizer.from_pretrained('klue/bert-base')

    tfidfv = TfidfVectorizer(
                tokenizer=tokenize_fn.tokenize,
                ngram_range=(1, 2),
                max_features=534298,
            )

    context_path = sys.argv[1]
    data_path = sys.argv[2]
    with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
        wiki = json.load(f)

    contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    test_set = load_from_disk(sys.argv[3])
    queries = test_set["question"]

    c = tfidfv.fit_transform(contexts)
    q = tfidfv.transform(queries)
    
    clf = TruncatedSVD(768)
    c_svd = clf.fit_transform(c)
    q_svd = clf.transform(q)
    
    with open(sys.argv[4], "rb") as file:
        p_emb = pickle.load(file)
    with open(sys.argv[5], "rb") as file:
        q_emb = pickle.load(file)

    p_emb = normalize(p_emb)
    q_emb = normalize(q_emb)
    
    aug_p = p_emb + c_svd
    aug_q = q_emb + q_svd

    cqas = retrieve_all(aug_q, aug_p)


    print(retriever_acc_k([1,3,5,10], cqas))
