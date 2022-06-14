import numpy as numpy
import torch
import pickle, os, json
import transformers
from sklearn.decomposition import TruncatedSVD
from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


data_path = "/content/drive/MyDrive/ADSR/baseline2"
pickle_name = "sparse_embeddingk_020.bin"
tfidfv_name = "tfidvk_020.bin"
emd_path = os.path.join(data_path, pickle_name)
tfidfv_path = os.path.join(data_path, tfidfv_name)

if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
    with open(emd_path, "rb") as file:
        p_embedding = pickle.load(file)
    with open(tfidfv_path, "rb") as file:
        tfidfv = pickle.load(file)
    print("Embedding pickle load.")
