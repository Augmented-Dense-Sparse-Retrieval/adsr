import numpy as numpy
import torch
import pickle, os, json, sys
from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from sklearn.feature_extraction.text import TfidfVectorizer
import transformers
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.sparse import hstack

def normalize(emb):
    norm = np.linalg.norm(emb, axis=1)
    return emb /= norm[:, np.newaxis)
                       
def retrieve_all(q, p):
    k = 10
    result = q * p.T
    if not isinstance(result, np.ndarray):
        result = result.toarray()
    doc_scores = []
    doc_indices = []
    for i in range(result.shape[0]):
        sorted_result = np.argsort(result[i, :])[::-1]
        doc_scores.append(result[i, :][sorted_result].tolist()[:k])
        doc_indices.append(sorted_result.tolist()[:k])

    total = []
    for idx, example in tqdm(enumerate(test_set)):
        tmp = {
            "question": example["question"],
            "id": example["id"],
            "context_id": doc_indices[idx],
            "context": "<SEP>".join(
                [contexts[pid] for pid in doc_indices[idx]]
            ),
        }
        if "context" in example.keys() and "answers" in example.keys():
        #     # if validation set
            tmp["original_context"] = example["context"]
            # tmp["answers"] = example["answers"]
        total.append(tmp)
    cqas = pd.DataFrame(total)

    return cqas


def retriever_prec_k(topk_list, retrieved_df):
    result_dict = {}
    count = [0]*len(topk_list)

    # iterate through each passage+query pair
    for ind, _ in enumerate(range(len(retrieved_df))):
        contexts = retrieved_df['context'][ind].split('<SEP>')
        gold_answer = retrieved_df['original_context'][ind]
        for order, k in enumerate(topk_list):
            if gold_answer in contexts[:k]: 
                count[order] += 1
    # print(count)

    # compute precision at each k
    for ind, k in enumerate(topk_list):        
        result_dict[f'P@{k}'] = f'{round(count[ind]/len(retrieved_df)*100,1)}%'

    return result_dict

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
    with open(sys.argv[4], "rb") as file:
        p_emb = pickle.load(file)
    with open(sys.argv[5], "rb") as file:
        q_emb = pickle.load(file)

    p_emb = normalize(p_emb)
    q_emb = normalize(q_emb)

    aug_p_2 = hstack((p_emb, c))
    aug_q_2 = hstack((q_emb, q))

    cqas = retrieve_all(aug_q_2, aug_p_2)


    print(retriever_prec_k([1,3,5,10], cqas))
