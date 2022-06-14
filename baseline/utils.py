import numpy as np
import pandas as pd

def normalize(emb):
    norm = np.linalg.norm(emb, axis=1)
    return emb /= norm[:, np.newaxis]
                       
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
            tmp["original_context"] = example["context"]

        total.append(tmp)
    cqas = pd.DataFrame(total)
                       
    return cqas

def retriever_acc_k(topk_list, retrieved_df):
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
