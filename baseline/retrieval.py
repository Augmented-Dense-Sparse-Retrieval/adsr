import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch

from rank_bm25 import BM25Okapi, BM25L, BM25Plus 

from tqdm.auto import tqdm
from contextlib import contextmanager
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, NoReturn, Any, Optional, Union

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)

from sklearn.feature_extraction.text import TfidfVectorizer

from train_dpr import Dense, BertEncoder
from utils import retriever_acc_k, get_dense_args, reconstruct_embedding, split_batch, timer
from transformers import AutoTokenizer, HfArgumentParser
from arguments import DatasetArguments, RetrieverArguments, DefaultArguments




#Sparse Retrieval based on BM25
class SparseRetrieval_BM25:
    def __init__(self, tokenize_fn,data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        file_suffix: Optional[str] = '',
    ) -> NoReturn:
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # for order invariance

        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        self.BM25 = None
        self.tokenizer = tokenize_fn
        self.file_suffix = file_suffix

    def get_sparse_embedding_bm25(self, bm25_type: Optional[str] = 'plus') -> NoReturn:
        """Create or import embeddings"""

        pickle_name = f"sparse_embedding_bm25_{self.file_suffix}.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.BM25 = pickle.load(file)
            print("BM25 Embedding pickle load.")

        else:
            print("Build passage BM25 embedding")
            if tokenizer == None:
                tokenized = [i.split() for i in self.contexts]
            else:
                tokenized = [self.tokenizer(i) for i in self.contexts]
            
            if bm25_type == 'plus':
                self.BM25 = BM25Plus(tokenized)
            elif bm25_type == 'okapi':
                self.BM25 = BM25Okapi(tokenized)
            elif bm25_type == 'l':
                self.BM25 = BM25L(tokenized)
            else:
                raise ValueError('Plug in a proper bm25 type')  
            

            with open(emd_path, "wb") as file:
                pickle.dump(self.BM25, file)
            print("BM25 Embedding pickle saved.")


    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """ Retrieve passages from corresponding queries """

        assert self.BM25 is not None, "Run get_sparse_embedding_BM25() first"

        if isinstance(query_or_dataset, Dataset):
            # Return a pd.DataFrame for the retrieved passages
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_BM25(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval(BM25): ")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx],
                    "context": "<SEP>".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_bulk_BM25(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """Build or load pickle file for the top-k passages"""

        # Set pickle file directory
        score_path = os.path.join(self.data_path, f"BM25_score_{self.file_suffix}.bin")      
        indice_path = os.path.join(self.data_path, f"BM25_indice_{self.file_suffix}.bin")

        # Load pickle if exists
        if os.path.isfile(score_path) and os.path.isfile(indice_path):
            with open(score_path, "rb") as file:
                doc_scores = pickle.load(file)  
            with open(indice_path, "rb") as file:
                doc_indices= pickle.load(file)            
            print("BM25 pickle load.")

        # Build pickle file if non-existing
        else:
            print("Build BM25 pickle")
            if tokenizer == None:
                tokenized_queries = [i.split() for i in queries]
            else:
                tokenized_queries = [self.tokenizer(i) for i in queries]      
            doc_scores = []
            doc_indices = []

            # Append score, indexes for top k passages
            for i in tqdm(tokenized_queries):
                scores = self.BM25.get_scores(i)

                sorted_score = np.sort(scores)[::-1]
                sorted_id = np.argsort(scores)[::-1]

                doc_scores.append(sorted_score[:k])
                doc_indices.append(sorted_id[:k])

            # Pickle dump
            with open(score_path, "wb") as file:
                pickle.dump(doc_scores, file)
            with open(indice_path, "wb") as file:
                pickle.dump(doc_indices, file)
            print("BM25 pickle saved.")        

        return doc_scores, doc_indices

#Sparse Retrieval based on TFIDF
class SparseRetrieval_TFIDF:
    def __init__(self, tokenize_fn,data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        file_suffix: Optional[str] = '',
        max_features: Optional[int] = None,
        ) -> NoReturn:

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()])) # for order invariance

        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=max_features,
        )

        self.p_embedding = None 
        self.file_suffix = file_suffix

    def get_sparse_embedding(self) -> NoReturn:
        """Create or import embeddings"""

        pickle_name = f"sparse_embedding{self.file_suffix}.bin"
        tfidfv_name = f"tfidv{self.file_suffix}.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file, protocol = -1)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file, protocol = -1)
            print("Embedding pickle saved.")


    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """ Retrieve passages from corresponding queries """

        assert self.p_embedding is not None, "Run get_sparse_embedding() first"


        if isinstance(query_or_dataset, Dataset):
            # Return a pd.DataFrame for the retrieved passages
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx],
                    "context": "<SEP>".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            # cqas.to_csv('example.csv')
            return cqas


    def get_relevant_doc_bulk( self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """Build or load pickle file for the top-k passages"""

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "Error: query consists only of OOV"

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices

#Dense Retrieval
class DenseRetrieval(Dense):
    def __init__(self, **kwargs):
        super(DenseRetrieval, self).__init__(**kwargs)

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, long_sum: Optional[bool] = False
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """ Retrieve passages from corresponding queries """

        if isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_dpr(
                    query_or_dataset["question"], k=topk)
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense Retriever: ")
            ):
                
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx],
                    "context": "<SEP>".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # if validation set
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                
                total.append(tmp)
            cqas = pd.DataFrame(total)
            # cqas.to_csv('dpr_example.csv') # if neccessary
            return cqas 

    def get_relevant_doc_bulk_dpr(
            self, queries, k= 1, args=None, p_encoder=None, q_encoder=None, long_sum = False
        ):
            """Build or load pickle file for the top-k passages"""
            if args is None:
                args = self.args
            if p_encoder is None:
                p_encoder = self.p_encoder
            if q_encoder is None:
                q_encoder = self.q_encoder
            batch_size = args.per_device_eval_batch_size
            
            p_encoder.to('cuda')
            q_encoder.to('cuda')

            # passage dataloader
            p_tokenized = self.tokenizer(
                        self.contexts,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
            p_dataset = TensorDataset(p_tokenized['input_ids'], p_tokenized['attention_mask'], p_tokenized['token_type_ids'])
            passage_dataloader = DataLoader(p_dataset,batch_size = batch_size)

            q_tokenized = self.tokenizer(
                        queries,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
            q_dataset = TensorDataset(q_tokenized['input_ids'], q_tokenized['attention_mask'], q_tokenized['token_type_ids'])
            query_dataloader = DataLoader(q_dataset,batch_size = batch_size)

            doc_scores = []
            doc_indices = []

            with torch.no_grad():
                self.p_encoder.eval()
                self.q_encoder.eval()

                p_embs = []
                for batch in tqdm(passage_dataloader):
                    batch = tuple(t.to(args.device) for t in batch)

                    if long_sum:
                        batch_new, split_info = split_pad(batch, batch_size, max_len = 512)
                        
                        p_inputs = {
                            "input_ids": batch_new[0].to(args.device),
                            "attention_mask": batch_new[1].to(args.device),
                            "token_type_ids": batch_new[2].to(args.device)
                            }
                        p_outputs = self.p_encoder(**p_inputs).to("cpu")
                        p_outputs = reconstruct_embedding(p_outputs, split_info, (batch_size,768))

                    # not dpr_base, dpr_neg
                    else:
                        p_inputs = {
                            "input_ids": batch[0],
                            "attention_mask": batch[1],
                            "token_type_ids": batch[2]
                            }
                        p_outputs = self.p_encoder(**p_inputs).to("cpu")

                    p_embs.append(p_outputs)

                p_embs = torch.stack(p_embs, dim=0).view(len(passage_dataloader.dataset), -1)

                q_embs = []
                for batch in tqdm(query_dataloader):
                    batch = tuple(t.to(args.device) for t in batch)
                    q_inputs = {
                            "input_ids": batch[0],
                            "attention_mask": batch[1],
                            "token_type_ids": batch[2]
                    }
                    q_outputs = self.q_encoder(**q_inputs).to("cpu")
                    q_embs.append(q_outputs)
                q_embs = torch.stack(q_embs, dim=0).view(len(query_dataloader.dataset), -1)


            # Pickle-dump DPR embeddings
            with open(f"../data/dense_embs/dense_pemb_{retriever_args.file_suffix}.bin", "wb") as file:
                pickle.dump(p_embs, file, protocol = -1)
                print("p_embs saved")

            with open(f"../data/dense_embs/dense_qemb_{retriever_args.file_suffix}.bin", "wb") as file:
                pickle.dump(q_embs, file, protocol = -1)
                print("q_embs saved")
            
            dot_prod = torch.matmul(q_embs,torch.transpose(p_embs,0,1))
            doc_scores, doc_indices = torch.sort(dot_prod, dim = 1, descending = True)

            return doc_scores[:,:k], doc_indices[:,:k]

# measuring topk retrieval performance
if __name__ == "__main__":

    parser = HfArgumentParser((RetrieverArguments))
    retriever_args,= parser.parse_args_into_dataclasses()

    # Test Set
    test_set = load_from_disk("../data/test/")
    print("*" * 40, "query dataset", "*" * 40)

    # Dense Retrieval
    if retriever_args.retriever_type == "DenseRetrieval":
        args, tokenizer, p_enc, q_enc = get_dense_args(retriever_args)
    
        tokenizer = AutoTokenizer.from_pretrained(retriever_args.dpr_model)
        retriever = DenseRetrieval(args=args,dataset=test_set, 
                            tokenizer=tokenizer,p_encoder=p_enc,q_encoder=q_enc)
    
        df =  retriever.retrieve(test_set,topk = 10, long_sum = retriever_args.dpr_longsum)
    
    # Sparse Retrieval
    else:
        if retriever_args.spr_tokenizer == 'none':
            tokenizer = None
        elif retriever_args.spr_tokenizer == 'klue':
            tokenizer = AutoTokenizer.from_pretrained('klue/bert-base').tokenize
        elif retriever_args.spr_tokenizer == 'bigbird':
            tokenizer = AutoTokenizer.from_pretrained('monologg/kobigbird-bert-base').tokenize
        elif retriever_args.spr_tokenizer == 'kobert_m':
            tokenizer = AutoTokenizer.from_pretrained('monologg/kobert').tokenize
        elif retriever_args.spr_tokenizer == 'kobert_s':
            tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1').tokenize
        elif retriever_args.spr_tokenizer == 'bert_multi':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased').tokenize
        else:
            raise ValueError('Plug in a tokenizer of choice; refer to arguments.py')
        
        if retriever_args.retriever_type == "SparseRetrieval_TFIDF":
            retriever = SparseRetrieval_TFIDF(tokenize_fn=tokenizer,
                                    file_suffix = retriever_args.file_suffix,
                                    max_features = retriever_args.max_features) 
            retriever.get_sparse_embedding()

        elif retriever_args.retriever_type == "SparseRetrieval_BM25":
            retriever = SparseRetrieval_BM25(tokenize_fn=tokenizer, file_suffix = retriever_args.file_suffix) 
            retriever.get_sparse_embedding_bm25(bm25_type = retriever_args.bm25_type)
        
        else:
            raise ValueError('Plug in a correct retriever type; refer to arguments.py')

        df =  retriever.retrieve(test_set, topk = 10)

    # Print Retriever Info & Performance
    print('#'*30)
    print(f'Retriever Info: {retriever_args.retriever_type}_{retriever_args.file_suffix}')
    print(retriever_acc_k([1,3,5,10], df))
    print('#'*30)

