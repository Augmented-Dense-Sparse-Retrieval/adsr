import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from typing import Any, Union, Dict, List, Optional
from contextlib import contextmanager
import time
import os

import torch
import torch.nn.functional as F

from datasets import (
    load_metric,
    load_from_disk,
    Sequence,
    Value,
    Features,
    Dataset,
    DatasetDict,
)

from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,HfArgumentParser
)

from arguments import DatasetArguments, RetrieverArguments, DefaultArguments
from train_dpr import Dense, BertEncoder

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


def split_batch(batch, batch_size = 8, max_len = 512):
    """ newly batchify the padded sequence of 1536 into valid batches """
    split_info = {}
    batch_new = [[],[],[]]
    for batch_ind in range(batch_size):
        if sum(batch[1][batch_ind]) <= max_len:
            split_info[batch_ind] = (batch_ind,1)
            for embed_type in range(3): # 3 types
                batch_new[embed_type].append(batch[embed_type][:max_len+1])

        elif sum(batch[1][batch_ind]) <= max_len*2:
            split_info[batch_ind] = (batch_ind,2) # batch_index, split_size
            for embed_type in range(3): # 3 types
                batch_new[embed_type].append(batch[embed_type][:max_len*2+1])
        
        else:
            split_info[batch_ind] = (batch_ind,3) # batch_index, split_size
            for embed_type in range(3): # 3 types
                batch_new[embed_type].append(batch[embed_type])
    return batch_new, split_info


def reconstruct_embedding(outputs, split_info, original_dim):
    """ reconstruct embedding by summing the same passage embeddings """
    embeddings_flattened = []
    ind = 0
    for batch_ind, split_len in split_info.values(): # ex> 24*786
        output_mean = torch.mean(outputs[batch_ind+ind:batch_ind+ind+split_len], dim = 0)
        embeddings_flattened.append(output_mean)
        ind += split_len-1
    embeddings_flattened = torch.stack(embeddings_flattened, dim = 0)
    reconstructed_outputs = embeddings_flattened.view(original_dim[0], original_dim[1]) 

    return reconstructed_outputs # ex> 8*786

def retriever_acc_k(topk_list, retrieved_df):
    """ retriever accuracy from the top-k retrieved passages """
    result_dict = {}
    count = [0]*len(topk_list)

        # iterate through each passage & query pair
    for ind, _ in enumerate(range(len(retrieved_df))):
        contexts = retrieved_df['context'][ind].split('<SEP>')
        gold_answer = retrieved_df['original_context'][ind]
        for order, k in enumerate(topk_list):
            if gold_answer in contexts[:k]: 
                count[order] += 1

        # compute precision at each k
    for ind, k in enumerate(topk_list):        
        result_dict[f'P@{k}'] = f'{round(count[ind]/len(retrieved_df)*100,1)}%'

    return result_dict


def get_dense_args(retriever_args:RetrieverArguments):
    """ Comprehensive arguments retrieval for important training.args & model/tokenizer """
    args = TrainingArguments(
            output_dir="dense_retrieval",
            evaluation_strategy="epoch",
            learning_rate=retriever_args.dpr_learning_rate,
            per_device_train_batch_size=retriever_args.dpr_train_batch,
            per_device_eval_batch_size=retriever_args.dpr_eval_batch,
            num_train_epochs=retriever_args.dpr_epochs,
            weight_decay=retriever_args.dpr_weight_decay,
            overwrite_output_dir = True,
            eval_steps = retriever_args.dpr_eval_steps,
            warmup_steps = retriever_args.dpr_warmup_steps,
            run_name = retriever_args.run_name
            )

    retriever_dir = retriever_args.retriever_dir
    p,q = 'p_encoder','q_encoder'

    if (os.path.isdir(os.path.join(retriever_dir,p)) and os.path.isdir(os.path.join(retriever_dir,q))):
        print('Fine-tuned DPR exists... check directory again if using model_checkpoints...')
        config_p =  AutoConfig.from_pretrained(os.path.join(retriever_dir, p))
        config_q =  AutoConfig.from_pretrained(os.path.join(retriever_dir, q))
        p_encoder  = BertEncoder.from_pretrained(os.path.join(retriever_dir, p), config = config_p)
        q_encoder = BertEncoder.from_pretrained(os.path.join(retriever_dir, q), config = config_q)

    else:
        p_encoder  = BertEncoder.from_pretrained(retriever_args.dpr_model)
        q_encoder = BertEncoder.from_pretrained(retriever_args.dpr_model)
        print('No fine-tuned DPR exists ... newly train Dense Passage Retriever...')
    
    tokenizer = AutoTokenizer.from_pretrained(retriever_args.dpr_model)
    
    return args, tokenizer, p_encoder, q_encoder