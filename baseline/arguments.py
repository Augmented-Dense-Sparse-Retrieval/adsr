from dataclasses import asdict, dataclass, field
from typing import Any, Union, Dict, List, Optional
from enum import Enum
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer

from transformers import TrainingArguments
from simple_parsing.helpers import Serializable

@dataclass
class BaseArguments(Serializable):
    def __str__(self):
        self_as_dict = asdict(self)
        self_as_dict = {k: f"<{k.upper()}>" for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

@dataclass
class DatasetArguments(BaseArguments):
    """Dataset/DataLoader Arguments"""

    dataset_path: str = "../data"
    """path for the dataset"""

@dataclass
class RetrieverArguments(BaseArguments):
    """Retriever Arguments"""

    retriever_type: str = "SparseRetrieval_BM25P"
    """
    - SparseRetrieval_BM25P
    - SparseRetrieval_TFIDF
    - DenseRetrieval
    
    """

    bm25_type: str = "plus"
    """
    - plus
    - okapi
    - l
    """

    max_features: int = None
    """ maximum features for TFIDF Vectorizer """


    file_suffix: str = ''
    """ file name fill-in  """

    spr_tokenizer: str = None
    """ 
    Tokenizer setting for sparse retriever 
    - klue
    - bigbird
    - kobert   (monologg)
    -> others will RaiseError
    
    """

    run_name: str = 'exp' 
    """train run name for wandb"""

    retriever_dir: str = './models/retriever' 
    """directory of retriever"""

    top_k_retrieval: int = 1
    """numb top-k passages to retrieve"""

    dpr_model: str = "monologg/kobigbird-bert-base"
    """path to pretrained model or model identifier from huggingface.co/models"""    

    dpr_learning_rate: float = 3e-5
    """learning rate for DPR fine-tuning"""

    dpr_train_batch: int = 1
    """train batch size for DPR fine-tuning"""

    dpr_eval_batch: int = 1
    """eval batch size for DPR fine-tuning"""

    dpr_epochs: int = 3
    """numb of epochs for DPR fine-tuning"""

    dpr_weight_decay: float = 0.01
    """weight decay for DPR fine-tuning"""

    dpr_eval_steps: int = 200
    """numb of epochs for DPR fine-tuning"""

    dpr_warmup_steps: int = 500
    """numb of warmup steps for DPR fine-tuning"""


@dataclass
class DefaultArguments(BaseArguments):
    """Default Arguments"""

    description: str = ""
    """brief description of the experiment"""

    wandb_entity: str = "cose461-team22"
    """wandb entity name"""

    wandb_project: str = "dpr"
    """wandb project name"""