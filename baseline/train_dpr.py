import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
import logging
import sys
import re
import os
import wandb
import logging
from typing import Any, Union, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
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
from utils import get_dense_args, timer


class Dense:
    """Base Dense Class"""
    def __init__(self, args, dataset,
        tokenizer, p_encoder, q_encoder,  data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json"
    ):
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        self.ids = list(range(len(self.contexts)))


class BertEncoder(BertPreTrainedModel):
    """ Customize BERT Output for Dense Embedding """
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self,input_ids, attention_mask=None,token_type_ids=None): 

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output


class DenseTrain(Dense):
    """DPR Custom Trainer and Dataloader"""
    def __init__(self, **kwargs):
        super(DenseTrain, self).__init__(**kwargs)
        self.prepare_dataloaders()
    
    # dataloader
    def prepare_dataloaders(self,dataset = None,tokenizer = None):
        if dataset is None:
            dataset = self.dataset
        if tokenizer is None:
            tokenizer = self.tokenizer

        # train split -> train, valid
        dataset = dataset.train_test_split(test_size=0.1)
        train_data = dataset['train']
        valid_data = dataset['test']

        q_seqs_t = tokenizer(train_data['question'],padding = "max_length",truncation = True,return_tensors = 'pt')
        p_seqs_t = tokenizer(train_data['context'],padding = "max_length",truncation = True,return_tensors = 'pt')

        train_dataset = TensorDataset(
            p_seqs_t['input_ids'], p_seqs_t['attention_mask'], p_seqs_t['token_type_ids'],
            q_seqs_t['input_ids'], q_seqs_t['attention_mask'], q_seqs_t['token_type_ids']
        )

        q_seqs_v = tokenizer(valid_data['question'],padding = "max_length",truncation = True,return_tensors = 'pt')
        p_seqs_v = tokenizer(valid_data['context'],padding = "max_length",truncation = True,return_tensors = 'pt')

        val_dataset = TensorDataset(
            p_seqs_v['input_ids'], p_seqs_v['attention_mask'], p_seqs_v['token_type_ids'],
            q_seqs_v['input_ids'], q_seqs_v['attention_mask'], q_seqs_v['token_type_ids']
        )

        train_sampler = RandomSampler(train_dataset)
        self.train_dataloader = DataLoader(train_dataset,sampler = train_sampler,
                                    num_workers = 0, drop_last = True,
                                    batch_size = self.args.per_device_train_batch_size)

        val_sampler = RandomSampler(val_dataset)
        self.val_dataloader = DataLoader(val_dataset,sampler = val_sampler,
                                    num_workers = 0, drop_last = True,
                                    batch_size = self.args.per_device_eval_batch_size)

    
    # customized trainer
    def train(self, args = None):
        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optim : AdamW
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
        t_total = len(self.train_dataloader) * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps= args.warmup_steps,
            num_training_steps=t_total
        )

        # Train Start
        global_step = 0.0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(self.train_dataloader, desc="Iteration")
            train_loss, train_acc, cnt = 0.0,0.0,0.0

            for step, batch in enumerate(epoch_iterator):
                    self.p_encoder.train()
                    self.q_encoder.train()
            
                    targets = torch.arange(0, batch_size).long()
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].to(args.device),
                        "attention_mask": batch[1].to(args.device),
                        "token_type_ids": batch[2].to(args.device)
                    }
            
                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }
                   
                    p_outputs = self.p_encoder(**p_inputs)
                    q_outputs = self.q_encoder(**q_inputs)

                    sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                    sim_scores = F.log_softmax(sim_scores, dim=1)
                    _, preds = torch.max(sim_scores, 1)

                    acc = torch.sum(preds.cpu() == targets.cpu())/batch_size
                    loss = F.nll_loss(sim_scores, targets)
                    # print(loss, acc)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    train_loss += loss
                    train_acc += acc
                    cnt += 1.0
                    global_step += 1.0
                    

                    if global_step % args.eval_steps == 0:
                        train_loss /= float(cnt)
                        train_acc /= float(cnt)
                        lr_ = scheduler.get_last_lr()[0]

                        val_loss, val_acc = self.validation()
                        print(f'Step{global_step}- Train Loss: {train_loss:.2f} Train Acc: {train_acc:.2f} Valid Loss:{val_loss:.2f} Valid Acc:{val_acc:.2f}')
                        wandb.log({
                        'learning_rate' : lr_,
                        'steps': global_step,
                        'train/acc': train_acc,
                        'eval/acc': val_acc,
                        'train/loss': train_loss,
                        'eval/loss': val_loss
                        })
                        train_loss, train_acc, cnt = 0.0,0.0,0.0
                    
                    torch.cuda.empty_cache()

        return self.p_encoder, self.q_encoder

    def validation(self, args = None):
        if args is None:
            args = self.args
        batch_size = args.per_device_eval_batch_size
    
        self.p_encoder.eval()
        self.q_encoder.eval()

        epoch_iterator = tqdm(self.val_dataloader, desc="Valid Iteration")
        
        loss_total, acc_total, cnt = 0,0,0
        for step, batch in enumerate(epoch_iterator):
            with torch.no_grad():

                targets = torch.arange(0, batch_size).long()
                targets = targets.to(args.device)

                p_inputs = {
                    "input_ids": batch[0].to(args.device),
                    "attention_mask": batch[1].to(args.device),
                    "token_type_ids": batch[2].to(args.device)
                }
            
                q_inputs = {
                    "input_ids": batch[3].to(args.device),
                    "attention_mask": batch[4].to(args.device),
                    "token_type_ids": batch[5].to(args.device)
                }

                p_outputs = self.p_encoder(**p_inputs)
                q_outputs = self.q_encoder(**q_inputs)

                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                sim_scores = F.log_softmax(sim_scores, dim=1)
                _, preds = torch.max(sim_scores, 1)

                acc = torch.sum(preds.cpu() == targets.cpu())/batch_size
                loss = F.nll_loss(sim_scores, targets)
                print(loss, acc)

                loss_total += loss
                acc_total += acc
                cnt +=1
                torch.cuda.empty_cache()

        val_loss = loss_total/float(cnt)
        val_acc = acc_total/float(cnt)
        return val_loss, val_acc
    

def main():
    logger = logging.getLogger(__name__)

    parser = HfArgumentParser((DefaultArguments, DatasetArguments, RetrieverArguments))
    default_args, data_args, retriever_args= parser.parse_args_into_dataclasses()

    print(f"data is from {data_args.dataset_path}")


    datasets = load_from_disk(os.path.join(data_args.dataset_path , 'train'))
    args, tokenizer, p_encoder, q_encoder = get_dense_args(retriever_args)
    p_encoder = p_encoder.to(args.device)
    q_encoder = q_encoder.to(args.device)

    wandb.login()
    wandb.init(
        project=default_args.wandb_project,
        entity=default_args.wandb_entity,
        name=args.run_name
    )

    retriever = DenseTrain(
        args=args,
        dataset=datasets,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder
    )

    p_encoder, q_encoder = retriever.train()

    model_dir = f'./models/retriever_{args.run_name}'

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    p_encoder.save_pretrained(os.path.join(model_dir,'p_encoder'))
    q_encoder.save_pretrained(os.path.join(model_dir,'q_encoder'))

    print(f'passage & question encoders successfully saved at {model_dir}')

if __name__ == '__main__':
    main()