import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from pprint import pprint
import os
import wandb
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

from train_dpr import Dense, BertEncoder
from arguments import DefaultArguments, DatasetArguments, RetrieverArguments
from utils import get_dense_args, timer



class DenseTrainV2(Dense):
    def __init__(self, num_neg,**kwargs):
        super(DenseTrainV2, self).__init__(**kwargs)
        self.num_neg = num_neg
        self.prepare_in_batch_negative(num_neg = num_neg)
    
    def prepare_in_batch_negative(
        self,
        dataset = None,
        num_neg = None,
        tokenizer = None
    ):
        if dataset is None:
            dataset = self.dataset
        if tokenizer is None:
            tokenizer = self.tokenizer
        if num_neg is None:
            num_neg = self.num_neg
        
        corpus = np.array(list(set([example for example in dataset["context"]])))
        p_with_neg = []
        
        # batchify with negative sampling
        for c in dataset['context']:
            while True:
                neg_idxs = np.random.randint(len(corpus), size = num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        q_seqs = tokenizer(
            dataset['question'],
            padding = "max_length",
            truncation = True,
            return_tensors = 'pt'
        )
        p_seqs = tokenizer(
            p_with_neg,
            padding = "max_length",
            truncation = True,
            return_tensors = 'pt'
        )

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        train_sampler = RandomSampler(train_dataset)
        self.train_dataloader = DataLoader(
            train_dataset,
            sampler = train_sampler,
            drop_last = True,
            batch_size = self.args.per_device_train_batch_size
        )

        valid_seqs = tokenizer(
            self.contexts,
            padding = "max_length",
            truncation = True,
            return_tensors = 'pt'
        )

        passage_dataset = TensorDataset(
            valid_seqs['input_ids'],
            valid_seqs['attention_mask'],
            valid_seqs['token_type_ids']
        )

        self.passage_dataloader = DataLoader(
            passage_dataset,
            batch_size = self.args.per_device_train_batch_size
        )

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
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )
        self.p_encoder.train()
        self.q_encoder.train()

        # Train Start
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")

        for _ in train_iterator:
            train_loss, train_acc, cnt = 0.0,0.0,0.0
            epoch_iterator = tqdm(self.train_dataloader, desc="Iteration")

            with epoch_iterator as tepoch:
                for batch in tepoch:
                    self.p_encoder.train()
                    self.q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() 
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }

                   
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs) # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = torch.transpose(p_outputs.view(batch_size, self.num_neg+1,-1), 1, 2)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)
                    _, preds = torch.max(sim_scores, 1)

                    loss = F.nll_loss(sim_scores, targets)
                    acc = torch.sum(preds.cpu() == targets.cpu())/batch_size
                    # print(loss)
                    # train_loss += loss

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

                        print(f'Step{global_step}- Train Loss: {train_loss:.2f} Train Acc: {train_acc:.2f}')
                        wandb.log({
                        'learning_rate' : lr_,
                        'steps': global_step,
                        'train/acc': train_acc,
                        'train/loss': train_loss,
                        })
                        train_loss, train_acc, cnt = 0.0,0.0,0.0
                    torch.cuda.empty_cache()

                    # del p_inputs, q_inputs
        return self.p_encoder, self.q_encoder


def main():

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

    retriever = DenseTrainV2(
        args=args,
        dataset=datasets,
        num_neg=retriever_args.num_neg,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder
    )
    p_encoder, q_encoder = retriever.train()

    model_dir =  f'./models/retriever_{args.run_name}'

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    p_encoder.save_pretrained(os.path.join(model_dir,'p_encoder'))
    q_encoder.save_pretrained(os.path.join(model_dir,'q_encoder'))

    print(f'passage & question encoders successfully saved at {model_dir}')


if __name__ == '__main__':
    main()
