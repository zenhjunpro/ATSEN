import argparse
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
import pickle as pkl
from apex import amp

import torch.optim as optim

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from model_test import RobertaForTokenClassification_Modified
from utils.data_utils import load_and_cache_examples, get_labels
from utils.model_utils import mask_tokens, soft_frequency, opt_grad, get_hard_label, _update_mean_model_variables, _update_mean_model_variables_v2,_update_mean_model_variables_v3,_update_mean_model_variables_v4
from utils.eval import evaluate
from utils.config import config
from utils.loss_utils import NegEntropy

from utils.model_utils import _update_mean_model_variables_Bond, _update_mean_model_variables_EMA, _update_mean_model_variables_SE, _update_mean_model_variables_STS

from models.m2_teacher import M2Teacher

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# def _update_mean_model_variables_Bond(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
#     alpha = min(1 - 1 / (global_step + 1), alpha)
#     m = get_param_momentum(param_momentum,global_step,t_total)
#     for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
#         p2.data = p1.detach().data

# def _update_mean_model_variables_EMA(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
#     alpha = min(1 - 1 / (global_step + 1), alpha)
#     for m_param, param in zip(teach_model.parameters(), stu_model.parameters()):
#         m_param.data.mul_(alpha).add_(1 - alpha, param.data) 

# def _update_mean_model_variables_SE(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
#     alpha = min(1 - 1 / (global_step + 1), alpha)
#     m = get_param_momentum(param_momentum,global_step,t_total)
#     for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
#         tmp_prob = np.random.rand()
#         if tmp_prob < 0.8:
#             pass
#         else:
#             p2.data = p1.detach().data

# def _update_mean_model_variables_STS(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
#     alpha = min(1 - 1 / (global_step + 1), alpha)
#     m = get_param_momentum(param_momentum,global_step,t_total)
#     for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
#         tmp_prob = np.random.rand()
#         if tmp_prob < 0.8:
#             pass
#         else:
#             p2.data = m * p2.data + (1.0 - m) * p1.detach().data


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y
def plot_sigmoid():
    x = np.arange(-8,8,0.5)
    y = sigmoid(x)
    plt.plot(x,y)
    plt.show()




# -*- coding:utf-8 -*-
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.data_utils import load_and_cache_examples, tag_to_id, get_chunks
from flashtool import Logger
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
# )
# logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
# logging_fh.setLevel(logging.DEBUG)
# logger.addHandler(logging_fh)
# logger.warning(
#     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
#     args.local_rank,
#     device,
#     args.n_gpu,
#     bool(args.local_rank != -1),
#     args.fp16,
# )
def validation(args, model, tokenizer, labels, pad_token_label_id, best_dev, best_test, 
                  global_step, t_total, epoch, tors):
    
    model_type = MODEL_NAMES[tors].lower()

    results, _, best_dev, is_updated1 = evaluate(args, model, tokenizer, labels, pad_token_label_id, best_dev, mode="dev", \
        logger=logger, prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)

    results, _, best_test, is_updated2 = evaluate(args, model, tokenizer, labels, pad_token_label_id, best_test, mode="test", \
        logger=logger, prefix='test [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
   
    # output_dirs = []
    if args.local_rank in [-1, 0] and is_updated1:
        # updated_self_training_teacher = True
        path = os.path.join(args.output_dir+tors, "checkpoint-best-1")
        logger.info("Saving model checkpoint to %s", path)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = (
                model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(path)
        tokenizer.save_pretrained(path)
    # output_dirs = []
    if args.local_rank in [-1, 0] and is_updated2:
        # updated_self_training_teacher = True
        path = os.path.join(args.output_dir+tors, "checkpoint-best-2")
        logger.info("Saving model checkpoint to %s", path)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = (
                model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(path)
        tokenizer.save_pretrained(path)

    return best_dev, best_test, is_updated1

def evaluate(args, model, tokenizer, labels, pad_token_label_id, best, mode, logger, prefix="", verbose=True):
    
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation %s *****", prefix)
    if verbose:
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": {"pseudo": batch[3]}}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "mobilebert"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss_dict, logits = outputs[:2]
            tmp_eval_loss = tmp_eval_loss_dict["pseudo"]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"]["pseudo"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"]["pseudo"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    # print(preds)
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    out_id_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_id_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                preds_list[i].append(label_map[preds[i][j]])
                out_id_list[i].append(out_label_ids[i][j])
                preds_id_list[i].append(preds[i][j])

    correct_preds, total_correct, total_preds = 0., 0., 0. # i variables
    for ground_truth_id,predicted_id in zip(out_id_list,preds_id_list):
        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks      = set(get_chunks(ground_truth_id, tag_to_id(args.data_dir, args.dataset)))
        lab_pred_chunks = set(get_chunks(predicted_id, tag_to_id(args.data_dir, args.dataset)))

        # Updating the i variables
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds   += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p   = correct_preds / total_preds if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    new_F  = 2 * p * r / (p + r) if correct_preds > 0 else 0

    is_updated = False
    if new_F > best[-1]:
        best = [p, r, new_F]
        is_updated = True

    results = {
       "loss": eval_loss,
       "precision": p,
       "recall": r,
       "f1": new_F,
       "best_precision": best[0],
       "best_recall":best[1],
       "best_f1": best[-1]
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list, best, is_updated


if __name__ == "__main__":
    plot_sigmoid()































# def _update_mean_model_variables_v4(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
#     alpha = min(1 - 1 / (global_step + 1), alpha)
#     m = get_param_momentum(param_momentum,global_step,t_total)
#     layer = 1
#     temp1 = []
#     temp2 = []
#     for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
#         if layer <= 5:
#             temp1.append(p1)
#             temp2.append(p2)
#             if layer == 5 :
#                 sts(temp1,temp2)
#                 temp1 = []
#                 temp2 = []
#         elif layer <= 21:
#             temp1.append(p1)
#             temp2.append(p2)
#             if layer == 21 :
#                 sts(temp1,temp2)
#                 temp1 = []
#                 temp2 = []
#         elif layer <= 37:
#             temp1.append(p1)
#             temp2.append(p2)
#                 sts(temp1,temp2)
#                 temp1 = []
#                 temp2 = []
#         elif layer <= 53:
#             temp1.append(p1)
#             temp2.append(p2)
#             if layer == 53 :
#                 sts(temp1,temp2)
#                 temp1 = []
#                 temp2 = []
#         elif layer <= 69:
#             temp1.append(p1)
#             temp2.append(p2)
#             if layer == 69 :
#                 sts(temp1,temp2)
#                 temp1 = []
#                 temp2 = []
#         elif layer <= 85:
#             temp1.append(p1)
#             temp2.append(p2)
#             if layer == 85 :
#                 sts(temp1,temp2)
#                 temp1 = []
#                 temp2 = []
#         elif layer <= 101:
#             temp1.append(p1)
#             temp2.append(p2)
#             if layer == 101 :
#                 sts(temp1,temp2)
#                 temp1 = []
#                 temp2 = []
#         elif layer <= 117:
#             temp1.append(p1)
#             temp2.append(p2)
#             if layer == 117 :
#                 sts(temp1,temp2)
#                 temp1 = []
#                 temp2 = []
#         elif layer <= 133:
#             temp1.append(p1)
#             temp2.append(p2)
#             if layer == 133 :
#                 sts(temp1,temp2)
#                 temp1 = []
#                 temp2 = []      
#         elif layer <= 149:
#             temp1.append(p1)
#             temp2.append(p2)
#             if layer == 149 :
#                 sts(temp1,temp2)
#                 temp1 = []
#                 temp2 = []
#         elif layer <= 165:
#             temp1.append(p1)
#             temp2.append(p2)
#             if layer == 165 :
#                 sts(temp1,temp2)
#                 temp1 = []
#                 temp2 = []
#         elif layer <= 181:
#             temp1.append(p1)
#             temp2.append(p2)
#             if layer == 181 :
#                 sts(temp1,temp2)
#                 temp1 = []
#                 temp2 = []
#         elif layer <= 197:
#             temp1.append(p1)
#             temp2.append(p2)
#             if layer == 197 :
#                 sts(temp1,temp2)
#                 temp1 = []
#                 temp2 = []
#         else :
#             temp1.append(p1)
#             temp2.append(p2)
#             if layer == 201 :
#                 sts(temp1,temp2)
#         layer += 1
# def sts(temp1,temp2):
#     tmp_prob = np.random.rand()
#     if tmp_prob < 0.8:
#         pass
#     else:
#         for tmp1, tmp2 in zip(temp1,temp2)
#             temp2.data = m * temp2.data + (1.0 - m) * tmp1.detach().data
        