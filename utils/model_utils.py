# -*- coding:utf-8 -*-
import logging
import os
import json
import torch.nn.functional as F
import torch
import numpy as np
import math

logger = logging.getLogger(__name__)

def soft_frequency(logits, power=2, probs=False):
    """
    Unsupervised Deep Embedding for Clustering Analysiszaodian
    https://arxiv.org/abs/1511.06335
    """
    if not probs:
        softmax = torch.nn.Softmax(dim=1)
        y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
    else:
        y = logits
    f = torch.sum(y, dim=(0, 1))
    t = y**power / f
    p = t/torch.sum(t, dim=2, keepdim=True)
    # m = torch.argmax(y, dim=2, keepdim=True)
    # m = (m==0)
    # m = m.repeat(1,1,y.size(2))
    # p = p.masked_fill(mask=m,value=torch.tensor(0))
    # m = ~m
    # y = y.masked_fill(mask=m,value=torch.tensor(0))
    # p = p+y

    return p

def get_hard_label(args, combined_labels, pred_labels, pad_token_label_id, pred_logits=None):
    pred_labels[combined_labels==pad_token_label_id] = pad_token_label_id

    return pred_labels, None

def mask_tokens(args, combined_labels, pred_labels, pad_token_label_id, pred_logits=None):

    if args.self_learning_label_mode == "hard":
        softmax = torch.nn.Softmax(dim=1)
        y = softmax(pred_logits.view(-1, pred_logits.shape[-1])).view(pred_logits.shape)
        _threshold = args.threshold
        pred_labels[y.max(dim=-1)[0]>_threshold] = pad_token_label_id
        # if args.self_training_hp_label < 5:
        #     pred_labels[combined_labels==pad_token_label_id] = pad_token_label_id
        # pred_labels[combined_labels==pad_token_label_id] = pad_token_label_id
        return pred_labels, None

    elif args.self_learning_label_mode == "soft":
        label_mask = (pred_labels.max(dim=-1)[0]>args.threshold)
        # label_mask = None

        
        return pred_labels, label_mask

def opt_grad(loss, in_var, optimizer):
    
    if hasattr(optimizer, 'scalar'):
        loss = loss * optimizer.scaler.loss_scale
    return torch.autograd.grad(loss, in_var)

def _update_mean_model_variables_Bond(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    m = get_param_momentum(param_momentum,global_step,t_total)
    for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
        p2.data = p1.detach().data

def _update_mean_model_variables_EMA(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for m_param, param in zip(teach_model.parameters(), stu_model.parameters()):
        m_param.data.mul_(alpha).add_(1 - alpha, param.data) 

def _update_mean_model_variables_SE(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    m = get_param_momentum(param_momentum,global_step,t_total)
    for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
        tmp_prob = np.random.rand()
        if tmp_prob < 0.8:
            pass
        else:
            p2.data = p1.detach().data

def _update_mean_model_variables_STS(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    m = get_param_momentum(param_momentum,global_step,t_total)
    for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
        tmp_prob = np.random.rand()
        if tmp_prob < 0.8:
            pass
        else:
            p2.data = m * p2.data + (1.0 - m) * p1.detach().data

def _update_mean_model_variables(model, m_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for m_param, param in zip(m_model.parameters(), model.parameters()):
        m_param.data.mul_(alpha).add_(1 - alpha, param.data)   
    # for p1, p2 in zip(model.parameters(), m_model.parameters()):    
    #     tmp_prob = np.random.rand()
    #     if tmp_prob < 0.7:
    #         pass
    #     else:
    #         p2.data = alpha * p2.data + (1.0 - alpha) * p1.detach().data


def _update_mean_model_variables_v2(stu_model, teach_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
        tmp_prob = np.random.rand()
        if tmp_prob < 0.7:
            pass
        else:
            p2.data = 0.99 * p2.data + (1.0 - 0.99) * p1.detach().data


def _update_mean_model_variables_v3(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    m = get_param_momentum(param_momentum,global_step,t_total)
    # time = 0
    for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
        tmp_prob = np.random.rand()
        # print(p1.shape)
        # time += 1
        if tmp_prob < 0.8:
            pass
        else:
            p2.data = m * p2.data + (1.0 - m) * p1.detach().data
    # print(time)
def get_param_momentum(param_momentum,current_train_iter,total_iters):

    return 1.0 - (1.0 - param_momentum) * (
        (math.cos(math.pi * current_train_iter / total_iters) + 1) * 0.5
    )

def _update_mean_prediction_variables(prediction, m_prediction, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # for m_param, param in zip(m_model.parameters(), model.parameters()):
    m_prediction.data.mul_(alpha).add_(1 - alpha, prediction.data)

def _update_mean_model_variables_v4(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    m = get_param_momentum(param_momentum,global_step,t_total)
    layer = 1
    temp1 = []
    temp2 = []
    start = 182
    end = 197
    for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
        if start <= layer <= end:
            temp1.append(p1)
            temp2.append(p2)
            if layer == end :
                sts(temp1,temp2,m)
                temp1 = []
                temp2 = []
        # elif layer <= 21:
        #     temp1.append(p1)
        #     temp2.append(p2)
        #     if layer == 21 :
        #         sts(temp1,temp2,m)
        #         temp1 = []
        #         temp2 = []
        # elif layer <= 37:
        #     temp1.append(p1)
        #     temp2.append(p2)
        #     if layer == 37 :
        #         sts(temp1,temp2,m)
        #         temp1 = []
        #         temp2 = []
        # elif layer <= 53:
        #     temp1.append(p1)
        #     temp2.append(p2)
        #     if layer == 53 :
        #         sts(temp1,temp2,m)
        #         temp1 = []
        #         temp2 = []
        # elif layer <= 69:
        #     temp1.append(p1)
        #     temp2.append(p2)
        #     if layer == 69 :
        #         sts(temp1,temp2,m)
        #         temp1 = []
        #         temp2 = []
        # elif layer <= 85:
        #     temp1.append(p1)
        #     temp2.append(p2)
        #     if layer == 85 :
        #         sts(temp1,temp2,m)
        #         temp1 = []
        #         temp2 = []
        # elif layer <= 101:
        #     temp1.append(p1)
        #     temp2.append(p2)
        #     if layer == 101 :
        #         sts(temp1,temp2,m)
        #         temp1 = []
        #         temp2 = []
        # elif layer <= 117:
        #     temp1.append(p1)
        #     temp2.append(p2)
        #     if layer == 117 :
        #         sts(temp1,temp2,m)
        #         temp1 = []
        #         temp2 = []
        # elif layer <= 133:
        #     temp1.append(p1)
        #     temp2.append(p2)
        #     if layer == 133 :
        #         sts(temp1,temp2,m)
        #         temp1 = []
        #         temp2 = []      
        # elif layer <= 149:
        #     temp1.append(p1)
        #     temp2.append(p2)
        #     if layer == 149 :
        #         sts(temp1,temp2,m)
        #         temp1 = []
        #         temp2 = []
        # elif layer <= 165:
        #     temp1.append(p1)
        #     temp2.append(p2)
        #     if layer == 165 :
        #         sts(temp1,temp2,m)
        #         temp1 = []
        #         temp2 = []
        # elif layer <= 181:
        #     temp1.append(p1)
        #     temp2.append(p2)
        #     if layer == 181 :
        #         sts(temp1,temp2,m)
        #         temp1 = []
        #         temp2 = []
        # elif layer <= 197:
        #     temp1.append(p1)
        #     temp2.append(p2)
        #     if layer == 197 :
        #         sts(temp1,temp2,m)
        #         temp1 = []
        #         temp2 = []
        else :
            # temp1.append(p1)
            # temp2.append(p2)
            # if layer == 201 :
            #     sts(temp1,temp2,m)
            p2.data.mul_(alpha).add_(1 - alpha, p1.data)
        layer += 1
def sts(temp1,temp2,m):
    tmp_prob = np.random.rand()
    if tmp_prob < 0.8:
        pass
    else:
        for tmp1, tmp2 in zip(temp1,temp2):
            tmp2.data = m * tmp2.data + (1.0 - m) * tmp1.detach().data
