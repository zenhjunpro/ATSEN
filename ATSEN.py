# -*- coding:utf-8 -*-
from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss
from utils.loss_utils import GCELoss, DistillKL
from optimization import find_optimal_svm


import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torch.optim as optim

# from loss_utils import FocalLoss, SoftFocalLoss

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}

class RobertaForTokenClassification_Modified(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        # self.focalloss = FocalLoss(gamma=2)
        # self.softfocalloss = SoftFocalLoss(gamma=2)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        module_list=None,
        optimizer=None,
        args=None,
        TEST=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        final_embedding = outputs[0]
        sequence_output = self.dropout(final_embedding)
        logits = self.classifier(sequence_output)

        outputs = (logits, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here

        loss_dict = {}

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if labels is not None:
            # logits = self.logsoftmax(logits)
            # Only keep active parts of the loss
            active_loss = True
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                # active_loss = True
                # if attention_mask is not None:
                #     active_loss = attention_mask.view(-1) == 1
                # if label_mask is not None:
                #     active_loss = active_loss & label_mask.view(-1)
                # active_logits = logits.view(-1, self.num_labels)[active_loss]
            
            for key in labels:
                label = labels[key]
                if label is None:
                    continue
                # if key=="pseudo" and label_mask is not None:
                if label_mask is not None:
                    all_active_loss = active_loss & label_mask.view(-1)
                    ####
                    all_deactive_loss = active_loss & (~label_mask).view(-1)
                    ####
                else:
                    all_active_loss = active_loss
                active_logits = logits.view(-1, self.num_labels)[all_active_loss]

                if label.shape == logits.shape:
                    loss_fct = KLDivLoss()
                    # loss_fct = SoftFocalLoss(gamma=2)
                    if attention_mask is not None or label_mask is not None:
                        active_labels = label.view(-1, self.num_labels)[all_active_loss]
                        loss_ce = loss_fct(active_logits, active_labels)
                    else:
                        loss_ce = loss_fct(logits, label)
                else:
                    loss_fct = CrossEntropyLoss()
                    # loss_fct = FocalLoss(gamma=2)
                    # loss_fct = NLLLoss()
                    # loss_fct = GCELoss()

                    if attention_mask is not None or label_mask is not None:
                        active_labels = label.view(-1)[all_active_loss]
                        loss_ce = loss_fct(active_logits, active_labels)
                        # loss_ce = loss_ce.to(args.device)
                    else:
                        loss_ce = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
                        # loss_ce = loss_ce.to(args.device)
                
                if module_list != None:
                    if len(module_list) == 2 and TEST:
                        deactive_logits = logits.view(-1, self.num_labels)[all_deactive_loss]
                        for module in module_list:
                            module.eval()
                        logit_t_list = []
                        with torch.no_grad():
                            for model_t in module_list:
                                outputs = model_t(**inputs)
                                logit_t = outputs[0].view(-1,self.num_labels)[all_active_loss]
                                logit_t_list.append(logit_t)    
                        criterion_kd = DistillKL(2)
                        logit_s = active_logits
                        loss_div_list = []
                        grads = []
                        logit_s.register_hook(lambda grad: grads.append(
                            Variable(grad.data.clone(), requires_grad=False)))
                        for logit_t in logit_t_list:
                            optimizer.zero_grad()
                            logit_t = logit_t.to(args.device)
                            loss_s = criterion_kd(logit_s, logit_t)
                            loss_s.backward(retain_graph=True)
                            loss_div_list.append(loss_s)
                        # logit_t_co = (logit_t_list[0] + logit_t_list[1]) / 2
                        # logit_t_co = logit_t_co.to(args.device)
                        # loss_s = criterion_kd(logit_s, logit_t_co)

                        nu = 1 / (0.8 / 4)
                        scale = find_optimal_svm(torch.stack(grads),
                                                nu=nu,
                                                gpu_id=0,
                                                is_norm=False)
                        losses_div_tensor = torch.stack(loss_div_list)
                        if torch.cuda.is_available():
                            scale = scale.to(args.device)
                            losses_div_tensor.to(args.device)
                        loss_div = torch.dot(scale, losses_div_tensor)
                        # loss_div = loss_s

                if module_list != None:
                    if len(module_list) == 2 and TEST:
                        loss_dict[key] = args.bate * loss_div
                    else:
                        loss_dict[key] = loss_ce
                else:
                    loss_dict[key] = loss_ce


            outputs = (loss_dict,) + outputs

        return outputs  # (loss dict), scores, final_embedding, (hidden_states), (attentions)
