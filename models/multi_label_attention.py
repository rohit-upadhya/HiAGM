#!/usr/bin/env python
# coding:utf-8

import torch
from torch import nn
from torch.nn import functional
from models.embedding_layer import EmbeddingLayer
from models.Graph_encoder.graph import GraphEncoder
import os
from transformers import BertModel
class HiAGMLA(nn.Module):
    def __init__(self, config, label_map, model_mode, graph_model, device):
        """
        Hierarchy-Aware Global Model : (Parallel) Multi-label attention Variant
        :param config: helper.configure, Configure Object
        :param label_map: helper.vocab.Vocab.v2i['label'] -> Dict{str:int}
        :param model_mode: 'TRAIN'， 'EVAL'
        :param graph_model: computational graph for graph model
        :param device: torch.device, config.train.device_setting.device
        """
        super(HiAGMLA, self).__init__()

        self.config = config
        self.device = device
        self.label_map = label_map
        self.pretrain_config = BertModel.from_pretrained('bert-base-uncased')
        self.label_embedding = EmbeddingLayer(
            vocab_map=self.label_map,
            embedding_dim=config.embedding.label.dimension,
            vocab_name='label',
            config=config,
            padding_index=None,
            pretrained_dir=None,
            model_mode=model_mode,
            initial_type=config.embedding.label.init_type
        )
        # model = GraphEncoder(config=self.pretrain_config.config, graph=False, layer=1,
        #                      data_path=os.getcwd() + "/data/rcv1",
        #                      #  threshold=0.01, tau=1,)
        self.graph_model = GraphEncoder(config=self.pretrain_config.config, graph=False, layer=1,
                              data_path=os.getcwd() + "/data/rcv1",
                                threshold=0.01, tau=1,)  #  replace model

        # classifier
        self.linear = nn.Linear(len(self.label_map) * config.embedding.label.dimension,
                                len(self.label_map))

        # dropout
        self.dropout = nn.Dropout(p=config.model.classifier.dropout)

        self.model_mode = model_mode

    def get_label_representation(self):
        """
        get output of each node as the structure-aware label representation
        """
        label_embedding = self.label_embedding(torch.arange(0, len(self.label_map)).long().to(self.device))
        label_embedding = label_embedding.unsqueeze(0)
        # print(label_embedding.shape,"label_embedding")
        tree_label_feature = self.graph_model(label_embedding)
        # print(tree_label_feature.shape,"tree_label_feature,graph_output")
        label_feature = tree_label_feature.squeeze(0)
        self.label_feature = label_feature

    @staticmethod
    def _soft_attention(text_f, label_f):
        """
        soft attention module
        :param text_f -> torch.FloatTensor, (batch_size, K, dim)
        :param label_f ->  torch.FloatTensor, (N, dim)
        :return: label_align ->  torch.FloatTensor, (batch, N, dim)
        """
        # print(text_f.shape, "text_f")
        # print(text_f.shape, "label_f")
        # print(text_f.shape,label_f.shape,"attention")
        att = torch.matmul(text_f, label_f.transpose(0, 1))
        # print(att.shape,"att")
        weight_label = functional.softmax(att.transpose(1, 2), dim=-1)
        label_align = torch.matmul(weight_label, text_f)
        return label_align

    def forward(self, text_feature):
        """
        forward pass with multi-label attention
        :param text_feature ->  torch.FloatTensor, (batch_size, K0, text_dim)
        :return: logits ->  torch.FloatTensor, (batch, N)
        """

        # text_feature = torch.cat(text_feature, 1)
        # print(text_feature.shape, "text_feature before view")
        # print(text_feature.shape,"text_feature")
        # text_feature = text_feature.view(text_feature.shape[0], -1,
        #  self.config.embedding.label.dimension)

        # print(text_feature.shape, "text_feature start")

        text_feature = text_feature.unsqueeze(1).to(self.device)

        # print(text_feature.shape, "text_feature after unsqueeze")

        if self.model_mode == 'TEST':
            label_feature = self.label_feature
        else:
            label_embedding = self.label_embedding(torch.arange(0, len(self.label_map)).long().to(self.device))
            # print(label_embedding.shape,"label_embedding")
            label_embedding = label_embedding.unsqueeze(0).to(self.device)
            # print(label_embedding.shape,"label_embedding after unaqueeze")
            tree_label_feature = self.graph_model(label_embedding)
            # pxrint(tree_label_feature.shape,"tree_label_feature_graph output")
            label_feature = tree_label_feature.squeeze(0).to(self.device)
            # print(label_feature.shape,"label_feature")

        label_aware_text_feature = self._soft_attention(text_feature, label_feature).to(self.device)
        label_information = label_aware_text_feature
        # print(label_aware_text_feature.shape,"label_aware_text_feature")
        label_aware_text_feature = self.linear(label_aware_text_feature.view(label_aware_text_feature.shape[0], -1)).to(
            self.device)
        # print(label_aware_text_feature.shape,"label_aware_text_feature after linear")
        logits = self.dropout(label_aware_text_feature).to(self.device)
        # print(logits.shape,"logits")
        return logits, label_information
