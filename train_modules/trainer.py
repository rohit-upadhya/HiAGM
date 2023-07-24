#!/usr/bin/env python
# coding:utf-8

import helper.logger as logger
from train_modules.evaluation_metrics import evaluate
# from models.proto.train import proto_trainer
from models.proto.models.prototype import Prototype
from models.proto.contrastive import MultiLabelContrastiveLoss
import torch
import tqdm
import torch.optim as optim
import torch.nn.functional as F

import math


class Trainer(object):
    def __init__(self, model, criterion, optimizer, vocab, config):
        """
        :param model: Computational Graph
        :param criterion: train_modules.ClassificationLoss object
        :param optimizer: optimization function for backward pass
        :param vocab: vocab.v2i -> Dict{'token': Dict{vocabulary to id map}, 'label': Dict{vocabulary
        to id map}}, vocab.i2v -> Dict{'token': Dict{id to vocabulary map}, 'label': Dict{id to vocabulary map}}
        :param config: helper.Configure object
        """
        super(Trainer, self).__init__()
        self.model = model
        self.vocab = vocab
        self.config = config
        self.criterion = criterion
        self.device = self.config.train.device_setting.device
        self.proto_model = None
        self.optimizer = optimizer
        self.contrastive = MultiLabelContrastiveLoss()
        self.global_prototype_tensors = None
        # self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.proto_model.parameters()),
                                    # lr=self.config.train.optimizer.learning_rate)

    def update_lr(self):
        """
        (callback function) update learning rate according to the decay weight
        """
        logger.warning('Learning rate update {}--->{}'
                       .format(self.optimizer.param_groups[0]['lr'],
                               self.optimizer.param_groups[0]['lr'] * self.config.train.optimizer.lr_decay))
        for param in self.optimizer.param_groups:
            param['lr'] = self.config.train.optimizer.learning_rate * self.config.train.optimizer.lr_decay

    def run(self, data_loader, epoch, stage, global_prototype_tensor, mode='TRAIN'):
        """
        training epoch
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, e.g. 'TRAIN'/'DEV'/'TEST', figure out the corpus
        :param mode: str, ['TRAIN', 'EVAL'], train with backward pass while eval without it
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        predict_probs = []
        target_labels = []
        total_loss = 0.0
        num_batch = data_loader.__len__()

        for batch in tqdm.tqdm(data_loader):
            logits, label_information = self.model(batch)
            num_labels = label_information.size(1)
            if self.proto_model is None: #or self.proto_model.num_prototypes != num_labels:
                self.proto_model = Prototype(embedding_size=768, hidden_size=256, label_size=num_labels, device=self.device).to(self.config.train.device_setting.device)
                # Update optimizer with new model parameters
                self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.proto_model.parameters()),
                                    lr=self.config.train.optimizer.learning_rate)
            # similarity, proto_output = self.proto_model(label_information, torch.tensor(batch['label_list']).to(self.config.train.device_setting.device))
            # proto_loss = -torch.mean(torch.log(similarity))
            self.global_prototype_tensors = self.proto_model(label_information.to(self.device), batch, global_prototype_tensor).to(self.device)
            print(global_prototype_tensor.shape,"self.global_prototype_tensor")
            # print(label_information.shape,"label_information")
            # print(len(self.vocab.v2i['label']),"vocab")
            # proto_loss = self.prototype_loss(self.global_prototype_tensor, batch, label_information)
            global_prototype_tensor_copy = global_prototype_tensor.detach()
            contrastive_loss = self.contrastive(label_information, batch['label_list'])
            contrastive_loss = contrastive_loss.detach()
            s_to_z_loss = self.prototype_loss_s_to_z(global_prototype_tensor_copy, label_information, batch)
            s_to_z_loss = s_to_z_loss.detach()
            print(s_to_z_loss,"s_to_z_loss")
            s_to_z_dash_loss = self.prototype_loss_s_to_z_dash(global_prototype_tensor_copy, label_information, batch)
            s_to_z_dash_loss = s_to_z_dash_loss.detach()
            print(s_to_z_dash_loss,"s_to_z_dash_loss")
            classification_loss, logits_proto = self.classification_loss(global_prototype_tensor.to(self.device),label_information.to(self.device),batch)
            # classification_loss = classification_loss.detach()
            # print(logits_classification.shape,"logits_classification")
            print(classification_loss,"classification_loss")
            # proto_loss = self.prototype_loss(self.global_prototype_tensor, label_information, batch)
            # print(proto_loss,"proto_loss")
            print(contrastive_loss,"contrastive_loss")
            if self.config.train.loss.recursive_regularization.flag:
                recursive_constrained_params = self.model.hiagm.linear.weight
            else:
                recursive_constrained_params = None
            # loss = self.criterion(logits_proto,
                                  # batch['label'].to(self.config.train.device_setting.device),
                                  # recursive_constrained_params)
            # print(loss,"loss")
            # total_loss = loss + proto_loss
            
            weight_s_to_z = 1.0
            weight_s_to_z_dash = 1.0
            weight_classification = 1.0
            weight_contrastive = 1.0

            # calculate total loss
            total_loss = (weight_s_to_z * s_to_z_loss +
                          weight_s_to_z_dash * s_to_z_dash_loss +
                          weight_classification * classification_loss +
                          weight_contrastive * contrastive_loss)
            if mode == 'TRAIN':
                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                # proto_loss.backward()
                self.optimizer.step()
            # logits_classification = logits_classification.cpu()
            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            # print(batch)
            target_labels.extend(batch['label_list'])
        total_loss = total_loss / num_batch
        if mode == 'EVAL':
            metrics = evaluate(predict_probs,
                               target_labels,
                               self.vocab,
                               self.config.eval.threshold)
            # metrics = {'precision': precision_micro,
            #             'recall': recall_micro,
            #             'micro_f1': micro_f1,
            #             'macro_f1': macro_f1}
            logger.info("%s performance at epoch %d --- Precision: %f, "
                        "Recall: %f, Micro-F1: %f, Macro-F1: %f, Loss: %f.\n"
                        % (stage, epoch,
                           metrics['precision'], metrics['recall'], metrics['micro_f1'], metrics['macro_f1'],
                           total_loss))
            return metrics

    def train(self, data_loader, epoch, global_prototype_tensor):
        """
        training module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.train()
        return self.run(data_loader, epoch, 'Train', mode='TRAIN', global_prototype_tensor=global_prototype_tensor), self.global_prototype_tensors

    def eval(self, data_loader, epoch, stage):
        """
        evaluation module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, TRAIN/DEV/TEST, log the result of the according corpus
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.eval()
        return self.run(data_loader, epoch, stage, mode='EVAL', global_prototype_tensor=self.global_prototype_tensors), self.global_prototype_tensors
    
    # def prototype_loss(self, prototype_embeddings, batch_embeddings, batch):
    #     batch_size = batch_embeddings.shape[0]
    #     loss = 0.0
    #     N = batch_size**2  # normalize constant
        
    #     for i in range(prototype_embeddings.shape[0] - 1):  # Exclude the anti-prototype
    #         
    #         positive_mask = torch.zeros(batch_size, dtype=torch.bool)
    #         negative_mask = torch.ones(batch_size, dtype=torch.bool)
    #         for idx, labels in enumerate(batch['label_list']):
    #             if i in labels:
    #                 positive_mask[idx] = 1
    #                 negative_mask[idx] = 0
                    
    #         
    #         positive_examples = batch_embeddings[positive_mask, i, :]
    #         negative_examples = batch_embeddings[negative_mask, i, :]
            
    #         
    #         prototype = prototype_embeddings[i, :].unsqueeze(0)  # Add batch dimension
    #         positive_similarities = F.cosine_similarity(prototype, positive_examples) if positive_examples.shape[0] != 0 else 0
    #         negative_similarities = F.cosine_similarity(prototype, negative_examples) if negative_examples.shape[0] != 0 else 0
    #         
    #         positive_loss = -torch.log(positive_similarities + 1e-8).sum().item() if positive_examples.shape[0] != 0 else 0
    #         positive_loss = torch.nan_to_num(positive_loss)
    #         # if(math.isnan(positive_loss)):
    #         #     positive_loss = 0
    #         # print(positive_loss,"positive_loss")
    #         negative_loss = -torch.log(1 - negative_similarities + 1e-8).sum().item() if negative_examples.shape[0] != 0 else 0
    #         negative_loss = torch.nan_to_num(negative_loss)
    #         # if(math.isnan(negative_loss)):
    #         #     negative_loss = 0
    #         # print(negative_loss,"negative_loss")
    #         loss += (positive_loss + negative_loss) / N
    #     return loss

    def prototype_loss_s_to_z(self, prototype_embeddings, batch_embeddings, batch):
        batch_size = batch_embeddings.shape[0]
        loss = 0.0
        N = batch_size**2  
        
        for i in range(prototype_embeddings.shape[0] - 1): 
            
            positive_mask = torch.zeros(batch_size, dtype=torch.bool)
            negative_mask = torch.ones(batch_size, dtype=torch.bool)
            for idx, labels in enumerate(batch['label_list']):
                if i in labels:
                    positive_mask[idx] = 1
                    negative_mask[idx] = 0
                    
            
            positive_examples = batch_embeddings[positive_mask, i, :]
            negative_examples = batch_embeddings[negative_mask, i, :]
            
            
            prototype = prototype_embeddings[i, :].unsqueeze(0)  
            zero_es = torch.tensor(0)

            
          
            positive_similarities = F.cosine_similarity(prototype, positive_examples) if positive_examples.shape[0] != 0 else zero_es
            negative_similarities = F.cosine_similarity(prototype, negative_examples) if negative_examples.shape[0] != 0 else zero_es
            # print(positive_similarities,"positive_similarities")
            # print(negative_similarities,"negative_similarities")
            
            positive_loss = -torch.log(positive_similarities + 1e-8).sum() if positive_examples.shape[0] != 0 else zero_es
            positive_loss = torch.nan_to_num(positive_loss).to(self.device)
            negative_loss = -torch.log(1 - positive_similarities + 1e-8).sum() if positive_examples.shape[0] != 0 else zero_es
            negative_loss = torch.nan_to_num(negative_loss).to(self.device)

            # if positive_loss.dim() == 0:
            #     positive_loss = positive_loss.view(1, 1)
            # if negative_loss.dim() == 0:
            #     negative_loss = negative_loss.view(1, 1)
            
            # N = N.view(1, 1)
            
            # print(positive_loss,"positive_loss")
            # print(negative_loss,"negative_loss")
            
            loss += (positive_loss + negative_loss) / N
        return loss

    def prototype_loss_s_to_z_dash(self, prototype_embeddings, batch_embeddings, batch):
        batch_size = batch_embeddings.shape[0]
        loss = 0.0
        N = batch_size**2  

        for i in range(prototype_embeddings.shape[0] - 1):  
            
            positive_mask = torch.zeros(batch_size, dtype=torch.bool)
            for idx, labels in enumerate(batch['label_list']):
                if i in labels:
                    positive_mask[idx] = 1
                        
            
            positive_examples = batch_embeddings[positive_mask, i, :]

            
            prototype = prototype_embeddings[i, :].unsqueeze(0)  
            positive_similarities = F.cosine_similarity(prototype, positive_examples) if positive_examples.shape[0] != 0 else 0

           
            other_prototype_mask = torch.ones(prototype_embeddings.shape[0], dtype=torch.bool)
            other_prototype_mask[i] = 0
            other_prototypes = prototype_embeddings[other_prototype_mask, :]
            
            if positive_examples.shape[0] != 0:
                other_prototype_similarities = torch.Tensor(
                    [F.cosine_similarity(other_prototypes[j,:], positive_examples, dim=1).mean() for j in range(other_prototypes.shape[0])]).to(self.device)
            else:
                other_prototype_similarities = 0

            
            zero_es = torch.tensor(0)
            positive_loss = zero_es
            negative_loss = zero_es
            positive_loss = -torch.log(positive_similarities + 1e-8).sum() if positive_examples.shape[0] != 0 else zero_es
            positive_loss = torch.nan_to_num(positive_loss).to(self.device)
            negative_loss = -torch.log(1 - other_prototype_similarities + 1e-8).sum() if positive_examples.shape[0] != 0 else zero_es
            negative_loss = torch.nan_to_num(negative_loss).to(self.device)

            
            loss += (positive_loss + negative_loss) / N
        return loss
    

    def classification_loss(self, prototype_embeddings, batch_embeddings, batch):
        batch_size = batch_embeddings.shape[0]
        num_labels = batch_embeddings.shape[1]
        logits = torch.empty((batch_size, num_labels))

        for i in range(batch_size):
            for j in range(num_labels):
                label_embedding = batch_embeddings[i, j, :]
                prototype = prototype_embeddings[j, :]

                
                logits[i, j] = F.cosine_similarity(prototype.unsqueeze(0), label_embedding.unsqueeze(0)).to(self.device)
        
        
        logits = torch.sigmoid(logits).to(self.device)

        bce_loss = torch.nn.BCEWithLogitsLoss().to(self.device)
        targets = torch.zeros((batch_size, num_labels)).to(self.device)
        for i in range(batch_size):
            targets[i, batch['label_list'][i]] = 1

        loss = bce_loss(logits, targets).to(self.device)
        return loss.to(self.device), logits.to(self.device)