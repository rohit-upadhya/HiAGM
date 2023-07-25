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
import numpy as np 
import math


class Trainer(object):
    def __init__(self, model, criterion, optimizer, vocab, config, global_prototype_tensor, ac_matrix):
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
        # self.proto_model = None
        self.optimizer = optimizer
        self.contrastive = MultiLabelContrastiveLoss()
        self.global_prototype_tensor = global_prototype_tensor
        self.ac_matrix = ac_matrix
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

    def run(self, data_loader, epoch, stage, mode='TRAIN'):
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
            # if self.proto_model is None: #or self.proto_model.num_prototypes != num_labels:
            #     self.proto_model = Prototype(embedding_size=768, hidden_size=256, label_size=num_labels, device=self.device).to(self.config.train.device_setting.device)
            #     # Update optimizer with new model parameters
            #     self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.proto_model.parameters()),
            #                         lr=self.config.train.optimizer.learning_rate)
            # # similarity, proto_output = self.proto_model(label_information, torch.tensor(batch['label_list']).to(self.config.train.device_setting.device))
            # proto_loss = -torch.mean(torch.log(similarity))
            # self.global_prototype_tensors = self.proto_model(label_information.to(self.device), batch, global_prototype_tensor).to(self.device)
            # print(global_prototype_tensor.shape,"self.global_prototype_tensor")
            # print(label_information.shape,"label_information")
            # print(len(self.vocab.v2i['label']),"vocab")
            # proto_loss = self.prototype_loss(self.global_prototype_tensor, batch, label_information)
            # global_prototype_tensor_copy = global_prototype_tensor.detach()
            contrastive_loss = self.contrastive(label_information, batch['label_list'])
            s_to_z_loss = self.prototype_loss_s_to_z(self.global_prototype_tensor, label_information, batch, self.ac_matrix)
            # s_to_z_loss = s_to_z_loss.detach()
            print(s_to_z_loss,"s_to_z_loss")
            s_to_z_dash_loss = self.prototype_loss_s_to_z_dash(self.global_prototype_tensor, label_information, batch, self.ac_matrix)
            # s_to_z_dash_loss = s_to_z_dash_loss.detach()
            print(s_to_z_dash_loss,"s_to_z_dash_loss")
            # classification_loss, logits_proto = self.classification_loss(self.global_prototype_tensor.to(self.device),label_information.to(self.device),batch)
            # classification_loss = classification_loss.detach()
            # print(logits_classification.shape,"logits_classification")
            # print(classification_loss,"classification_loss")
            # proto_loss = self.prototype_loss(self.global_prototype_tensor, label_information, batch)
            # print(proto_loss,"proto_loss")
            # print(contrastive_loss,"contrastive_loss")
            if self.config.train.loss.recursive_regularization.flag:
                recursive_constrained_params = self.model.hiagm.linear.weight
            else:
                recursive_constrained_params = None
            classification_loss = self.criterion(logits,
                                  batch['label'].to(self.config.train.device_setting.device),
                                  recursive_constrained_params)
            # print(loss,"loss")
            # total_loss = loss + proto_loss
            
            weight_s_to_z = 0.5
            weight_s_to_z_dash = 0.5
            weight_classification = 1.0
            weight_contrastive = 0.5

            # calculate total loss
            total_loss = (weight_s_to_z * s_to_z_loss +
                          weight_s_to_z_dash * s_to_z_dash_loss +
                          weight_classification * classification_loss +
                          weight_contrastive * contrastive_loss)
            print(total_loss,"total_loss")
            if mode == 'TRAIN':
                self.optimizer.zero_grad()
                total_loss.backward()
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

    def train(self, data_loader, epoch):
        """
        training module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.train()
        return self.run(data_loader, epoch, 'Train', mode='TRAIN')

    def eval(self, data_loader, epoch, stage):
        """
        evaluation module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, TRAIN/DEV/TEST, log the result of the according corpus
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.eval()
        return self.run(data_loader, epoch, stage, mode='EVAL')
    
    # def prototype_loss(self, prototype_embeddings, batch_embeddings, batch):
    #     batch_size = batch_embeddings.shape[0]
    #     loss = 0.0
    #     N = batch_size**2  # normalize constant
        
    #     for i in range(prototype_embeddings.shape[0] - 1):  # Exclude the anti-prototype
            
    #         positive_mask = torch.zeros(batch_size, dtype=torch.bool)
    #         negative_mask = torch.ones(batch_size, dtype=torch.bool)
    #         for idx, labels in enumerate(batch['label_list']):
    #             if i in labels:
    #                 positive_mask[idx] = 1
    #                 negative_mask[idx] = 0
                    
            
    #         positive_examples = batch_embeddings[positive_mask, i, :]
    #         negative_examples = batch_embeddings[negative_mask, i, :]
            
            
    #         prototype = prototype_embeddings[i, :].unsqueeze(0)  # Add batch dimension
    #         positive_similarities = F.cosine_similarity(prototype, positive_examples) if positive_examples.shape[0] != 0 else 0
    #         negative_similarities = F.cosine_similarity(prototype, negative_examples) if negative_examples.shape[0] != 0 else 0
            
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

    # def prototype_loss_s_to_z(self, prototype_embeddings, batch_embeddings, batch):
    #     batch_size = batch_embeddings.shape[0]
    #     N = (batch_size ** 2) * prototype_embeddings.shape[0]

    #     positive_mask = torch.zeros((batch_size, prototype_embeddings.shape[0]-1), dtype=torch.bool)
    #     negative_mask = torch.ones((batch_size, prototype_embeddings.shape[0]-1), dtype=torch.bool)

    #     for idx, labels in enumerate(batch['label_list']):
    #         positive_mask[idx, labels] = 1
    #         negative_mask[idx, labels] = 0

    #     positive_examples = batch_embeddings[positive_mask].view(-1, batch_embeddings.shape[-1])
    #     negative_examples = batch_embeddings[negative_mask].view(-1, batch_embeddings.shape[-1])

    #     positive_distances = self.distance(prototype_embeddings[:-1].unsqueeze(0), positive_examples.unsqueeze(1)) if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)
    #     negative_distances = self.distance(prototype_embeddings[:-1].unsqueeze(0), negative_examples.unsqueeze(1)) if negative_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)

    #     positive_loss = -torch.log(positive_distances + 1e-8).nan_to_num(0.).sum() if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)
    #     negative_loss = -torch.log(1 - negative_distances + 1e-8).nan_to_num(0.).sum() if negative_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)

    #     loss = (positive_loss + negative_loss) / N if N > 0 else torch.tensor([0.0]).to(self.device)
    #     return loss

    # def prototype_loss_s_to_z(self, prototype_embeddings, batch_embeddings, batch, cost_matrix):
    #     num_prototypes = prototype_embeddings.shape[0] - 1
    #     batch_size = batch_embeddings.shape[0]
    #     num_labels = batch_embeddings.shape[1]
    #     N = batch_size ** 2

    #     positive_mask = torch.zeros((num_prototypes, batch_size, num_labels), dtype=torch.bool)
    #     negative_mask = torch.ones((num_prototypes, batch_size, num_labels), dtype=torch.bool)

    #     for idx, labels in enumerate(batch['label_list']):
    #         for label in labels:
    #             positive_mask[:, idx, label, labels] = 1
    #             negative_mask[:, idx, label, labels] = 0

    #     positive_examples = batch_embeddings.unsqueeze(0)[positive_mask].view(-1, batch_embeddings.shape[-1])
    #     negative_examples = batch_embeddings.unsqueeze(0)[negative_mask].view(-1, batch_embeddings.shape[-1])

    #     positive_distances = self.distance(prototype_embeddings[:-1].unsqueeze(0), positive_examples.unsqueeze(1)) if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)
    #     negative_distances = self.distance(prototype_embeddings[:-1].unsqueeze(0), negative_examples.unsqueeze(1)) if negative_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)

    #     negative_costs = cost_matrix[:, None, :]
    #     negative_distances *= negative_costs

    #     positive_loss = -torch.log(positive_distances + 1e-8).nan_to_num(0.).sum() if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)
    #     negative_loss = -torch.log(1 - negative_distances + 1e-8).nan_to_num(0.).sum() if negative_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)

    #     loss = (positive_loss + negative_loss) / N if N > 0 else torch.tensor([0.0]).to(self.device)

    #     return loss
    def prototype_loss_s_to_z(self, prototype_embeddings, batch_embeddings, batch, cost_matrix):
        ### we are taking the batch['label_list'] for each datapoint, contrustingt positive negative samples based on the prototype
        ### so if we assume there are only 3 labels and 2 datapoint, a = {0,1}, b={1,2}, positive samples for label 0 will be a0, positive samples for label 1 will be a1 and b1
        ### positive samples for label 2 will be b2. Negative samples for label 0 will be b0, for label 1 will be a0, a2, b0, b2 for label 2 will be a0, a1, b0, b1.
        ### It was easier to apply cost matrix here, since I was treating it as an undirected graph, I had to only take the column corresponding to the prototype at hand
        ### and simply element wise multiply with the negative distance.
        cost_matrix = torch.tensor(cost_matrix).to(self.device)  # Convert cost_matrix to a PyTorch tensor
        batch_size = batch_embeddings.shape[0]
        loss = 0.0
        N = batch_size ** 2 # normalize constant

        for i in range(prototype_embeddings.shape[0] - 1):  # Exclude the anti-prototype

            positive_mask = torch.zeros(batch_size, dtype=torch.bool)
            negative_mask = torch.ones(batch_size, dtype=torch.bool)
            for idx, labels in enumerate(batch['label_list']):
                if i in labels:
                    positive_mask[idx] = 1
                    negative_mask[idx] = 0

            positive_examples = batch_embeddings[positive_mask, i, :]
            negative_examples = batch_embeddings[negative_mask, i, :]

            prototype = prototype_embeddings[i, :].unsqueeze(0)  # Add batch dimension
            positive_distances = self.distance(prototype, positive_examples.unsqueeze(1)) if positive_examples.shape[0] != 0 else torch.tensor(0.0).to(self.device)

            # Get the costs for the negative examples
            negative_costs = cost_matrix[i]
            negative_distances = self.distance(prototype, negative_examples.unsqueeze(1)) * negative_costs if negative_examples.shape[0] != 0 else torch.tensor(0.0).to(self.device)

            positive_loss = -torch.log(positive_distances + 1e-8).sum() if positive_examples.shape[0] != 0 else torch.tensor(0.0).to(self.device)
            positive_loss = torch.nan_to_num(positive_loss)

            negative_loss = -torch.log(1 - negative_distances + 1e-8).sum() if negative_examples.shape[0] != 0 else torch.tensor(0.0).to(self.device)
            negative_loss = torch.nan_to_num(negative_loss)

            loss += (positive_loss + negative_loss) / N
        return loss

    # def prototype_loss_s_to_z_dash(self, prototype_embeddings, batch_embeddings, batch, cost_matrix):
    #     ### Here I am picking up each positive sample and calculating the distance to the corresponding prototype as the positive distance
    #     ### and calculating the distance to the other prototypes as the negative distance
    #     ### It proved to be much more difficult to apply te mask here as 
    #     ###
    #     ###
    #     ###
    #     ###
    #     batch_size = batch_embeddings.shape[0]
    #     N = (batch_size ** 2) * prototype_embeddings.shape[0]

    #     positive_mask = torch.zeros((batch_size, prototype_embeddings.shape[0]-1), dtype=torch.bool)
    #     for idx, labels in enumerate(batch['label_list']):
    #         positive_mask[idx, labels] = 1

    #     positive_examples = batch_embeddings[positive_mask].view(-1, batch_embeddings.shape[-1])

    #     positive_distances = self.distance(prototype_embeddings[:-1].unsqueeze(0), positive_examples.unsqueeze(1)) if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)

    #     other_prototype_mask = torch.ones(prototype_embeddings.shape[0], dtype=torch.bool)
    #     other_prototype_mask[torch.arange(prototype_embeddings.shape[0]-1)] = 0
    #     other_prototypes = prototype_embeddings[other_prototype_mask, :]

    #     other_prototype_distances = self.distance(other_prototypes.unsqueeze(0), positive_examples.unsqueeze(1)) if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)

    #     positive_loss = -torch.log(positive_distances + 1e-8).nan_to_num(0.).sum() if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)
    #     negative_loss = -torch.log(1 - other_prototype_distances + 1e-8).nan_to_num(0.).sum() if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)

    #     loss = (positive_loss + negative_loss) / N if N > 0 else torch.tensor([0.0]).to(self.device)
        # return loss

    def prototype_loss_s_to_z_dash(self, prototype_embeddings, batch_embeddings, batch, cost_matrix):

        ### Here I am picking up each positive sample and calculating the distance to the corresponding prototype as the positive distance
        ### and calculating the distance to the other prototypes as the negative distance
        ### for negative distance I am calculating the distance to all prototypes and multiplying it by the cost matrix, and since in the cost matrix the cost of going from the
        ### label to the same label is 0, it gets ignored, so we only get the distance to the other prototypes
        ### for both of these losses I have decided to exclude the antiprototype for distance measure as it did not make sense to me to do so, as he antiprototype was only used
        ### for handling the change of multiple path to the same datapoint

        batch_size = batch_embeddings.shape[0]
        N = (batch_size ** 2) * (prototype_embeddings.shape[0] - 1)  # -1 to exclude anti-prototype

        positive_mask = torch.zeros((batch_size, prototype_embeddings.shape[0]-1), dtype=torch.bool)
        for idx, labels in enumerate(batch['label_list']):
            positive_mask[idx, labels] = 1

        positive_examples = batch_embeddings[positive_mask].view(-1, batch_embeddings.shape[-1])

        positive_distances = self.distance(prototype_embeddings[:-1].unsqueeze(0), positive_examples.unsqueeze(1)) if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)

        
        prototype_embeddings = prototype_embeddings[:-1]

        
        all_distances = self.distance(batch_embeddings.unsqueeze(1), prototype_embeddings.unsqueeze(0))

        
        weighted_distances = all_distances * torch.Tensor(cost_matrix).unsqueeze(0).to(self.device)

        positive_loss = -torch.log(positive_distances + 1e-8).nan_to_num(0.).sum() if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)
        negative_loss = -torch.log(1 - weighted_distances + 1e-8).nan_to_num(0.).sum() if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)

        loss = (positive_loss + negative_loss) / N if N > 0 else torch.tensor([0.0]).to(self.device)
        return loss


    # def prototype_loss_s_to_z_dash(self, prototype_embeddings, batch_embeddings, batch, cost_matrix):
    # # Convert cost_matrix to a PyTorch tensor and append a row of zeros at the end
    #     cost_matrix = torch.tensor(cost_matrix, dtype=torch.float32).to(self.device)  # Convert cost_matrix to a PyTorch tensor
    #     zeros_row = torch.zeros((1, cost_matrix.shape[1]), dtype=torch.float32).to(self.device)
    #     zeros_column = torch.zeros((cost_matrix.shape[0]+1, 1), dtype=torch.float32).to(self.device)
    #     cost_matrix = torch.cat([cost_matrix, zeros_row], dim=0)
    #     cost_matrix = torch.cat([cost_matrix, zeros_column], dim=1)
    #     print(cost_matrix.shape,"cost_matrix")
    #     batch_size = batch_embeddings.shape[0]
    #     N = (batch_size ** 2) * prototype_embeddings.shape[0]

    #     positive_mask = torch.zeros((batch_size, prototype_embeddings.shape[0]-1), dtype=torch.bool)
    #     for idx, labels in enumerate(batch['label_list']):
    #         positive_mask[idx, labels] = 1
    #     print(positive_mask.shape,"positive_mask")
    #     positive_examples = batch_embeddings[positive_mask].view(-1, batch_embeddings.shape[-1])
    #     print(positive_examples.shape,"positive_examples")
    #     positive_distances = self.distance(prototype_embeddings[:-1].unsqueeze(0), positive_examples.unsqueeze(1)) if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)
    #     print(positive_distances.shape,"positive_distances")
    #     other_prototype_mask = torch.ones(prototype_embeddings.shape[0], dtype=torch.bool)
    #     print(other_prototype_mask.shape,"other_prototype_mask")
    #     other_prototype_mask[torch.arange(prototype_embeddings.shape[0]-1)] = 0
    #     other_prototypes = prototype_embeddings[other_prototype_mask, :]
    #     print(other_prototypes.shape,"other_prototypes")
    #     other_prototype_distances = self.distance(other_prototypes.unsqueeze(0), positive_examples.unsqueeze(1)) if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)
    #     print(other_prototype_distances.shape,"other_prototype_distances")
    #     # Incorporate the cost matrix into the calculation of the negative loss
        
    #     other_prototype_costs = cost_matrix[:, other_prototype_mask]
    #     print(other_prototype_costs.shape,"other_prototype_costs")
    #     other_prototype_distances *= other_prototype_costs.unsqueeze(-1)
    #     print(other_prototype_distances.shape,"other_prototype_distances")

    #     positive_loss = -torch.log(positive_distances + 1e-8).nan_to_num(0.).sum() if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)
    #     negative_loss = -torch.log(1 - other_prototype_distances + 1e-8).nan_to_num(0.).sum() if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)

    #     loss = (positive_loss + negative_loss) / N if N > 0 else torch.tensor([0.0]).to(self.device)
    #     return loss


    # def prototype_loss_s_to_z_dash(self, prototype_embeddings, batch_embeddings, batch, cost_matrix):
    #     cost_matrix_padded = np.pad(cost_matrix, ((0, 1), (0, 1)), 'constant', constant_values=0)
    #     cost_matrix_tensor = torch.tensor(cost_matrix_padded).to(self.device)
    #     batch_size = batch_embeddings.shape[0]
    #     N = (batch_size ** 2) * prototype_embeddings.shape[0]

    #     positive_mask = torch.zeros((batch_size, prototype_embeddings.shape[0]-1), dtype=torch.bool)
    #     for idx, labels in enumerate(batch['label_list']):
    #         positive_mask[idx, labels] = 1

    #     positive_examples = batch_embeddings[positive_mask].view(-1, batch_embeddings.shape[-1])
    #     print(f"Shape of positive_examples: {positive_examples.shape}")

    #     positive_distances = self.distance(prototype_embeddings[:-1].unsqueeze(0), positive_examples.unsqueeze(1)) if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)
    #     print(f"Shape of positive_distances: {positive_distances.shape}")

    #     other_prototype_mask = torch.ones(prototype_embeddings.shape[0], dtype=torch.bool)
    #     print(other_prototype_mask.shape,"other_prototype_mask before arrange")
    #     other_prototype_mask[torch.arange(prototype_embeddings.shape[0]-1)] = 0
    #     print(other_prototype_mask.shape,"other_prototype_mask")
    #     other_prototypes = prototype_embeddings[other_prototype_mask, :]
    #     print(f"Shape of other_prototypes: {other_prototypes.shape}")
    #     print(f"Shape of positive_examples.unsqueeze(1): {positive_examples.unsqueeze(1).shape}")
    #     other_prototype_distances = self.distance(other_prototypes.unsqueeze(0), positive_examples.unsqueeze(1)) if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)
    #     print(f"Shape of other_prototype_distances: {other_prototype_distances.shape}")

    #     cost_matrix_batch = cost_matrix_tensor.repeat(batch_size, 1, 1)
    #     print(f"Shape of cost_matrix_batch: {cost_matrix_batch.shape}")

    #     other_prototype_costs = cost_matrix_batch[:, other_prototype_mask]
    #     print(f"Shape of other_prototype_costs: {other_prototype_costs.shape}")

    #     other_prototype_distances *= other_prototype_costs.unsqueeze(-1)
    #     print(f"Shape of other_prototype_distances after cost multiplication: {other_prototype_distances.shape}")

    #     positive_loss = -torch.log(positive_distances + 1e-8).nan_to_num(0.).sum() if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)
    #     negative_loss = -torch.log(1 - other_prototype_distances + 1e-8).nan_to_num(0.).sum() if positive_examples.shape[0] > 0 else torch.tensor([0.0]).to(self.device)

    #     loss = (positive_loss + negative_loss) / N if N > 0 else torch.tensor([0.0]).to(self.device)
    #     return loss







    def distance(self, s_i, s_j):
        s_i_norm = F.normalize(s_i, p=2, dim=-1)
        s_j_norm = F.normalize(s_j, p=2, dim=-1)
        cosine_similarity = F.cosine_similarity(s_i_norm, s_j_norm, dim=-1)
        distance = 1 / (1 + torch.exp(cosine_similarity))
        return distance
    

    # def classification_loss(self, prototype_embeddings, batch_embeddings, batch):
    #     batch_size = batch_embeddings.shape[0]
    #     num_labels = batch_embeddings.shape[1]
    #     logits = torch.empty((batch_size, num_labels))

    #     for i in range(batch_size):
    #         for j in range(num_labels):
    #             label_embedding = batch_embeddings[i, j, :]
    #             prototype = prototype_embeddings[j, :]

                
    #             logits[i, j] = F.cosine_similarity(prototype.unsqueeze(0), label_embedding.unsqueeze(0)).to(self.device)
        
        
    #     logits = torch.sigmoid(logits).to(self.device)

    #     bce_loss = torch.nn.BCEWithLogitsLoss().to(self.device)
    #     targets = torch.zeros((batch_size, num_labels)).to(self.device)
    #     for i in range(batch_size):
    #         targets[i, batch['label_list'][i]] = 1

    #     loss = bce_loss(logits, targets).to(self.device)
    #     return loss.to(self.device), logits.to(self.device)