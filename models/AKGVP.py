from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.model_util import norm_col_init, weights_init

from .model_io import ModelOutput

import scipy.sparse as sp
import numpy as np
import scipy.io as scio
import os
import json
import copy
from models.object_distribution import Object_Distribution
import pickle

def load_scene_graph(path):
    graph = {}
    scenes = ['Kitchens', 'Living_Rooms', 'Bedrooms', 'Bathrooms']
    for s in scenes:
        data = scio.loadmat(os.path.join(path, s+'.mat'))
        graph[s] = data

    return graph


class AKGVPModel(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        self.num_cate = args.num_category
        # resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(AKGVPModel, self).__init__()

        clip_embedding_sz = 2048
        self.conv1 = nn.Conv2d(clip_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)

        self.detection_feature = nn.Linear(6, self.num_cate)
        self.graph_detection_other_indicator_linear_1 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_other_indicator_linear_2 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_indicator = nn.Sequential(
            # nn.Linear(518, 128),
            nn.Linear(1541, 128),
            nn.ReLU(),
            nn.Linear(128, 49),
        )

        self.action_at_a = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), requires_grad=False)
        self.action_at_b = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]), requires_grad=False)
        self.action_at_scale = nn.Parameter(torch.tensor(0.58), requires_grad=False)

        self.graph_detection_feature = nn.Sequential(
            # nn.Linear(518, 128),
            nn.Linear(1541, 128),
            nn.ReLU(),
            nn.Linear(128, 49),
        )

        self.graph_detection_other_info_linear_1 = nn.Linear(6, self.num_cate)
        self.graph_detection_other_info_linear_2 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_other_info_linear_3 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_other_info_linear_4 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_other_info_linear_5 = nn.Linear(self.num_cate, self.num_cate)

        self.embed_action = nn.Linear(action_space, 10)

        pointwise_in_channels = 64 + self.num_cate + self.num_cate + 10

        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        self.lstm_input_sz = 7 * 7 * 64

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTM(self.lstm_input_sz, hidden_state_sz, 2)
        num_outputs = action_space
        self.critic_linear_1 = nn.Linear(hidden_state_sz, 64)
        self.critic_linear_2 = nn.Linear(64, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear_1.weight.data = norm_col_init(
            self.critic_linear_1.weight.data, 1.0
        )
        self.critic_linear_1.bias.data.fill_(0)
        self.critic_linear_2.weight.data = norm_col_init(
            self.critic_linear_2.weight.data, 1.0
        )
        self.critic_linear_2.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_ih_l1.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.bias_hh_l1.data.fill_(0)

        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.info_embedding = nn.Linear(5,49)
        self.scene_embedding = nn.Conv2d(86,64,1,1)
        self.scene_classifier = nn.Linear(64*7*7,4)

        self.object_distribution = Object_Distribution(dataset='AI2THOR', feature_memory=args.feature_memory, mode=args.model_phase)  # choose memory save feature or one-hot vector 
        self.TDE_threshold = torch.tensor(args.TDE_threshold)  # p - TDE_threshold* P^
        self.scale_min = torch.tensor(args.scale_min)  
        self.scale_max = torch.tensor(args.scale_max)
        self.mode = args.TDE_mode  # zero, one, random
        self.phase = args.model_phase


        device = "cuda" if torch.cuda.is_available() else "cpu"
        with open('models/word_clip_feat.data', 'rb') as f:
            self.clip_scene, self.clip_object = pickle.load(f)
        self.clip_scene = torch.from_numpy(self.clip_scene).to(device)
        self.clip_object = torch.from_numpy(self.clip_object).to(device)

        self.W0 = nn.Linear(22, 22, bias=False)

    def embedding(self, state, target, action_embedding_input, scene, target_object, counterfact):
        target_object_one_hot = target['indicator']

        at_v = torch.mul(target['info'][:, -1].view(target['info'].shape[0], 1), target['indicator'])
        at = torch.mul(torch.max(at_v), self.action_at_scale)
        action_at = torch.mul(at, self.action_at_a) + self.action_at_b

        self.object_distribution.observation_memory_update(target, scene, target_object_one_hot)

        target_info_org = torch.cat((target['info'], target['indicator']), dim=1)
        target_info = F.relu(self.graph_detection_other_info_linear_1(target_info_org))
        target_info = target_info.t()
        target_info = F.relu(self.graph_detection_other_info_linear_2(target_info))
        target_info = F.relu(self.graph_detection_other_info_linear_3(target_info))
        target_info = F.relu(self.graph_detection_other_info_linear_4(target_info))
        target_info = F.relu(self.graph_detection_other_info_linear_5(target_info))
        target_appear = torch.mm(target['appear'].t(), target_info).t()

        target_clip_feat = target['indicator'] * self.clip_object
        target_exp = torch.cat((target_appear, target['info'], target_clip_feat), dim=1)

        attention_weight = torch.mm(self.object_distribution.prob_matrix, target_object_one_hot)
        target_exp = torch.mul(target_exp, attention_weight)

        target_exp = F.relu(self.graph_detection_feature(target_exp))
        target_exp_embedding = target_exp.reshape(1, self.num_cate, 7, 7)

        action_embedding = F.relu(self.embed_action(action_embedding_input))
        action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)
        
        target_indicator = F.relu(self.detection_feature(target_info_org))
        target_indicator = target_indicator.t()
        target_indicator = F.relu(self.graph_detection_other_indicator_linear_1(target_indicator))
        target_indicator = F.relu(self.graph_detection_other_indicator_linear_2(target_indicator))
        target_indicator = torch.mm(target['appear'].t(), target_indicator).t()
        target_indicator = torch.cat((target_indicator, target['info'], target_clip_feat), dim=1)
        target_indicator = self.graph_detection_indicator(target_indicator)
        target_indicator_embedding = target_indicator.reshape(1, self.num_cate, 7, 7)
        image_embedding = F.relu(self.conv1(state))
        rand_zero = torch.where(torch.rand(1)>0.5, 1, 0).to(image_embedding.device)
        
        if self.phase == 'train':
            target_indicator_embedding = target_indicator_embedding*rand_zero
            image_embedding = image_embedding*rand_zero
        elif self.phase == 'test':
            if counterfact:
                # set variable to specific value
                if self.mode == 'zero':
                    image_embedding = torch.zeros(1, 64, 7, 7).to(target_exp.device)
                    target_indicator_embedding = torch.zeros(1, self.num_cate, 7, 7).to(target_exp.device)
                    # print(self.mode)
                elif self.mode == 'one':
                    image_embedding = torch.ones(1, 64, 7, 7).to(target_exp.device)
                    target_indicator_embedding = torch.ones(1, self.num_cate, 7, 7).to(target_exp.device)
                    # print(self.mode)
                else:
                    image_embedding = torch.rand(1, 64, 7, 7).to(target_exp.device)
                    target_indicator_embedding = torch.rand(1, self.num_cate, 7, 7).to(target_exp.device)
                    # print(self.mode)
            else:
                pass  # not change the variable (for the fact ithem)
        else:
            raise NotImplementedError()

        x = self.dropout(image_embedding)

        x = torch.cat((x, target_exp_embedding, target_indicator_embedding, action_reshaped), dim=1)

        x = F.relu(self.pointwise(x))
        x = self.dropout(x)
        out = x.view(x.size(0), -1)

        return out, image_embedding, action_at

    def a3clstm(self, embedding, prev_hidden_h, prev_hidden_c):

        embedding = embedding.reshape([1, 1, self.lstm_input_sz])
        output, (hx, cx) = self.lstm(embedding, (prev_hidden_h, prev_hidden_c))
        x = output.reshape([1, self.hidden_state_sz])

        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear_1(x)
        critic_out = self.critic_linear_2(critic_out)

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options, counterfact=False):
        scene = model_input.scene
        target_object = model_input.target_object

        state = model_input.state
        (hx, cx) = model_input.hidden

        target = model_input.target_class_embedding
        action_probs = model_input.action_probs

        x, image_embedding, action_at = self.embedding(state, target, action_probs, scene, target_object, counterfact)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx)
        actor_out = torch.mul(actor_out, action_at)

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )
