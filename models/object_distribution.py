from asyncio import constants
import torch
import scipy.io as scio
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
import copy
import numpy as np
import torch.nn as nn


class Object_Distribution(torch.nn.Module):

    def __init__(self, dataset='AI2THOR', stack_length=20, feature_memory=False, mode='train'):
        '''
        :param prior_alpha: class_num*(class_num-1)
        '''
        super(Object_Distribution, self).__init__()
        self.dataset = dataset
        self.stack_length = stack_length
        self.init_prior_alpha = self.upload_prior_alpha()
        self.class_num = self.init_prior_alpha.shape[1]
        self.prior_alpha = None
        self.posterior_alpha = None
        self.feature_stack = None
        self.object_stack = None
        self.prob_matrix = None
        self.kl = 0.0
        self.mode = mode  # train or test
        self.ii_pro = torch.ones(1)
        # self.base_scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.similarity_scale = nn.Parameter(torch.tensor(0.0001), requires_grad=True)
        self.feature_memory = feature_memory
        self.update_scale = nn.Sequential(
            nn.Linear(22, 22),
            nn.ReLU(),
            nn.Linear(22, 22),
            nn.ReLU(),
        )

    def upload_prior_alpha(self):
        init_prior_alpha = []
        if self.dataset == 'AI2THOR':
            scenes = ['Kitchens', 'Living_Rooms', 'Bedrooms', 'Bathrooms']
        elif self.dataset == 'MP3D' or 'RoboTHOR':
            scenes = ['scenes']
        data = scio.loadmat('prior_distribution/dis_param_{}_v2.mat'.format(self.dataset))
        for s in scenes:
            raw_data = data[s]
            up_tri = np.pad(np.triu(raw_data),((0,0),(1,0)),'constant',constant_values=0.0)
            low_tri = np.pad(np.tril(raw_data,-1),((0,0),(0,1)),'constant',constant_values=0.0)
            init_prior_alpha.append(up_tri+low_tri)
        return torch.tensor(init_prior_alpha, dtype=torch.float32)

    def get_scene_id(self, current_scene):
        id_number = "".join(filter(str.isdigit, current_scene))
        pass
        if 0 < int(id_number) < 200:
            idx = 0
        elif 200 < int(id_number) < 300:
            idx = 1
        elif 300 < int(id_number) < 400:
            idx = 2
        elif 400 < int(id_number) < 500:
            idx = 3
        return idx

    def observation_memory_update(self, observation_t, current_scene, target_object, update_co=5.0):
        '''
        observation_t['info']:bounding box + probability (22, 5)
        observation_t['indicator']:target object (22, 1)
        observation_t['appear']:region features (22, 512)
        '''

        if self.dataset == 'AI2THOR':
            scene_id = self.get_scene_id(current_scene)
        elif self.dataset == 'MP3D' or 'RoboTHOR':
            scene_id = 0

        if (self.feature_stack is None) and (self.object_stack is None):  # init the memory and parm
            device = observation_t['indicator'].device
            if self.feature_memory:
                self.feature_stack = torch.zeros(self.stack_length, 22, 512).to(device)
            else:
                self.object_stack = torch.zeros(self.stack_length, 22).to(device)
            self.prior_alpha = copy.deepcopy(self.init_prior_alpha[scene_id, :, :]).to(device)
            self.posterior_alpha = copy.deepcopy(self.prior_alpha)
            self.pro_matrix_update()
            self.kl = torch.tensor([0.0]).to(device)

        object_t = torch.sign(observation_t['info'][:, -1])
        if self.feature_memory:
            current_similarity = self.feature_stack - observation_t['appear'].view(1,22,512).repeat(self.stack_length,1,1)
            current_similarity = torch.min(torch.square(current_similarity.view(self.stack_length, 11264)).mean(1))
        else:
            current_similarity = self.object_stack - object_t.view(1,22).repeat(self.stack_length,1)
            current_similarity = torch.min(torch.square(current_similarity).mean(1))
        if current_similarity > self.similarity_scale and object_t.sum(-1) > 1:
            # memory update
            if self.feature_memory:
                self.feature_stack = self.memory_update(observation_t['appear'], self.feature_stack.detach())
            else:
                self.object_stack = self.memory_update(object_t, self.object_stack.detach())
            # self.object_stack = self.memory_update(object_t, self.object_stack)
            # compute distribution
            self.prior_alpha.detach()
            self.posterior_alpha.detach()
            update_scale = self.update_scale(object_t+target_object.squeeze())
            # update_scale = update_scale.mul(10.0)
            update_scale = update_co  # hoz 10.0, has great influence 5.0 has better performance
            self.posterior_update(update_scale, object_t)
            if self.mode == 'test':
                self.compute_kl()
            self.pro_matrix_update()
            self.prior_update()
        else:
            self.kl = self.kl * 0.8
            
    def reset_memory(self):
        self.feature_stack = None
        self.object_stack = None

    def posterior_update(self, update_scale, observed_objects):
        # observed_objects = self.object_memory[-1]
        # observed_objects = torch.tensor([0,1,0,0,1,1,0]).to(observed_objects.device)
        num_cls = observed_objects.shape[0]
        observed_mat = observed_objects.view(1, num_cls).repeat(num_cls,1)
        observed_mat = observed_mat*(observed_objects.view(num_cls,1))-torch.diag(observed_objects)
        self.posterior_alpha = self.prior_alpha + torch.mul(observed_objects, update_scale)
        

    def prior_update(self):
        self.prior_alpha = copy.deepcopy(self.posterior_alpha.detach())
    
    def memory_update(self, arr, memory):
        current = torch.unsqueeze(arr, dim=0)
        return torch.cat((current, memory[1:]), dim=0)

    def pro_matrix_update(self):
        # expation of Dirichlet distribution url: https://en.wikipedia.org/wiki/Dirichlet_distribution
        pro_matrix = self.posterior_alpha/torch.sum(self.posterior_alpha,dim=(1,),keepdim=True)
        pro_matrix = pro_matrix + torch.eye(pro_matrix.shape[0]).to(pro_matrix.device)
        self.prob_matrix = pro_matrix

    def compute_kl(self):
        # computing kl divergence is resource intensive
        # kl is computed only on inference 
        num_cls = self.prior_alpha.shape[0]
        # each alpha should be >0
        pr_distribution = Dirichlet(self.prior_alpha + torch.eye(num_cls).to(self.prior_alpha.device))
        post_distribution = Dirichlet(self.posterior_alpha + torch.eye(num_cls).to(self.posterior_alpha.device))
        kl = kl_divergence(post_distribution, pr_distribution).sum()
        # kl = kl_divergence(post_distribution, pr_distribution).mean()
        # delete the diagonal
        # prior_alpha_d = torch.tril(self.prior_alpha, diagonal=-1)[:,0:-1]+torch.triu(self.prior_alpha, diagonal=1)[:,1:]
        # posterior_alpha_d = torch.tril(self.posterior_alpha, diagonal=-1)[:,0:-1]+torch.triu(self.posterior_alpha, diagonal=1)[:,1:]
        # pr_distribution_d = Dirichlet(prior_alpha_d)
        # post_distribution_d = Dirichlet(posterior_alpha_d)
        # kl_d = kl_divergence(post_distribution_d, pr_distribution_d).sum()
        self.kl = torch.tanh(torch.abs(kl))
        # self.kl = torch.tanh(kl)




