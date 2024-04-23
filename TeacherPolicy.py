import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math
import time
from collections import OrderedDict
import numpy as np
from IPython import embed
import action_filter

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(self, val).sum(-1, keepdim=True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()


class MuscleNN(nn.Module):
    def __init__(self, num_total_muscle_related_dofs, num_dofs, num_muscles):
        super(MuscleNN, self).__init__()
        self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
        self.num_dofs = num_dofs
        self.num_muscles = num_muscles

        num_h1 = 1024
        num_h2 = 512
        num_h3 = 512
        self.fc = nn.Sequential(
            nn.Linear(num_total_muscle_related_dofs + num_dofs, num_h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h1, num_h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h2, num_h3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h3, num_muscles),
            nn.Tanh(),
            nn.ReLU()
        )
        self.std_muscle_tau = torch.zeros(self.num_total_muscle_related_dofs)
        self.std_tau = torch.zeros(self.num_dofs)

        for i in range(self.num_total_muscle_related_dofs):
            self.std_muscle_tau[i] = 200.0

        for i in range(self.num_dofs):
            self.std_tau[i] = 200.0
        if use_cuda:
            self.std_tau = self.std_tau.cuda()
            self.std_muscle_tau = self.std_muscle_tau.cuda()
            self.cuda()
        self.fc.apply(weights_init)

    def forward(self, muscle_tau, tau):
        muscle_tau = muscle_tau / self.std_muscle_tau

        tau = tau / self.std_tau
        out = self.fc.forward(torch.cat([muscle_tau, tau], dim=1))
        return out

    def load(self, path):
        print('load muscle nn {}'.format(path))
        self.load_state_dict(torch.load(path))

    def save(self, path):
        print('save muscle nn {}'.format(path))
        torch.save(self.state_dict(), path)

    def get_activation(self, muscle_tau, tau):
        act = self.forward(Tensor(muscle_tau.reshape(1, -1)), Tensor(tau.reshape(1, -1)))
        return act.cpu().detach().numpy()


class TeacherNN(nn.Module):
    def __init__(self, num_targets, num_states, num_actions):
        super(TeacherNN, self).__init__()

        num_h1 = 256
        num_h2 = 256
        num_h1_net = 128
        num_h2_net = 128
        num_latent_code = 20  
        self.num_actions = num_actions

        # 84 (6*14) -> 256 ->256 -> 20
        self.policy_enc = nn.Sequential(
                        nn.Linear(num_targets, num_h1),
                        nn.ReLU(),
                        # nn.Linear(num_h1, num_h2),
                        # nn.ReLU(),
                        nn.Linear(num_h1, num_latent_code))
        # 50 (22 + 8 +20) -> 128 ->128 -> 8
        self.policy_net = nn.Sequential(
                        nn.Linear(num_latent_code + num_states, num_h1_net),
                        nn.ReLU(),
                        nn.Linear(num_h1_net, num_h2_net),
                        nn.ReLU(),
                        nn.Linear(num_h2_net, num_actions))



        # self.p_fc1 = nn.Linear(num_states, num_h1)
        # self.p_fc2 = nn.Linear(num_h1, num_h2)
        # self.p_fc3 = nn.Linear(num_h2, num_actions)
        self.log_std = nn.Parameter(torch.zeros(num_actions))

        # 84 (16*4) -> 256 ->256 -> 20
        # self.value_enc = nn.Sequential(
        #                 nn.Linear(num_targets, num_h1),
        #                 nn.ReLU(),
        #                 nn.Linear(num_h1, num_h2),
        #                 nn.ReLU(),
        #                 nn.Linear(num_h2, num_latent_code))

        num_h1 = 256
        num_h2 = 256
        # 114 (22 + 8 +84) -> 256 ->256 -> 8
        self.value_net = nn.Sequential(
                        nn.Linear(num_targets + num_states, num_h1),
                        nn.ReLU(),
                        nn.Linear(num_h1, num_h2),
                        nn.ReLU(),
                        nn.Linear(num_h2, 1))


        # self.reward_container = Container(10000)
        #torch.nn.init.xavier_uniform_(self.policy_enc)
        #torch.nn.init.xavier_uniform_(self.policy_net)
        #torch.nn.init.xavier_uniform_(self.p_fc1.weight)
        #torch.nn.init.xavier_uniform_(self.p_fc2.weight)
        #torch.nn.init.xavier_uniform_(self.p_fc3.weight)

        #self.p_fc1.bias.data.zero_()
        #self.p_fc2.bias.data.zero_()
        #self.p_fc3.bias.data.zero_()

        #torch.nn.init.xavier_uniform_(self.v_fc1.weight)
        #torch.nn.init.xavier_uniform_(self.v_fc2.weight)
        #torch.nn.init.xavier_uniform_(self.v_fc3.weight)

        #self.v_fc1.bias.data.zero_()
        #self.v_fc2.bias.data.zero_()
        #self.v_fc3.bias.data.zero_()
        self._initialize_weights()

        self._action_filter = self._BuildActionFilter()
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


    def _BuildActionFilter(self):
        sampling_rate = 30
        num_joints = self.num_actions
        a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate, num_joints=num_joints)
        return a_filter

    def _FilterAction(self, action):
        if sum(self._action_filter.xhist[0])[0] == 0:
            self._action_filter.init_history(action)
        a =  self._action_filter.filter(action)
        return a 


    def forward(self, targets, states):
        p_latent = self.policy_enc(targets) 
    
        if len(targets.shape) == 1:
            p_out =  self.policy_net(torch.cat((p_latent, states), dim=0))
            # p_out =  self.policy_net(torch.cat((targets, states), dim=0))
            v_out =  self.value_net(torch.cat((targets, states), dim=0))
        else:
            p_out =  self.policy_net(torch.cat((p_latent, states), dim=1))
            # p_out =  self.policy_net(torch.cat((targets, states), dim=1))
            v_out =  self.value_net(torch.cat((targets, states), dim=1))

        p_out = MultiVariateNormal(p_out, self.log_std.exp())
           
        #v_out = self.value_net(torch.cat((v_latent, states), dim=1))
        return p_out, v_out, p_latent
    
    def get_policy_output(self, targets, states):
        p_latent = self.policy_enc(targets) 
        if len(targets.shape) == 1:
            p_out =  self.policy_net(torch.cat((p_latent, states), dim=0))
        else:
            p_out =  self.policy_net(torch.cat((p_latent, states), dim=1))

        return p_out, p_latent


    def load(self, path):
        print('load simulation nn {}'.format(path))
        self.load_state_dict(torch.load(path))

    def save(self, path):
        print('save simulation nn {}'.format(path))
        torch.save(self.state_dict(), path)

    def get_action(self,targets,states):
        t = torch.tensor(targets)
        s = torch.tensor(states)

        p, _, _ = self.forward(t,s)

        p_ = self._FilterAction(p.loc.cpu().detach().numpy())
        return p_.astype(np.float32)
        
        # p = p.loc.cpu().detach().numpy()
        # return self._FilterAction(p)

    def get_random_action(self, s):
        ts = torch.tensor(s)
        p, _, _ = self.forward(ts)
        return p.sample().cpu().detach().numpy()
