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


class StudentNN(nn.Module):
    def __init__(self, num_state_history, num_states, num_actions):
        super(StudentNN, self).__init__()
        num_h1 = 256
        num_h2 = 256
        num_h1_net = 128
        num_h2_net = 128
        num_latent_code = 20  
        self.num_actions = num_actions

        # (66)22*3 -> 256 ->256 -> 20
        self.policy_enc = nn.Sequential(
                        nn.Linear(num_state_history, num_h1),
                        nn.ReLU(),
                        # nn.Linear(num_h1, num_h2),
                        # nn.ReLU(),
                        nn.Linear(num_h1, num_latent_code))
        # 50 (22 + 8 +20) -> 128 ->128 -> 8
        self.policy_net = nn.Sequential(
                        nn.Linear(num_states + num_latent_code, num_h1_net),
                        nn.ReLU(),
                        nn.Linear(num_h1_net, num_h2_net),
                        nn.ReLU(),
                        nn.Linear(num_h2_net, num_actions))

        self.log_std = nn.Parameter(torch.zeros(num_actions))



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


    def forward(self, state_history, states):
        p_latent = self.policy_enc(state_history) 

        if len(p_latent.shape) == 1:
            p_out =  self.policy_net(torch.cat((p_latent, states), dim=0))
        else:
            p_out =  self.policy_net(torch.cat((p_latent, states), dim=1))

        #p_out = MultiVariateNormal(p_out, self.log_std.exp())
    
        #v_out = self.value_net(torch.cat((v_latent, states), dim=1))
        return p_out, p_latent
    
    def get_action(self, state_history,states):
        t = torch.tensor(state_history)
        s = torch.tensor(states)
        p, _ = self.forward(t,s)
        p_ = self._FilterAction(p.cpu().detach().numpy())
        return p_.astype(np.float32)
        
    def load(self, path):
        print('load simulation nn {}'.format(path))
        self.load_state_dict(torch.load(path))

    def save(self, path):
        print('save simulation nn {}'.format(path))
        torch.save(self.state_dict(), path)

# if __name__=="__main__": 
#     model = StudentNN(66,46,8)
#     print(model)
#     model.load("../nn/student_policy/student_current.pt")
#     state_history = torch.rand(1,66)
#     state = torch.rand(1,46)
#     a = model.get_action(state_history, state)
#     print(a)
