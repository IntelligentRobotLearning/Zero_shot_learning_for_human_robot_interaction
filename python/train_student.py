import math
import random
import time
import os
import sys
from datetime import datetime

import collections
from collections import namedtuple
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import copy
import collections
import numpy as np
from pymss import EnvManager
from IPython import embed
from TeacherPolicy import *
import action_filter
import time

from StudentPolicy import StudentNN 


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
Episode = namedtuple('Episode',('s','a','r', 'value', 'logprob'))
class EpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def Push(self, *args):
		self.data.append(Episode(*args))
	def Pop(self):
		self.data.pop()
	def GetData(self):
		return self.data
MuscleTransition = namedtuple('MuscleTransition',('JtA','tau_des','L','b'))
class MuscleBuffer(object):
	def __init__(self, buff_size = 10000):
		super(MuscleBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def Push(self,*args):
		self.buffer.append(MuscleTransition(*args))

	def Clear(self):
		self.buffer.clear()
Transition = namedtuple('Transition',('s','a', 'logprob', 'TD', 'GAE'))
class ReplayBuffer(object):
	def __init__(self, buff_size = 10000):
		super(ReplayBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def Push(self,*args):
		self.buffer.append(Transition(*args))

	def Clear(self):
		self.buffer.clear()
class PPO_Student(object):
	def __init__(self,meta_file):
		np.random.seed(seed=int(time.time()))
		self.num_slaves = 16  # origin: 16
		self.env = EnvManager(meta_file,self.num_slaves)
		self.use_muscle = self.env.UseMuscle()
		self.use_symmetry = self.env.UseSymmetry()

		self.num_current_state = self.env.GetNumState()  # get number of current state observation

		self.num_target = self.env.GetNumFutureTargetmotions()
		self.num_state_history = self.env.GetNumStateHistory() + self.env.GetNumRootInfo()
		self.num_state = self.env.GetNumFullObservation() - self.num_target - self.num_state_history   # current state and action history 
		if(self.use_symmetry):
			self.num_action = int(self.env.GetNumAction()/2)
		else:
			self.num_action = int(self.env.GetNumAction())
		self.num_muscles = self.env.GetNumMuscles()
		self.save_path = ''

		self.num_epochs = 10
		self.num_epochs_muscle = 3
		self.num_evaluation = 0
		self.num_tuple_so_far = 0
		self.num_episode = 0
		self.num_tuple = 0
		self.num_simulation_Hz = self.env.GetSimulationHz()
		self.num_control_Hz = self.env.GetControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.gamma = 0.99
		self.lb = 0.99

		self.batch_size = 256
		self.muscle_batch_size = 128
		self.buffer_size = 2048   # origin: 2048
		self.replay_buffer = ReplayBuffer(30000)
		self.muscle_buffer = MuscleBuffer(30000)

		# num_target:  6 step future target motions
		self.teacher = TeacherNN(self.num_target, self.num_state,self.num_action)  # teacher model 

		self.student = StudentNN(self.num_state_history, self.num_state,self.num_action) # student model 
		#self.loss_mse = nn.MSELoss(size_average=True)
		print(self.teacher)
		print(self.student)

		if not self.use_muscle:
			self.num_muscles = 1  # original = 0, set it to 1 as temporary value
		self.muscle_model = MuscleNN(self.env.GetNumTotalMuscleRelatedDofs(),self.num_action,self.num_muscles)
		if use_cuda:
			self.teacher.cuda()
			self.muscle_model.cuda()
			self.student.cuda()

		self.default_learning_rate = 3E-4
		self.default_clip_ratio = 0.2
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio
		self.optimizer = optim.Adam(self.student.parameters(),lr=self.learning_rate)
		self.optimizer_muscle = optim.Adam(self.muscle_model.parameters(),lr=self.learning_rate)
		self.max_iteration = 120000

		self.w_entropy = -0.001

		self.sum_loss = 0.0
		self.sum_latent_loss=0.0
		self.sum_cnt = 0 

		self.avg_loss = []
		self.avg_latent_loss = []

		self.sum_return = 0.0


		self.max_return = -1.0
		self.max_return_epoch = 1
		self.tic = time.time()

		self.episodes = [None]*self.num_slaves
		for j in range(self.num_slaves):
			self.episodes[j] = EpisodeBuffer()

		self._action_filter = [self._BuildActionFilter() for _i in range(self.num_slaves)]
		self.env.Resets(True)

		# self._history_buffer_state = collections.deque(maxlen=3)
		# self._history_buffer_action = collections.deque(maxlen=3)
		# self._last_action = self.env.GetActions()
		#
		# while len(self._history_buffer_state) < 3:
		# 	self._history_buffer_state.appendleft(self.env.GetStates())
		# while len(self._history_buffer_action) < 3:
		# 	self._history_buffer_action.appendleft(self.env.GetActions())

	def _BuildActionFilter(self):
		sampling_rate = self.num_control_Hz #1 / (self.time_step * self._action_repeat)
		num_joints = self.num_action
		a_filter = action_filter.ActionFilterButter(
			sampling_rate=sampling_rate, num_joints=num_joints)
		return a_filter

	def _ResetActionFilter(self):
		for filter in self._action_filter:
			filter.reset()
		return

	def _FilterAction(self, action):
		# initialize the filter history, since resetting the filter will fill
		# the history with zeros and this can cause sudden movements at the start
		# of each episode
		for i in range(action.shape[0]):
			if sum(self._action_filter[i].xhist[0])[0] == 0:
				self._action_filter[i].init_history(action[i])

		filtered_action = []
		for i in range(action.shape[0]):
			filtered_action.append(self._action_filter[i].filter(action[i]))
		return np.vstack(filtered_action)

	def SetSaveModePath(self, path):
		self.save_path = path

	def SaveModel(self, path):
		if not os.path.exists('../nn/'+path):
			os.makedirs('../nn/'+path)
			print('create folder ../nn/'+path)

		self.student.save('../nn/'+path+'/student_current.pt')
		# self.muscle_model.save('../nn/'+path+'/current_muscle.pt')
		
		if self.max_return_epoch == self.num_evaluation:
			self.student.save('../nn/'+path+'/student_max.pt')
			# self.muscle_model.save('../nn/'+path+'/max_muscle.pt')
		if self.num_evaluation%100 == 0:
			self.student.save('../nn/'+ path + '/student_'+str(self.num_evaluation//100)+'.pt')
			# self.muscle_model.save('../nn/'+ path + '/'+ str(self.num_evaluation//100)+'_muscle.pt')

	def LoadModel(self,path):
		print('load')
		self.teacher.load('../nn/'+path+'.pt')
		self.muscle_model.load('../nn/'+path+'_muscle.pt')

	def LoadTeacherModel(self,path):
		print('load')
		self.teacher.load(path)

	def LoadStudentModel(self,path):
		print('load')
		self.student.load(path)

	def ComputeTDandGAE(self):
		self.replay_buffer.Clear()
		self.muscle_buffer.Clear()
		self.sum_return = 0.0

		for epi in self.total_episodes:
			data = epi.GetData()
			size = len(data)
			if size == 0:
				continue
			states, actions, rewards, values, logprobs = zip(*data)

			values = np.concatenate((values, np.zeros(1)), axis=0)
			advantages = np.zeros(size)
			ad_t = 0
			if np.any(np.isnan(rewards)):
				print("here")

			epi_return = 0.0
			for i in reversed(range(len(data))):
				epi_return += rewards[i]
				delta = rewards[i] + values[i+1] * self.gamma - values[i]
				ad_t = delta + self.gamma * self.lb * ad_t
				advantages[i] = ad_t
			#self.sum_return += epi_return
			TD = values[:size] + advantages
			if np.any(np.isnan(TD)):
				print("here")

			for i in range(size):
				self.replay_buffer.Push(states[i], actions[i], logprobs[i], TD[i], advantages[i])
		self.num_episode = len(self.total_episodes)
		self.num_tuple = len(self.replay_buffer.buffer)
		print('SIM : {}'.format(self.num_tuple))
		self.num_tuple_so_far += self.num_tuple

		muscle_tuples = self.env.GetMuscleTuples()
		for i in range(len(muscle_tuples)):
			self.muscle_buffer.Push(muscle_tuples[i][0],muscle_tuples[i][1],muscle_tuples[i][2],muscle_tuples[i][3])


	def GenerateTransitions(self):
		self.total_episodes = []
		# states = [None]*self.num_slaves
		# actions = [None]*self.num_slaves
		rewards = [None]*self.num_slaves
		states = self.env.GetFullObservations()
		# targets = self.env.GetTargetObservations()
		local_step = 0
		terminated = [False]*self.num_slaves
		counter = 0
		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')

			_state = Tensor(states[:,self.num_state_history:-self.num_target]) 
			_target = Tensor(states[:,-self.num_target:])
			a_dist,v,latent = self.teacher(_target,_state)

			actions = a_dist.sample().cpu().detach().numpy()

			# actions = a_dist.loc.cpu().detach().numpy()
			logprobs = a_dist.log_prob(Tensor(actions)).cpu().detach().numpy().reshape(-1)
			values = v.cpu().detach().numpy().reshape(-1)
			filterd_actions = self._FilterAction(actions).astype(np.float32)
			# if abs(filterd_actions-actions).mean() > 1e-5:
			# 	print(abs(filterd_actions-actions).mean())
			if(self.use_symmetry):
				actions_full = np.concatenate((filterd_actions, filterd_actions), axis=1)
			else:
				actions_full = filterd_actions
	
			self.env.SetActions(actions_full)
			self.env.UpdateActionBuffers(actions_full)  # update action buffer

			self.env.StepsAtOnce()  # 20 step with same action
			self.env.UpdateStateBuffers()  # update state buffer

			for j in range(self.num_slaves):
				nan_occur = False	
				terminated_state = True

				if np.any(np.isnan(states[j])) or np.any(np.isnan(actions[j])) or np.any(np.isnan(states[j])) or np.any(np.isnan(values[j])) or np.any(np.isnan(logprobs[j])):
					# print("State")
					# print(states[j])
					# print("Actions")
					# print(actions[j])
					# print("Values")
					# print(values[j])
					# print("logprobs")
					# print(logprobs[j])
					nan_occur = True
				
				elif self.env.IsEndOfEpisode(j) is False:
					terminated_state = False
					rewards[j] = self.env.GetReward(j)
					if rewards[j] > 1000000:
						print(rewards[j])

					self.episodes[j].Push(states[j], actions[j], rewards[j], values[j], logprobs[j])
					local_step += 1

				if terminated_state or (nan_occur is True):
				#if (nan_occur is True):	
					if (nan_occur is True):
						self.episodes[j].Pop()
					self.total_episodes.append(self.episodes[j])
					self.episodes[j] = EpisodeBuffer()
					self.env.Reset(True, j) # state/action buffer is reset too.
					self._action_filter[j].init_history(self.env.GetAction(j)[:self.num_action])

			if local_step >= self.buffer_size:
				break

			states = self.env.GetFullObservations()
		
	def OptimizeStudent(self):
		all_transitions = np.array(self.replay_buffer.buffer)
		for j in range(self.num_epochs):
			np.random.shuffle(all_transitions)
			for i in range(len(all_transitions)//self.batch_size):
				transitions = all_transitions[i*self.batch_size:(i+1)*self.batch_size]
				batch = Transition(*zip(*transitions))

				stack_s = np.vstack(batch.s).astype(np.float32)
				_state = stack_s[:,self.num_state_history:-self.num_target]
				_target = stack_s[:,-self.num_target:]
				_state_history = stack_s[:,:self.num_state_history]
		
				stack_a = np.vstack(batch.a).astype(np.float32)
				stack_lp = np.vstack(batch.logprob).astype(np.float32)
				stack_td = np.vstack(batch.TD).astype(np.float32)
				stack_gae = np.vstack(batch.GAE).astype(np.float32)

				with torch.no_grad():
					a_teacher, l_teacher = self.teacher.get_policy_output(Tensor(_target),Tensor(_state))
				a_student, l_student = self.student(Tensor(_state_history),Tensor(_state))

				'''MSE loss'''
				loss = (F.mse_loss(a_teacher, a_student, True) + F.mse_loss(l_teacher, l_student, True))/self.batch_size
				latent_loss = (F.mse_loss(l_teacher, l_student, True))/self.batch_size
				self.optimizer.zero_grad()
				loss.backward(retain_graph=True)
				for param in self.student.parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5,0.5)
				self.optimizer.step()
				self.sum_loss += loss.item()
				self.sum_latent_loss += latent_loss.item()
				self.sum_cnt += 1 
			print('Optimizing sim nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')

	def OptimizeModel(self):
		self.ComputeTDandGAE()
		self.OptimizeStudent()
		# if self.use_muscle:
		# 	self.OptimizeMuscleNN()
		
	def Train(self):
		print("Start Generating Transitions.")
		start = time.process_time()
		self.GenerateTransitions()
		print("GenerateTransitions: {:.2f}s".format(time.process_time() - start))

		print("Start Optimizing Model.")
		start = time.process_time()
		self.OptimizeModel()
		print("OptimizeModel: {:.2f}s".format(time.process_time() - start))
	
	def Evaluate(self):
		self.num_evaluation = self.num_evaluation + 1
		h = int((time.time() - self.tic)//3600.0)
		m = int((time.time() - self.tic)//60.0)
		s = int((time.time() - self.tic))
		m = m - h*60
		s = int((time.time() - self.tic))
		s = s - h*3600 - m*60
		if self.num_episode is 0:
			self.num_episode = 1
		if self.num_tuple is 0:
			self.num_tuple = 1

		print('# {} === {}h:{}m:{}s ==='.format(self.num_evaluation,h,m,s))		
		print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
		print('||Num Transition           : {}'.format(self.num_tuple))
		print('||Num Episode              : {}'.format(self.num_episode))
		print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple/self.num_episode))

		self.SaveModel(self.save_path)
		
		print('=============================================')
		if (self.sum_cnt==0):
			self.avg_loss.append(0)
			self.avg_latent_loss.append(0)
		else:
			self.avg_loss.append(self.sum_loss / self.sum_cnt)
			self.avg_latent_loss.append(self.sum_latent_loss / self.sum_cnt)
		self.sum_loss = 0.0
		self.sum_latent_loss = 0.0
		self.sum_cnt = 0 
		return np.array(self.avg_loss),np.array(self.avg_latent_loss)

import matplotlib
import matplotlib.pyplot as plt

plt.ion()

def Plot(y,title, file_name, num_fig=1,ylim=True):
	temp_y = np.zeros(y.shape)
	if y.shape[0]>5:
		temp_y[0] = y[0]
		temp_y[1] = 0.5*(y[0] + y[1])
		temp_y[2] = 0.3333*(y[0] + y[1] + y[2])
		temp_y[3] = 0.25*(y[0] + y[1] + y[2] + y[3])
		for i in range(4,y.shape[0]):
			temp_y[i] = np.sum(y[i-4:i+1])*0.2

	plt.figure(num_fig)
	plt.clf()
	# plt.title(title)
	# plt.plot(y,'b', label='True value')	
	plt.plot(temp_y,'b', label='Moving average')
	plt.xlabel('Timestep')
	plt.ylabel(title)
	# plt.legend(loc='upper right',fontsize=11)
	

	plt.show()
	if ylim:
		plt.ylim([0,1])
	plt.pause(0.001)
	plt.savefig('{}.png'.format(file_name),  bbox_inches='tight') 

def Plot_Error(y,title, file_name, num_fig=1,ylim=True):

	plt.figure(num_fig)
	plt.clf()
	plt.title(title)
	plt.plot(y,'g')
	
	plt.show()
	if ylim:
		plt.ylim([0,1])
	plt.pause(0.001)
	plt.savefig('{}.png'.format(file_name),  bbox_inches='tight') 




import argparse
import os
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model',help='model path')
	parser.add_argument('-ms', '--model_student', help='model path')
	parser.add_argument('-d','--meta',help='meta file')
	parser.add_argument('-r','--dir', help='save path')

	args =parser.parse_args()
	if args.meta is None:
		print('Provide meta file')
		exit()
	

	ppo = PPO_Student(args.meta)

	nn_dir = '../nn'
	
	if not os.path.exists(nn_dir):
		os.makedirs(nn_dir)
	
	if args.dir is not None:
		ppo.SetSaveModePath(args.dir)
		if not os.path.exists('../nn/'+args.dir):
			os.makedirs('../nn/'+args.dir)
			print('Created directory ../nn/' + args.dir)

	if args.model is None:
		raise ValueError("Trained teacher policy should be provided.")
	ppo.LoadTeacherModel(args.model)
	# ppo.LoadStudentModel(args.model_student)

	print('num states: {}, num actions: {}'.format(ppo.env.GetNumState(),ppo.env.GetNumAction()))

	for i in range(ppo.max_iteration-5):
		ppo.Train()
		avg_loss, avg_latent_loss = ppo.Evaluate()
		Plot(avg_loss,'Student training loss', "avg_loss", 0,False) # x:iteration(num_evaluation)
		Plot(avg_latent_loss,'Student & teacher latent loss', "avg_latent_loss", 1,False)

