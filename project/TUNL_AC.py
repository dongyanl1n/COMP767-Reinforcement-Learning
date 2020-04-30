import gym
from gym import spaces
from gym.utils import seeding
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd, optim, nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from numpy import array
from collections import namedtuple
import matplotlib.pyplot as plt




#===============================
#
#
# Define TUNL environment - without delay
#
#
#================================

class SimpleTunlEnv(object):
    def __init__(self):
      """
      In this simplified version of TUNL environment, we abandoned the delay period. 
      The animal will receive a sample (L or R), and needs to immediately choose the nonmatch location (R or L).
      Correct choice will lead to a reward (r = 1) and finish the episode, while incorrect choice will lead to a punishment (r = -1)
      and animal needs to choose again until correct. 
      
      Observation space:
      [1,0] = left sample
      [0,1] = right sample

      action space:
      0 = choose left
      1 = choose right
      """
      self.observation = None
      self.action_space = spaces.Discrete(2)
      self.observation_space = spaces.MultiBinary(2)
      self.reward = 0
      self.done = False
      self.seed()
    def get_obs(self, sample):
      self.observation = sample
    def step(self, action):
        """
        :param action:
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action)
        if (np.all(self.observation == array([[1,0]])) and action == 0): 
          self.reward = -1
          self.done = False        
        elif (np.all(self.observation == array([[0,1]])) and action == 1):
          self.reward = -1
          self.done = False
        elif (np.all(self.observation == array([[1,0]])) and action == 1):
          self.reward = 1
          self.done = True 
        elif (np.all(self.observation == array([[0,1]])) and action == 0):
          self.reward = 1
          self.done = True
        return self.observation, self.reward, self.done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.observation = None
        self.reward = 0
        self.done = False
  
  

#===============================
#
#
# Define TUNL environment - with delay
#
#
#================================


  class SimpleMemoryTunlEnv(object):
    def __init__(self):
      """
      In this simplified version of TUNL environment, we keep the delay period of 1 time-step length. 
      The animal will receive a sample (L or R), and chooses the nonmatch location (R or L).
      Correct choice will lead to a reward (r = 1) and finish the episode, while incorrect choice will lead to a punishment (r = -1)
      and animal needs to choose again until correct. 
      
      Compared to the two-input environment, this environment does not involve randomness.

      Observation space (s_t):
      [1,0] = left sample
      [0,1] = right sample
      [0,0] = delay period
      [1,1] = waiting for choice

      action space (a_t):
      0 = choose left
      1 = choose right
      """
      self.observation = None
      self.sample = None
      self.action_space = spaces.Discrete(2)
      self.observation_space = spaces.MultiBinary(2)
      self.reward = 0 # reward at each step
      self.done = False
      self.seed()

    def get_obs(self, sample):
      self.sample = sample
      self.observation = sample

    def step(self, action):
        """
        :param action:
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action)
        if ((np.all(self.observation == array([[1,0]])) and action == 0) or (np.all(self.observation == array([[0,1]])) and action == 1)):  # poke sample to initiate
            self.observation = array([[0,0]]) # enter delay
        elif np.all(self.observation == array([[0,0]])):
          self.observation = array([[1,1]])
        elif ((np.all(self.observation == array([[1,1]])) and np.all(self.sample == array([[1,0]])) and action == 0) or (np.all(self.observation == array([[1,1]])) and np.all(self.sample == array([[0,1]])) and action == 1)): # did not choose non-match location
          self.reward = -1
        elif ((np.all(self.observation == array([[1,1]])) and np.all(self.sample == array([[1,0]])) and action == 1) or (np.all(self.observation == array([[1,1]])) and np.all(self.sample == array([[0,1]])) and action == 0)): # chose non-match location
          self.reward = 1
          self.done = True
        return self.observation, self.reward, self.done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.observation = None
        self.sample = None
        self.reward = 0
        self.done = False

#===============================
#
#
# Define TUNL environment - without delay, with only one possible sample (L)
# For testing the validity of LSTM network
#
#
#================================

class SimpleTunlEnv_OneInput(object):
    def __init__(self):
      """
      In this simplified version of TUNL environment, we abandoned the delay period, and simplified the sample phase to one possible input. 
      After each reset, the animal will receive a sample L, and needs to immediately choose the nonmatch location (R).
      Correct choice will lead to a reward (r = 1) and finish the episode, while incorrect choice will lead to a punishment (r = -1)
      and animal needs to choose again until correct. 
      
      Observation space:
      [1,0] = left sample
      [0,1] = right sample

      action space:
      0 = choose left
      1 = choose right
      """
      self.observation = None
      self.action_space = spaces.Discrete(2)
      self.observation_space = spaces.MultiBinary(2)
      self.reward = 0 # reward at each step
      self.done = False
      self.seed()
    def step(self, action):
        """
        :param action:
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action)
        if (np.all(self.observation == array([[1,0]])) and action == 0) or (np.all(self.observation == array([[0,1]])) and action == 1):
          self.reward = -1
          self.done = False
        elif (np.all(self.observation == array([[1,0]])) and action == 1) or (np.all(self.observation == array([[0,1]])) and action == 0):
          self.reward = 1
          self.done = True
        return self.observation, self.reward, self.done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.observation = array([[1,0]])
        self.reward = 0
        self.done = False




#===============================
#
#
# Define actor-critic network and functions for end-of-trial
#
#
#================================






class AC_Net(nn.Module):

    """
    An actor-critic neural network class. Takes sensory inputs and generates a policy and a value estimate.
    """

    def __init__(self, input_dimensions, action_dimensions, batch_size, hidden_types, hidden_dimensions):

        """
        AC_Net(input_dimensions, action_dimensions, hidden_types=[], hidden_dimensions=[])
        Create an actor-critic network class.
        Required arguments:
        - input_dimensions (int): the dimensions of the input space
        - action_dimensions (int): the number of possible actions
        Optional arguments:
        - batch_size (int): the size of the batches (default = 4).
        - hidden_types (list of strings): the type of hidden layers to use, options are 'linear', 'lstm'.
        If list is empty no hidden layers are used (default = []).
        - hidden_dimensions (list of ints): the dimensions of the hidden layers. Must be a list of
                                        equal length to hidden_types (default = []).
        """

        # call the super-class init
        super(AC_Net, self).__init__()

        # store the input dimensions
        self.input_d = input_dimensions

        # check input type
        assert (hidden_types[0] == 'linear' or hidden_types[0] == 'lstm')
        self.input_type = 'vector'

        # store the batch size
        self.batch_size = batch_size

        # check that the correct number of hidden dimensions are specified
        assert len(hidden_types) is len(hidden_dimensions)

        # check whether we're using hidden layers
        if not hidden_types:
            self.layers = [input_dimensions, action_dimensions]
            # no hidden layers, only input to output, create the actor and critic layers
            self.output = nn.ModuleList([
            nn.Linear(input_dimensions, action_dimensions),  # ACTOR
            nn.Linear(input_dimensions, 1)])  # CRITIC
        else:
            # to store a record of the last hidden states
            self.hx = []
            self.cx = []
            # create the hidden layers
            self.hidden = nn.ModuleList()
            for i, htype in enumerate(hidden_types):

                #check if hidden layer type is correct
                assert htype in ['linear', 'lstm']

                #get the input dimensions
                #first hidden layer
                if i is 0:
                   input_d = input_dimensions
                   output_d = hidden_dimensions[i]
                   if htype is 'linear':
                       self.hidden.append(nn.Linear(input_d, output_d))
                       self.hx.append(None)
                       self.cx.append(None)
                   elif htype is 'lstm':
                       self.hidden.append(nn.LSTMCell(input_d, output_d))
                       self.hx.append(Variable(torch.zeros(self.batch_size,output_d)))
                       self.cx.append(Variable(torch.zeros(self.batch_size,output_d)))
                #second hidden layer onwards
                else:
                    input_d = hidden_dimensions[i - 1]
                    # get the output dimension
                    output_d = hidden_dimensions[i]
                    # construct the layer
                    if htype is 'linear':
                       self.hidden.append(nn.Linear(input_d, output_d))
                       self.hx.append(None)
                       self.cx.append(None)
                    elif htype is 'lstm':
                       self.hidden.append(nn.LSTMCell(input_d, output_d))
                       self.hx.append(Variable(torch.zeros(self.batch_size,output_d)))
                       self.cx.append(Variable(torch.zeros(self.batch_size,output_d)))
        # create the actor and critic layers
        self.layers = [input_dimensions]+hidden_dimensions+[action_dimensions]
        self.output = nn.ModuleList([
        nn.Linear(output_d, action_dimensions), #actor
        nn.Linear(output_d, 1)                  #critic
        ])

        # store the output dimensions
        self.output_d = output_d

        # to store a record of actions and rewards
        self.saved_actions = []
        self.rewards = []

    def forward(self, x, temperature=1):
        '''
        forward(x):
        Runs a forward pass through the network to get a policy and value.
        Required arguments:
          - x (torch.Tensor): sensory input to the network, should be of size batch x input_d
        '''

        # check the inputs
        if type(self.input_d) == int:
          assert x.shape[-1] == self.input_d
        elif type(self.input_d) == tuple:
          assert (x.shape[2], x.shape[3], x.shape[1]) == self.input_d
          if not  (isinstance(self.hidden[0],nn.Conv2d) or isinstance(self.hidden[0],nn.MaxPool2d)):
            raise Exception('image to non {} layer'.format(self.hidden[0]))

        # pass the data through each hidden layer
        for i, layer in enumerate(self.hidden):
          # run input through the layer depending on type
          if isinstance(layer, nn.Linear):
            x = F.relu(layer(x))
            lin_activity = x
          elif isinstance(layer, nn.LSTMCell):
            x, cx = layer(x, (self.hx[i], self.cx[i]))
            self.hx[i] = x.clone()
            self.cx[i] = cx.clone()
        # pass to the output layers
        policy = F.softmax(self.output[0](x), dim=1)
        value  = self.output[1](x)
        
        if isinstance(self.hidden[-1], nn.Linear):
          return policy, value, lin_activity
        else:
          return policy, value


    def reinit_hid(self):
        # to store a record of the last hidden states
        self.hx = []
        self.cx = []
      
        for i, layer in enumerate(self.hidden):
          if isinstance(layer, nn.Linear):
            pass
          elif isinstance(layer, nn.LSTMCell):
            self.hx.append(Variable(torch.zeros(self.batch_size,layer.hidden_size)))
            self.cx.append(Variable(torch.zeros(self.batch_size,layer.hidden_size)))
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def select_action(model, policy_, value_):
    a = Categorical(policy_)
    action = a.sample()
    model.saved_actions.append(SavedAction(a.log_prob(action), value_))
    return action.item(), policy_.data[0], value_.item()


def discount_rwds(r, gamma): # takes [1,1,1,1] and makes it [3.439,2.71,1.9,1]
    disc_rwds = np.zeros_like(r).astype(float)
    r_asfloat = r.astype(float)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r_asfloat[t]
        disc_rwds[t] = running_add
    return disc_rwds



def finish_trial(model, discount_factor, optimizer, **kwargs):
    '''
    Finishes a given training trial and backpropagates.
    '''

    # set the return to zero
    R = 0
    returns_ = discount_rwds(np.asarray(model.rewards), gamma=discount_factor) # [1,1,1,1] into [3.439,2.71,1.9,1]
    saved_actions = model.saved_actions

    policy_losses = []
    value_losses = []

    returns_ = torch.Tensor(returns_).clone()

    for (log_prob, value), r in zip(saved_actions, returns_):
        rpe = r - value.item()
        policy_losses.append(-log_prob * rpe)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([[r]]))).unsqueeze(-1))    
    optimizer.zero_grad()
    p_loss = (torch.cat(policy_losses).sum())
    v_loss = (torch.cat(value_losses).sum())
    total_loss = p_loss + v_loss
    total_loss.backward(retain_graph=True)
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]

    return p_loss, v_loss


#===============================
#
#
# Experiment - Random Policy Baseline
#
#
#================================
env = SimpleMemoryTunlEnv()
runs = 10
episodes = 500


#For recording and plotting purposes

result_cum_returns = []
result_avg_returns = [] # average return in an episode, not the average of episode return at a certain time step
result_episode_returns = [] 

results = []

for i_run in range(runs):
  cum_return = 0
  cum_returns = []
  avg_returns = []
  episode_returns = []
  for i_episode in range(episodes): 
      episode_return = 0
      env.seed()
      env.reset()
      env.get_obs(np.expand_dims(np.random.choice(2,2,p=[0.5,0.5], replace=False),axis=0))
      done = False
      while not done:
          observation, reward, done, info = env.step(env.action_space.sample()) # here, reward is either 1 or -1
          episode_return = episode_return + reward # undiscounted episode return
      episode_returns.append(episode_return)
      cum_return = cum_return + episode_return
      cum_returns.append(cum_return)
      avg_returns.append(cum_return / (i_episode + 1))
  result_cum_returns.append(cum_returns)
  result_avg_returns.append(avg_returns)
  result_episode_returns.append(episode_returns)

mean_cum_rts = np.array(result_cum_returns).mean(axis=0)
mean_avg_rts = np.array(result_avg_returns).mean(axis=0)
mean_episode_rts = np.array(result_episode_returns).mean(axis=0)
std_cum_rts = np.array(result_cum_returns).std(axis=0)
std_avg_rts = np.array(result_avg_returns).std(axis=0)
std_episode_rts = np.array(result_episode_returns).std(axis=0)

x = list(range(episodes))

fig, ax = plt.subplots()
ax.plot(x, mean_cum_rts, label='random policy',color = 'deeppink')
ax.fill_between(x, mean_cum_rts, mean_cum_rts + std_cum_rts, color = 'pink')
ax.fill_between(x, mean_cum_rts, mean_cum_rts - std_cum_rts, color = 'pink')
ax.set_xlabel('episode')  
ax.set_ylabel('cumulative return')  
ax.set_title("random policy cumulative undiscounted return") 
ax.legend()

fig, ax = plt.subplots()
ax.plot(x, mean_avg_rts, label='random policy',color = 'deeppink')
ax.fill_between(x, mean_avg_rts, mean_avg_rts + std_avg_rts, color = 'pink')
ax.fill_between(x, mean_avg_rts, mean_avg_rts - std_avg_rts, color = 'pink')
ax.set_xlabel('episode')  
ax.set_ylabel('average undiscounted return per episode')  
ax.set_title("random policy average undiscounted return per episode") 
ax.legend()

fig, ax = plt.subplots()
ax.plot(x, mean_episode_rts, label='random policy',color = 'deeppink')
ax.fill_between(x, mean_episode_rts, mean_episode_rts + std_episode_rts, color = 'pink')
ax.fill_between(x, mean_episode_rts, mean_episode_rts - std_episode_rts, color = 'pink')
ax.set_xlabel('episode')  
ax.set_ylabel('undiscounted return of each episode')  
ax.set_title("random policy undiscounted return of each episode") 
ax.legend()


#===============================
#
#
# Experiment - without delay
#
#
#================================



env = SimpleTunlEnv()
net_lstm = AC_Net(2,2,1,['lstm','linear'],[2,2]) 
net_fc = AC_Net(2,2,1,['linear','linear'],[2,2])
optimizer_lstm = torch.optim.SGD(net_lstm.parameters(), lr=0.01, momentum=0.9)
optimizer_fc = torch.optim.SGD(net_fc.parameters(), lr=0.01, momentum=0.9)

runs = 10
episodes = 500


#For recording and plotting purposes
result_cum_returns_lstm = []
result_cum_returns_fc = []
result_avg_returns_lstm = [] # average return in an episode, not the average of episode return at a certain time step
result_avg_returns_fc = [] 
result_episode_returns_lstm = [] 
result_episode_returns_fc = []

results = []

for i_run in range(runs):
  cum_return_lstm = 0
  cum_returns_lstm = []
  cum_return_fc = 0
  cum_returns_fc = []
  avg_returns_lstm = []
  avg_returns_fc = []
  episode_returns_lstm = []
  episode_returns_fc = []
  for i_episode in range(episodes): # one episode = one sample; finishes when either network is "done" (i.e. chooses correctly)
      done_lstm = False
      done_fc = False
      episode_return_lstm = 0
      episode_return_fc = 0
      env.seed()
      env.reset()
      env.get_obs(np.expand_dims(np.random.choice(2,2,p=[0.5,0.5], replace=False),axis=0))
      obs_lstm = torch.as_tensor(env.observation)
      obs_fc = torch.as_tensor(env.observation)
      while (not done_fc) and (not done_lstm):
        pol_lstm, val_lstm, lin_act_lstm = net_lstm.forward(obs_lstm.float()) # lin_act_lstm needed if the last layer is linear
        pol_fc, val_fc, lin_act_fc = net_fc.forward(obs_fc.float())
        act_lstm, p_lstm, v_lstm = select_action(net_lstm, pol_lstm, val_lstm)
        act_fc, p_fc, v_fc = select_action(net_fc, pol_fc, val_fc)
        new_obs_lstm, reward_lstm, done_lstm, info_lstm = env.step(act_lstm) #reward_lstm = 1 or -1
        new_obs_fc, reward_fc, done_fc, info_fc = env.step(act_fc)
        net_lstm.rewards.append(reward_lstm) # net.rewards = [-1, -1, 1] etc.
        net_fc.rewards.append(reward_fc)
        episode_return_lstm += reward_lstm # undiscounted episode return, equals to sum of rewards
        episode_return_fc += reward_fc
        obs_lstm = torch.as_tensor(new_obs_lstm)
        obs_fc = torch.as_tensor(new_obs_fc)
      episode_returns_lstm.append(episode_return_lstm)
      episode_returns_fc.append(episode_return_fc)
      p_loss_lstm, v_loss_lstm = finish_trial(net_lstm, 0.99, optimizer_lstm)
      p_loss_fc, v_loss_fc = finish_trial(net_fc, 0.99, optimizer_fc)
      cum_return_lstm += episode_return_lstm # cumulative undiscounted return for all episodes
      cum_return_fc += episode_return_fc
      cum_returns_lstm.append(cum_return_lstm) # cumulative undiscounted return, for plotting
      cum_returns_fc.append(cum_return_fc)
      avg_returns_fc.append(cum_return_fc / (i_episode + 1)) # avg undiscounted return per episode,for plotting
      avg_returns_lstm.append(cum_return_lstm / (i_episode + 1))
  result_cum_returns_lstm.append(cum_returns_lstm)
  result_cum_returns_fc.append(cum_returns_fc)
  result_avg_returns_lstm.append(avg_returns_lstm) # average return in an episode, not the average of episode return at a certain time step
  result_avg_returns_fc.append(avg_returns_fc)
  result_episode_returns_lstm.append(episode_returns_lstm)
  result_episode_returns_fc.append(episode_returns_fc) 

mean_cum_rts_lstm = np.array(result_cum_returns_lstm).mean(axis=0)
mean_cum_rts_fc = np.array(result_cum_returns_fc).mean(axis=0)
mean_avg_rts_lstm = np.array(result_avg_returns_lstm).mean(axis=0)
mean_avg_rts_fc = np.array(result_avg_returns_fc).mean(axis=0)
mean_episode_rts_lstm = np.array(result_episode_returns_lstm).mean(axis=0)
mean_episode_rts_fc = np.array(result_episode_returns_fc).mean(axis=0)
std_cum_rts_lstm = np.array(result_cum_returns_lstm).std(axis=0)
std_cum_rts_fc = np.array(result_cum_returns_fc).std(axis=0)
std_avg_rts_lstm = np.array(result_avg_returns_lstm).std(axis=0)
std_avg_rts_fc = np.array(result_avg_returns_fc).std(axis=0)
std_episode_rts_lstm = np.array(result_episode_returns_lstm).std(axis=0)
std_episode_rts_fc = np.array(result_episode_returns_fc).std(axis=0)

x = list(range(episodes))

fig, ax = plt.subplots()
ax.plot(x, mean_cum_rts_lstm, label='lstm',color = 'royalblue')
ax.plot(x, mean_cum_rts_fc, label='fully connected',color = 'orange')
ax.fill_between(x, mean_cum_rts_lstm, mean_cum_rts_lstm + std_cum_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_cum_rts_lstm, mean_cum_rts_lstm - std_cum_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_cum_rts_fc, mean_cum_rts_fc + std_cum_rts_fc, color = 'bisque')
ax.fill_between(x, mean_cum_rts_fc, mean_cum_rts_fc - std_cum_rts_fc, color = 'bisque')
ax.set_xlabel('episode')  
ax.set_ylabel('cumulative return')  
ax.set_title("cumulative undiscounted return") 
ax.legend()

fig, ax = plt.subplots()
ax.plot(x, mean_avg_rts_lstm, label='lstm',color = 'royalblue')
ax.plot(x, mean_avg_rts_fc, label='fully connected',color = 'orange')
ax.fill_between(x, mean_avg_rts_lstm, mean_avg_rts_lstm + std_avg_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_avg_rts_lstm, mean_avg_rts_lstm - std_avg_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_avg_rts_fc, mean_avg_rts_fc + std_avg_rts_fc, color = 'bisque')
ax.fill_between(x, mean_avg_rts_fc, mean_avg_rts_fc - std_avg_rts_fc, color = 'bisque')
ax.set_xlabel('episode')  
ax.set_ylabel('average undiscounted return per episode')  
ax.set_title("average undiscounted return per episode") 
ax.legend()

fig, ax = plt.subplots()
ax.plot(x, mean_episode_rts_lstm, label='lstm',color = 'royalblue')
ax.plot(x, mean_episode_rts_fc, label='fully connected',color = 'orange')
ax.fill_between(x, mean_episode_rts_lstm, mean_episode_rts_lstm + std_episode_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_episode_rts_lstm, mean_episode_rts_lstm - std_episode_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_episode_rts_fc, mean_episode_rts_fc + std_episode_rts_fc, color = 'bisque')
ax.fill_between(x, mean_episode_rts_fc, mean_episode_rts_fc - std_episode_rts_fc, color = 'bisque')
ax.set_xlabel('episode')  
ax.set_ylabel('undiscounted return of each episode')  
ax.set_title("undiscounted return of each episode") 
ax.legend()

#===============================
#
#
# Experiment - with delay
#
#
#================================

env = SimpleMemoryTunlEnv()
net_lstm = AC_Net(2,2,1,['lstm','linear'],[2,2]) 
net_fc = AC_Net(2,2,1,['linear','linear'],[2,2])
optimizer_lstm = torch.optim.SGD(net_lstm.parameters(), lr=0.01, momentum=0.9)
optimizer_fc = torch.optim.SGD(net_fc.parameters(), lr=0.01, momentum=0.9)

runs = 10
episodes = 500 


#For recording and plotting purposes
result_cum_returns_lstm = []
result_cum_returns_fc = []
result_avg_returns_lstm = [] # average return in an episode, not the average of episode return at a certain time step
result_avg_returns_fc = [] 
result_episode_returns_lstm = [] 
result_episode_returns_fc = []

results = []

for i_run in range(runs):
  cum_return_lstm = 0
  cum_returns_lstm = []
  cum_return_fc = 0
  cum_returns_fc = []
  avg_returns_lstm = []
  avg_returns_fc = []
  episode_returns_lstm = []
  episode_returns_fc = []
  for i_episode in range(episodes): # one episode = one sample; finishes when either network is "done" (i.e. chooses correctly)
      done_lstm = False
      done_fc = False
      episode_return_lstm = 0
      episode_return_fc = 0
      env.seed()
      env.reset()
      env.get_obs(np.expand_dims(np.random.choice(2,2,p=[0.5,0.5], replace=False),axis=0))
      obs_lstm = torch.as_tensor(env.observation)
      obs_fc = torch.as_tensor(env.observation)
      while (not done_fc) and (not done_lstm):
        pol_lstm, val_lstm, lin_act_lstm = net_lstm.forward(obs_lstm.float()) # lin_act_lstm needed if the last layer is linear
        pol_fc, val_fc, lin_act_fc = net_fc.forward(obs_fc.float())
        act_lstm, p_lstm, v_lstm = select_action(net_lstm, pol_lstm, val_lstm)
        act_fc, p_fc, v_fc = select_action(net_fc, pol_fc, val_fc)
        new_obs_lstm, reward_lstm, done_lstm, info_lstm = env.step(act_lstm) #reward_lstm = 1 or -1
        new_obs_fc, reward_fc, done_fc, info_fc = env.step(act_fc)
        net_lstm.rewards.append(reward_lstm) # net.rewards = [-1, -1, 1] etc.
        net_fc.rewards.append(reward_fc)
        episode_return_lstm += reward_lstm # undiscounted episode return, equals to sum of rewards
        episode_return_fc += reward_fc
        obs_lstm = torch.as_tensor(new_obs_lstm)
        obs_fc = torch.as_tensor(new_obs_fc)
      episode_returns_lstm.append(episode_return_lstm)
      episode_returns_fc.append(episode_return_fc)
      p_loss_lstm, v_loss_lstm = finish_trial(net_lstm, 0.99, optimizer_lstm)
      p_loss_fc, v_loss_fc = finish_trial(net_fc, 0.99, optimizer_fc)
      cum_return_lstm += episode_return_lstm # cumulative undiscounted return for all episodes
      cum_return_fc += episode_return_fc
      cum_returns_lstm.append(cum_return_lstm) # cumulative undiscounted return, for plotting
      cum_returns_fc.append(cum_return_fc)
      avg_returns_fc.append(cum_return_fc / (i_episode + 1)) # avg undiscounted return per episode,for plotting
      avg_returns_lstm.append(cum_return_lstm / (i_episode + 1))
  result_cum_returns_lstm.append(cum_returns_lstm)
  result_cum_returns_fc.append(cum_returns_fc)
  result_avg_returns_lstm.append(avg_returns_lstm) # average return in an episode, not the average of episode return at a certain time step
  result_avg_returns_fc.append(avg_returns_fc)
  result_episode_returns_lstm.append(episode_returns_lstm)
  result_episode_returns_fc.append(episode_returns_fc) 

mean_cum_rts_lstm = np.array(result_cum_returns_lstm).mean(axis=0)
mean_cum_rts_fc = np.array(result_cum_returns_fc).mean(axis=0)
mean_avg_rts_lstm = np.array(result_avg_returns_lstm).mean(axis=0)
mean_avg_rts_fc = np.array(result_avg_returns_fc).mean(axis=0)
mean_episode_rts_lstm = np.array(result_episode_returns_lstm).mean(axis=0)
mean_episode_rts_fc = np.array(result_episode_returns_fc).mean(axis=0)
std_cum_rts_lstm = np.array(result_cum_returns_lstm).std(axis=0)
std_cum_rts_fc = np.array(result_cum_returns_fc).std(axis=0)
std_avg_rts_lstm = np.array(result_avg_returns_lstm).std(axis=0)
std_avg_rts_fc = np.array(result_avg_returns_fc).std(axis=0)
std_episode_rts_lstm = np.array(result_episode_returns_lstm).std(axis=0)
std_episode_rts_fc = np.array(result_episode_returns_fc).std(axis=0)

x = list(range(episodes))

fig, ax = plt.subplots()
ax.plot(x, mean_cum_rts_lstm, label='lstm',color = 'royalblue')
ax.plot(x, mean_cum_rts_fc, label='fully connected',color = 'orange')
ax.fill_between(x, mean_cum_rts_lstm, mean_cum_rts_lstm + std_cum_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_cum_rts_lstm, mean_cum_rts_lstm - std_cum_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_cum_rts_fc, mean_cum_rts_fc + std_cum_rts_fc, color = 'bisque')
ax.fill_between(x, mean_cum_rts_fc, mean_cum_rts_fc - std_cum_rts_fc, color = 'bisque')
ax.set_xlabel('episode')  
ax.set_ylabel('cumulative return')  
ax.set_title("cumulative undiscounted return") 
ax.legend()

fig, ax = plt.subplots()
ax.plot(x, mean_avg_rts_lstm, label='lstm',color = 'royalblue')
ax.plot(x, mean_avg_rts_fc, label='fully connected',color = 'orange')
ax.fill_between(x, mean_avg_rts_lstm, mean_avg_rts_lstm + std_avg_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_avg_rts_lstm, mean_avg_rts_lstm - std_avg_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_avg_rts_fc, mean_avg_rts_fc + std_avg_rts_fc, color = 'bisque')
ax.fill_between(x, mean_avg_rts_fc, mean_avg_rts_fc - std_avg_rts_fc, color = 'bisque')
ax.set_xlabel('episode')  
ax.set_ylabel('average undiscounted return per episode')  
ax.set_title("average undiscounted return per episode") 
ax.legend()

fig, ax = plt.subplots()
ax.plot(x, mean_episode_rts_lstm, label='lstm',color = 'royalblue')
ax.plot(x, mean_episode_rts_fc, label='fully connected',color = 'orange')
ax.fill_between(x, mean_episode_rts_lstm, mean_episode_rts_lstm + std_episode_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_episode_rts_lstm, mean_episode_rts_lstm - std_episode_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_episode_rts_fc, mean_episode_rts_fc + std_episode_rts_fc, color = 'bisque')
ax.fill_between(x, mean_episode_rts_fc, mean_episode_rts_fc - std_episode_rts_fc, color = 'bisque')
ax.set_xlabel('episode')  
ax.set_ylabel('undiscounted return of each episode')  
ax.set_title("undiscounted return of each episode") 
ax.legend()

#===============================
#
#
# Experiment - Without delay, one input
# For testing the validity of LSTM network
#
#
#================================

env = SimpleTunlEnv_OneInput()
net_lstm = AC_Net(2,2,1,['lstm','linear'],[2,2]) 
net_fc = AC_Net(2,2,1,['linear','linear'],[2,2])
optimizer_lstm = torch.optim.SGD(net_lstm.parameters(), lr=0.01, momentum=0.9)
optimizer_fc = torch.optim.SGD(net_fc.parameters(), lr=0.01, momentum=0.9)

runs = 10
episodes = 500 


#For recording and plotting purposes
result_cum_returns_lstm = []
result_cum_returns_fc = []
result_avg_returns_lstm = [] # average return in an episode, not the average of episode return at a certain time step
result_avg_returns_fc = [] 
result_episode_returns_lstm = [] 
result_episode_returns_fc = []

results = []

for i_run in range(runs):
  cum_return_lstm = 0
  cum_returns_lstm = []
  cum_return_fc = 0
  cum_returns_fc = []
  avg_returns_lstm = []
  avg_returns_fc = []
  episode_returns_lstm = []
  episode_returns_fc = []
  for i_episode in range(episodes): # one episode = one sample; finishes when either network is "done" (i.e. chooses correctly)
      done_lstm = False
      done_fc = False
      episode_return_lstm = 0
      episode_return_fc = 0
      env.reset()
      obs = torch.as_tensor(env.observation)
      obs_lstm = obs
      obs_fc = obs
      while (not done_fc) and (not done_lstm):
        pol_lstm, val_lstm, lin_act_lstm = net_lstm.forward(obs_lstm.float()) # lin_act_lstm needed if the last layer is linear
        pol_fc, val_fc, lin_act_fc = net_fc.forward(obs_fc.float())
        act_lstm, p_lstm, v_lstm = select_action(net_lstm, pol_lstm, val_lstm)
        act_fc, p_fc, v_fc = select_action(net_fc, pol_fc, val_fc)
        new_obs_lstm, reward_lstm, done_lstm, info_lstm = env.step(act_lstm) #reward_lstm = 1 or -1
        new_obs_fc, reward_fc, done_fc, info_fc = env.step(act_fc)
        net_lstm.rewards.append(reward_lstm) # net.rewards = [-1, -1, 1] etc.
        net_fc.rewards.append(reward_fc)
        episode_return_lstm += reward_lstm # undiscounted episode return, equals to sum of rewards
        episode_return_fc += reward_fc
        obs_lstm = torch.as_tensor(new_obs_lstm)
        obs_fc = torch.as_tensor(new_obs_fc)
      episode_returns_lstm.append(episode_return_lstm)
      episode_returns_fc.append(episode_return_fc)
      p_loss_lstm, v_loss_lstm = finish_trial(net_lstm, 0.99, optimizer_lstm)
      p_loss_fc, v_loss_fc = finish_trial(net_fc, 0.99, optimizer_fc)
      cum_return_lstm += episode_return_lstm # cumulative undiscounted return for all episodes
      cum_return_fc += episode_return_fc
      cum_returns_lstm.append(cum_return_lstm) # cumulative undiscounted return, for plotting
      cum_returns_fc.append(cum_return_fc)
      avg_returns_fc.append(cum_return_fc / (i_episode + 1)) # avg undiscounted return per episode,for plotting
      avg_returns_lstm.append(cum_return_lstm / (i_episode + 1))
  result_cum_returns_lstm.append(cum_returns_lstm)
  result_cum_returns_fc.append(cum_returns_fc)
  result_avg_returns_lstm.append(avg_returns_lstm) # average return in an episode, not the average of episode return at a certain time step
  result_avg_returns_fc.append(avg_returns_fc)
  result_episode_returns_lstm.append(episode_returns_lstm)
  result_episode_returns_fc.append(episode_returns_fc) 

mean_cum_rts_lstm = np.array(result_cum_returns_lstm).mean(axis=0)
mean_cum_rts_fc = np.array(result_cum_returns_fc).mean(axis=0)
mean_avg_rts_lstm = np.array(result_avg_returns_lstm).mean(axis=0)
mean_avg_rts_fc = np.array(result_avg_returns_fc).mean(axis=0)
mean_episode_rts_lstm = np.array(result_episode_returns_lstm).mean(axis=0)
mean_episode_rts_fc = np.array(result_episode_returns_fc).mean(axis=0)
std_cum_rts_lstm = np.array(result_cum_returns_lstm).std(axis=0)
std_cum_rts_fc = np.array(result_cum_returns_fc).std(axis=0)
std_avg_rts_lstm = np.array(result_avg_returns_lstm).std(axis=0)
std_avg_rts_fc = np.array(result_avg_returns_fc).std(axis=0)
std_episode_rts_lstm = np.array(result_episode_returns_lstm).std(axis=0)
std_episode_rts_fc = np.array(result_episode_returns_fc).std(axis=0)

x = list(range(episodes))

fig, ax = plt.subplots()
ax.plot(x, mean_cum_rts_lstm, label='lstm',color = 'royalblue')
ax.plot(x, mean_cum_rts_fc, label='fully connected',color = 'orange')
ax.fill_between(x, mean_cum_rts_lstm, mean_cum_rts_lstm + std_cum_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_cum_rts_lstm, mean_cum_rts_lstm - std_cum_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_cum_rts_fc, mean_cum_rts_fc + std_cum_rts_fc, color = 'bisque')
ax.fill_between(x, mean_cum_rts_fc, mean_cum_rts_fc - std_cum_rts_fc, color = 'bisque')
ax.set_xlabel('episode')  
ax.set_ylabel('cumulative return')  
ax.set_title("cumulative undiscounted return") 
ax.legend()

fig, ax = plt.subplots()
ax.plot(x, mean_avg_rts_lstm, label='lstm',color = 'royalblue')
ax.plot(x, mean_avg_rts_fc, label='fully connected',color = 'orange')
ax.fill_between(x, mean_avg_rts_lstm, mean_avg_rts_lstm + std_avg_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_avg_rts_lstm, mean_avg_rts_lstm - std_avg_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_avg_rts_fc, mean_avg_rts_fc + std_avg_rts_fc, color = 'bisque')
ax.fill_between(x, mean_avg_rts_fc, mean_avg_rts_fc - std_avg_rts_fc, color = 'bisque')
ax.set_xlabel('episode')  
ax.set_ylabel('average undiscounted return per episode')  
ax.set_title("average undiscounted return per episode") 
ax.legend()

fig, ax = plt.subplots()
ax.plot(x, mean_episode_rts_lstm, label='lstm',color = 'royalblue')
ax.plot(x, mean_episode_rts_fc, label='fully connected',color = 'orange')
ax.fill_between(x, mean_episode_rts_lstm, mean_episode_rts_lstm + std_episode_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_episode_rts_lstm, mean_episode_rts_lstm - std_episode_rts_lstm, color = 'cornflowerblue')
ax.fill_between(x, mean_episode_rts_fc, mean_episode_rts_fc + std_episode_rts_fc, color = 'bisque')
ax.fill_between(x, mean_episode_rts_fc, mean_episode_rts_fc - std_episode_rts_fc, color = 'bisque')
ax.set_xlabel('episode')  
ax.set_ylabel('undiscounted return of each episode')  
ax.set_title("undiscounted return of each episode") 
ax.legend()
