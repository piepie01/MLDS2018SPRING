from agent_dir.agent import Agent
import numpy as np
import random
import torch.nn.functional as functional
import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch import autograd
import torch.nn as nn
import sys
import time
from collections import namedtuple
import math
random.seed(0)

Transition = namedtuple('Transition',('state','action','next_state','reward','done'))

class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self,*args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1)%self.capacity
    def sample(self,batch_size):
        sample = Transition(*zip(*random.sample(self.memory,batch_size)))
        state = list(sample.state)
        action = list(sample.action)
        next_state = list(sample.next_state)
        reward = list(sample.reward)
        done = list(sample.done)
        return state,action,next_state,reward,done
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.conv1 = nn.Conv2d(4,32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=2)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,4)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = x.transpose(-3,-1)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = out.view(-1,1024)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        self.cuda = True
        self.env = env
        self.lr = 1e-4
        self.gamma = 0.999
        self.eps = 1
        self.max_eps = 1
        self.min_eps = 0.025
        self.dec_eps = int(1e5)
        self.max_step = int(3e6)
        self.batch_size = 32
        self.replay_mem = 10000
        self.testmode = args.test_dqn
        self.load = False
        self.savedictpath = 'save/DQN.plk'
        self.loaddictpath = 'save/BEST.plk'
        self.init_model()

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.model.eval()


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.log = open('DQNLOG','w')
        self.replay = ReplayMemory(10000)
        self.state = self.env.reset()
        self.cur_step = 0
        self.num_episode = 0
        self.episode_reward = np.asarray([0])
        self.train_util()


    def make_action(self, state, test=True):
        if not test:
            #self.eps = self.min_eps+(self.max_eps-self.min_eps)*math.exp(-1*self.cur_step/self.dec_eps)
            self.eps = max(1-0.9*self.cur_step/4e5,0.1-0.09*self.cur_step/4e6,0.01)
            explore = random.random() < self.eps
        else:
            self.eps = 0.01
            explore = random.random() < self.eps
            #explore = False
        if explore:
            return random.randint(0,3)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            if self.cuda is True:
                state = state.cuda()
            actions = self.model(Variable(state))
            action = actions.data.max(-1)[1].cpu().numpy()
            return action[0]

    ####My Funtions####
    def init_model(self):
        self.target = DQN()
        self.model = DQN()
        if self.testmode or self.load:
            self.target.load_state_dict(torch.load(self.loaddictpath))
            self.model.load_state_dict(torch.load(self.loaddictpath))
        if self.cuda is True:
            self.target.cuda()
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        #self.optimizer = torch.optim.RMSprop(self.model.parameters(),lr=self.lr)
        #self.criterion = nn.SmoothL1Loss()
        self.criterion = nn.MSELoss()

    def get_state(self):
        act = self.make_action(self.state,False)
        now, reward, done, info = self.env.step(act)
        self.replay.push(self.state,act,now,float(reward),float(done))
        self.state = now
        self.episode_reward[-1]+=reward
        return done

    def train_util(self):
        while self.cur_step < self.max_step:
            for i in range(4):
                done = self.get_state()
                if done:
                    self.state = self.env.reset()
                    self.num_episode+=1
                    print('Episode[%d], Timestep = %d, r = %f, exp = %f'%(self.num_episode,self.cur_step,self.episode_reward[-1],self.eps),file = self.log)
                    self.log.flush()
                    self.episode_reward = np.append(self.episode_reward,0)
            self.cur_step+=1
            if self.cur_step*4 > self.replay_mem:
                self.update_model()
                if self.cur_step % 5000==0:
                    self.target.load_state_dict(self.model.state_dict())
                if self.num_episode==5000:
                    torch.save(self.model.state_dict(),'EARLY.plk')
                if self.num_episode==30000:
                    torch.save(self.model.state_dict(),'HALF.plk')
                    return

    def update_model(self):
        state,action,next_state,reward,done = self.replay.sample(self.batch_size)
        state = torch.FloatTensor(np.array(state))
        action = torch.LongTensor(np.array(action)).view(-1,1)
        next_state = torch.FloatTensor(np.array(next_state))
        reward = torch.FloatTensor(np.array(reward)).view(-1,1)
        done = torch.FloatTensor(np.array(done)).view(-1,1)
        if self.cuda is True:
            state = state.cuda()
            action = action.cuda()
            next_state = next_state.cuda()
            reward = reward.cuda()
            done = done.cuda()
        done = -done+1

        max_action = self.model(Variable(next_state)).max(-1)[1].view(-1,1)
        target_action_value = self.target(Variable(next_state)).gather(1,max_action)
        target_action_value = Variable(reward)+self.gamma*target_action_value*Variable(done)
        cur_action_value = self.model(Variable(state)).gather(1,Variable(action))
        loss = self.criterion(cur_action_value,target_action_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
