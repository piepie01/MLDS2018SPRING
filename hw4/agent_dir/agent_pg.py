from agent_dir.agent import Agent
import scipy
import scipy.misc
import numpy as np
import random
import torch.nn.functional as functional
import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch import autograd
import torch.nn as nn
from torch.distributions import Categorical
import sys
import time
def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)

def get_random_action():
    return random.randint(2,3) #2:UP, 3:DOWN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6400, 256)
        self.fc2 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, state):
        out = state.view(state.size()[0],-1)
        out = self.fc1(out)
        out = functional.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################

        self.lr = 1e-4
        self.batch_size = 32
        self.cuda = False
        self.reward_discount = 0.99
        self.savedictpath = 'save/model0.th'
        self.bestmodel = 'save/best.th'
        self.best = 0
        self.check_epoch = 2500


        self.env = env
        self.load = False
        #self.log = open('log.txt','w')
        self.log = sys.stdout
        self.render_dir = 'save/'

        if self.cuda:
            self.net = Net().cuda()
        else:
            self.net = Net()
        self.opt = torch.optim.Adam(self.net.parameters(), lr = self.lr)
        #self.opt = torch.optim.RMSprop(self.net.parameters(), lr = self.lr)


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        if not self.load:
            self.net.load_state_dict(torch.load(self.savedictpath))
            self.net.eval()
            self.load = True
            random.seed(2217)
            np.random.seed(2217)
            if self.cuda:
                torch.cuda.manual_seed(7122)
            else:
                torch.manual_seed(127)
        self.last = None

        pass
    def _get_episode(self):
        now_time = time.time()
        print('Start gaming !!! [time : {}]'.format(time.asctime(time.localtime(now_time))),file = self.log)
        last = self.env.reset()
        last = prepro(last)
        act = get_random_action()
        now, reward, done, info = self.env.step(act)
        now = prepro(now)
        whole_reward = []
        whole_action = []
        score = 0
        opponent_score = 0
        up_or_down = [0,0]
        while not done:
            pic = now - last
            last = now
            act, proba = self.get_action(pic)
            up_or_down[act-2]+=1
            now, reward, done, info = self.env.step(act)
            whole_reward.append(reward)
            whole_action.append(proba)
            now = prepro(now)
            if reward == 1:
                score+=1
                if len(whole_reward) > 7000:
                    break
            elif reward == -1:
                opponent_score+=1
                if len(whole_reward) > 7000:
                    break
        for i in list(range(len(whole_reward)))[::-1]:
            if whole_reward[i] == 0:
                whole_reward[i] = whole_reward[i+1] * self.reward_discount
        whole_reward = np.array(whole_reward)
        std, mean = np.std(whole_reward), np.mean(whole_reward)
        whole_reward = ( whole_reward - mean ) / std
        #print(whole_reward)
        #exit()
        print('Episode Done, Score : {}/{}, time cost : {}'.format(score, opponent_score, time.time() - now_time), file = self.log)
        print('length : {}, [up, down] : {}'.format(len(whole_reward), up_or_down), file = self.log)
        #print(observe)
        return whole_action, whole_reward, score, opponent_score
    def train(self):
        epoch = 0
        while True:
            print('----------[epoch {}]--------'.format(epoch), file = self.log)
            self.opt.zero_grad()
            act, reward, score, opponent_score = self._get_episode()
            reward = Variable(torch.FloatTensor(reward))
            loss = torch.cat(act)
            loss = torch.mul(loss, reward)
            loss = torch.mean(loss)
            loss.backward()
            self.opt.step()
            print('[ Loss : {} ]'.format(loss.data[0]/10),file = self.log)
            epoch+=1
            self.log.flush()
            if score > opponent_score:
                self.checkpoint()
                #import pexpect
                #test_model = pexpect.spawn('python3 test.py --test_pg')
                #test_model.expect(pexpect.EOF, timeout = None)
                #ret = test_model.before.decode('ascii')
                #mean = float(ret.split('Mean: ')[1].split()[0])
                #print('Mean : ',mean, file = self.log)
                #if mean > self.best:
                #    self.best = mean
                #    torch.save(self.net.state_dict(), self.bestmodel)



    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        observation = prepro(observation)
        if self.last is None:
            self.last = observation
            return get_random_action()
        else:
            pic = observation - self.last
            self.last = observation
            act, proba = self.get_action(pic)
            return act
        return self.env.get_random_action()
    def get_action(self, state):
        #trans_state = view_state(state)
        inp = torch.FloatTensor(state)
        inp = inp.view(1,1, inp.size()[0], inp.size()[1])
        out = self.predict(inp)
        m = Categorical(out)
        action = m.sample()
        log_prob = -m.log_prob(action)
        act = action.cpu().data[0]
        return act+2, log_prob
    def predict(self, inp):
        if self.cuda:
            inp = Variable(inp.cuda())
        else:
            inp = Variable(inp)
        out = self.net(inp)
        return out
    def checkpoint(self):
        if self.savedictpath is not None:
            torch.save(self.net.state_dict(), self.savedictpath)
    def render_state(self, state):
        from PIL import Image
        for i,txt in enumerate(state[::4]):
            im = Image.fromarray(txt.reshape(80,80)).convert('RGBA')
            im.save(self.render_dir+str(i)+'.png')
            #print(txt.shape)
