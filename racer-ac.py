import argparse
import os.path as path

import gym
import universe

import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from reframe import resize_frame
from monitor import dbclient
from monitor import track

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--model_file', type=str, default='racer-ac.pth.tar', metavar='F',
                    help='file to save/restore model (default: racer-ac.pth.tar)')
args = parser.parse_args()

is_train = True
is_load_model = True

env = gym.make('flashgames.CoasterRacer-v0')

torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['action', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv_drop = nn.Dropout2d()

        self.affine1 = nn.Linear(192, 256)
        self.affine2 = nn.Linear(256, 128)

        self.action_head = nn.Linear(128, 3)
        self.value_head = nn.Linear(128, 1)

        if is_train:
            self.train()

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv_drop(self.conv4(x)), 2))

        # flattening the last convolutional layer into this 1D vector x
        x = x.view(-1, 192)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        # rescaling them so that the elements of the n-dimensional output Tensor lie in the range (0,1) and sum to 1
        return F.softmax(action_scores * 0.7), state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)

if (path.exists(args.model_file) and is_load_model):
    persisted_model_state = torch.load(args.model_file)
    model.load_state_dict(persisted_model_state['model'])
    model.eval()
    optimizer.load_state_dict(persisted_model_state['optimizer'])


def select_action(state):
    # define our turns or keyboard actions
    left = [('KeyEvent', 'ArrowUp', True), ('KeyEvent',
                                            'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)]
    right = [('KeyEvent', 'ArrowUp', True), ('KeyEvent',
                                             'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]
    forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent',
                                               'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
    drive_actions = [left, right, forward]

    if(state[0] == None):
        # by default we go forward first
        action = forward
    else:
        actual_state = state[0]['vision']
        actual_state = resize_frame(actual_state)
        actual_state = torch.from_numpy(actual_state).float().unsqueeze(0)
        # predict next action
        probs, state_value = model(Variable(actual_state))
        # multinomial probability distribution located in the corresponding row of Tensor input
        action = probs.multinomial()
        if(is_train):
            model.saved_actions.append(SavedAction(action, state_value))
        # to drive action
        action_idx = action.data.numpy()[0][0]
        action = drive_actions[action_idx]

    return [action]


def learn():
    R = 0
    saved_actions = model.saved_actions
    value_loss = 0
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / \
        (rewards.std() + np.finfo(np.float32).eps)
    for (action, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0, 0]
        reinforce_reward = torch.Tensor([[reward]])
        action.reinforce(reinforce_reward)
        value_loss += F.smooth_l1_loss(value, Variable(torch.Tensor([r])))
    optimizer.zero_grad()
    final_nodes = [value_loss] + list(map(lambda p: p.action, saved_actions))
    gradients = [torch.ones(1)] + [None] * len(saved_actions)
    autograd.backward(final_nodes, gradients, retain_graph=False)
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def finish_episode():
    learn()
    # save model
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, args.model_file)


running_reward = 10
model_store_step = 20000
for i_episode in count(1):
    state = env.reset()
    for t in range(20000):  # Don't infinite loop while learning
        action = select_action(state)
        state, reward, done, info = env.step(action)
        env.render()

        if(state[0] != None and is_train):
            model.rewards.append(reward[0])
            # track values for later stats
            track(dbclient, 'reward', reward[0])
            if('rewarder.profile' in info['n'][0]):
                if('reward_parser.score.last_score' in info['n'][0]['rewarder.profile']['gauges']):
                    score = info['n'][0]['rewarder.profile']['gauges']['reward_parser.score.last_score']['value']
                    track(dbclient, 'score', score)

        if(state[0] != None and is_train and len(model.rewards) >= model_store_step):
            finish_episode()

    running_reward = running_reward * 0.99 + t * 0.01
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            i_episode, t, running_reward))
