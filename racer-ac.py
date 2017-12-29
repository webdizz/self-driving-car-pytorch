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
is_monitor = False

env = gym.make('flashgames.CoasterRacer-v0')
env.configure(fps=5.0, vnc_kwargs={
    'encoding': 'tight', 'compress_level': 0,
    'fine_quality_level': 50, 'subsample_level': 3})

torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['action', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=4, stride=2)
        self.conv_drop = nn.Dropout2d()

        self.affine1 = nn.Linear(256, 256)
        self.affine2 = nn.Linear(256, 128)
        self.affine3 = nn.Linear(128, 128)

        self.action_head = nn.Linear(128, 3)
        self.value_head = nn.Linear(128, 1)

        if is_train:
            self.train()

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if is_train:
            x = F.relu(self.conv_drop(self.conv3(x)))
        else:
            x = F.relu(self.conv3(x))

        # flattening the last convolutional layer into this 1D vector x
        x = x.view(-1, 256)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))

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
    value_losses = []
    policy_losses = []

    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / \
        (rewards.std() + np.finfo(np.float32).eps)
    for (action, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0, 0]
        value_losses.append(F.smooth_l1_loss(
            value, Variable(torch.Tensor([r]))))
        policy_losses.append(torch.Tensor(
            [-1 * reward * (action.data[0, 0] + 0.001)]))

    optimizer.zero_grad()
    loss = torch.cat(policy_losses).sum() + torch.cat(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def finish_episode():
    learn()
    # save model
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, args.model_file)


model_store_step = 500
for i_episode in count(1):
    state = env.reset()
    for t in range(20000):  # Don't infinite loop while learning
        action = select_action(state)
        state, reward, done, info = env.step(action)
        env.render()

        if(state[0] != None and is_train):
            # track values for later stats
            model.rewards.append(reward[0])
            if(is_monitor and 'rewarder.profile' in info['n'][0]):
                if('reward_parser.score.last_score' in info['n'][0]['rewarder.profile']['gauges']):
                    track(dbclient, 'reward', info['n'][0]['rewarder.profile']
                          ['counters']['agent_conn.reward']['total'])
                    score = info['n'][0]['rewarder.profile']['gauges']['reward_parser.score.last_score']['value']
                    track(dbclient, 'score', score)

            # save model each model_store_step
            if(t % model_store_step == 0):
                finish_episode()
