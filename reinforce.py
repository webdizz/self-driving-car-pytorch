import argparse
import logging
import cv2

import gym
import universe
from universe.wrappers import BlockingReset

import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()
reward_threshold = 100

LOGGING_FORMAT = '%(asctime)s - %(name)s - %(thread)d|%(process)d - %(levelname)s - %(message)s'
logging.basicConfig(format=LOGGING_FORMAT)
logger = logging.getLogger('Car')


env = gym.make('flashgames.CoasterRacer-v0')


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.affine1 = nn.Linear(32 * 5 * 7, 128)
        self.affine2 = nn.Linear(128, 3)

        self.train()

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        # flattening the last convolutional layer into this 1D vector x
        x = x.view(-1, 32 * 5 * 7)
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def resize_frame(frame):
    # crop and resize by 4. If we resize directly we lose pixels that
    # aren't close enough to the pixel boundary.
    x1 = 36
    y1 = 116
    x2 = 640
    y2 = 536
    # cv2.imwrite('/tmp/universe-frame-original.jpg', processed_frame)
    processed_frame = frame[y1:y2, x1:x2]
    # reduce by 2 in 2 times
    y = processed_frame.shape[0]
    x = processed_frame.shape[1]
    ratio = 100.0 / x
    processed_frame = cv2.resize(processed_frame, (100, int(y * ratio)))
    # cv2.imwrite('/tmp/universe-frame-cropped.jpg', processed_frame)
    # after crop shape is 69x100
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2GRAY)
    # cv2.imwrite('/tmp/universe-frame-gray.jpg', processed_frame)
    logger.debug("Current frame shape after preprocessing is {}".format(
        processed_frame.shape))
    processed_frame = np.reshape(processed_frame, [1, 69, 100])
    return processed_frame


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
        probs = policy(Variable(actual_state))
        action = probs.multinomial()
        policy.saved_actions.append(action)
        action_idx = action.data.numpy()[0][0]
        action = drive_actions[action_idx]

    return [action]


def finish_episode():
    R = 0
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / \
        (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(policy.saved_actions, rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(policy.saved_actions, [
                      None for _ in policy.saved_actions])
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]


running_reward = 10
for i_episode in count(1):
    state = env.reset()
    for t in range(10000):  # Don't infinite loop while learning
        action = select_action(state)
        state, reward, done, info = env.step(action)
        env.render()
        policy.rewards.append(reward[0])
        if reward[0] > 0:
            print("Got a reward {} for action {} while state is {}".format(
                reward, action, info))

    running_reward = running_reward * 0.99 + t * 0.01
    finish_episode()
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            i_episode, t, running_reward))
    if running_reward > reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break
