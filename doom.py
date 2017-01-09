import ppaquette_gym_doom
from collections import deque
import ppaquette_gym_doom.wrappers
import numpy.random as npr
import numpy as np
import random
import gym
from gym import wrappers
import prettytensor as pt
import tensorflow as tf
import cv2

ENV = gym.make('ppaquette/DoomBasic-v0')
ENV = wrappers.SkipWrapper(3)(ENV)
wrapper = ppaquette_gym_doom.wrappers.action_space.ToDiscrete('minimal')
ENV = wrapper(ENV)
X, Y = 320, 240
wrapper = ppaquette_gym_doom.wrappers.observation_space.SetResolution('%sx%s'%(X, Y))
XY = X * Y
ENV = wrapper(ENV)
crop_X_l, crop_X_u = 95, 120

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class DoomAgent(object):
    def __init__(self, restore = False):
        patch_size = 3
        patch = patch_size
        num_channels = 1
        depth = 4
        num_hidden = 32
	num_labels = 3

        self.episode_states_rgb = tf.placeholder(tf.float32, [None, crop_X_u - crop_X_l, X, num_channels])
        self.policy_net, _ = (pt.wrap(self.episode_states_rgb)
            .conv2d(patch, depth, stride = (2,2), batch_normalize=True)
            .conv2d(patch, depth, stride = (2,2), batch_normalize=True)
            .flatten()
            .dropout(0.5)
            .fully_connected(num_hidden, activation_fn=tf.nn.relu)
            .softmax_classifier(3))

        self.actions = tf.placeholder(tf.uint8, [None], 'actions')
        self.actions_one_hot = tf.one_hot(self.actions, 3, on_value = 1.0, off_value = 0.0, axis = -1)
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'rewards')

        self.per_action_log_prob = tf.reduce_sum(tf.mul(self.policy_net, self.actions_one_hot), reduction_indices = [1])
        self.entropy = tf.reduce_mean(self.policy_net * tf.log(self.policy_net), reduction_indices = [1])
        self.entropy = tf.reduce_mean(self.entropy)
        self.loss = -tf.reduce_mean(self.rewards * tf.log(self.per_action_log_prob)) + 1 * self.entropy

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=5)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)
        if restore:
            print 'saver restore'
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./chckpoints'))

    def start_episode(self):
        self.episode_rewards = []
        self.episode_observations = []
        self.episode_actions = []

    def feedback(self, reward):
        self.episode_rewards.append(reward)

    def end_episode(self):
        rewards = np.array(self.episode_rewards)
        rewards = (np.cumsum(rewards[::-1])[::-1])  * 0.01
        rewards = rewards.reshape((rewards.shape[0], 1))
        observations = np.array(self.episode_observations)
        actions = np.array(self.episode_actions)

        self.sess.run(self.train_step, feed_dict={self.episode_states_rgb: observations, self.rewards: rewards, self.actions: actions})

    def transform_input(self, ob):
        #obBGR = cv2.cvtColor(ob, cv2.COLOR_RGB2BGR)
        #cv2.imshow('before', obBGR)
        ob = cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY)
        ob = ob[crop_X_l:crop_X_u, :]
        #image = cv2.resize(image[0], (width, height), interpolation=cv2.INTER_AREA)

        #cv2.imshow('after', ob)
        #raw_input()
        #exit(0)
        return ob.reshape((ob.shape[0], ob.shape[1], 1))

    def act(self, ob, dump = False):
        ob = self.transform_input(ob)
        ob_as_batch = ob.reshape((1, ob.shape[0], ob.shape[1], ob.shape[2]))
        [softmax] = self.sess.run([self.policy_net], feed_dict={self.episode_states_rgb: ob_as_batch})

        softmax = softmax[0, :]
        if dump:
            print softmax
        try:
            action = npr.multinomial(1, softmax).argmax()
        except:
            action = softmax.argmax()

        return action + 1 #no NOOP
    def record_action(self, action, ob):
        self.episode_actions.append(action - 1)
        ob = self.transform_input(ob)
        self.episode_observations.append(ob)

def learn(episodes, visualisation_freq, restore, monitor):
    outdir = './output/'
    agent = DoomAgent(restore)
    env = ENV
    if monitor:
        env = wrappers.Monitor(ENV, './output/', force=True)
    for episode in xrange(episodes):
        done = False
        ob = env.reset()
        agent.start_episode()
        j = 0
        last_action = 0
        while not done:
            dump = (episode + 1) % visualisation_freq == 0
            action = agent.act(ob, dump)
            last_action = action
            agent.record_action(action, ob)
            new_ob, reward, done, info = env.step(action)
            agent.feedback(reward)
            if done:
                break
            ob = new_ob
            j += 1

        print info['TOTAL_REWARD']
        agent.end_episode()
        if (episode + 1) % visualisation_freq == 0:
            agent.saver.save(agent.sess, 'chckpoints/model', global_step=episode)

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--monitor', type=bool, default=False)
    parser.add_argument('--dump_freq', type=int, default=100)
    args = parser.parse_args()

    print args
    learn(50000, args.dump_freq, args.restore, args.monitor)
