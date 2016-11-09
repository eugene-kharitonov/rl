import ppaquette_gym_doom
import ppaquette_gym_doom.wrappers
import numpy.random as npr
import numpy as np
import gym
import tensorflow as tf
from PIL import Image

ENV = gym.make('ppaquette/DoomBasic-v0')
wrapper = ppaquette_gym_doom.wrappers.action_space.ToDiscrete('minimal')
ENV = wrapper(ENV)
X, Y = 320, 240
#X, Y = 160, 120
XY = X * Y
wrapper = ppaquette_gym_doom.wrappers.observation_space.SetResolution('%sx%s'%(X, Y))
ENV = wrapper(ENV)
#wrapper = ppaquette_gym_doom.wrappers.control.SetPlayingMode('human')
#ENV = wrapper(ENV)

class DoomAgent(object):
    def __init__(self):
        self.episode_states_rgb = tf.placeholder(tf.float32, [None, Y, X, 3])
        #self.episode_state_grey = tf.image.rgb_to_grayscale(self.episode_states_rgb)
        self.episode_state_grey = self.episode_states_rgb[:,:,:,1] - 128
        batch_size = tf.shape(self.episode_state_grey)[0]
        self.episode_state_as_vector = tf.reshape(self.episode_state_grey, [batch_size, XY])

        #one fully connected layer so far
        W1 = tf.Variable(tf.zeros([XY, 64]))
        b1 = tf.Variable(tf.zeros([64]))
        self.O1 = tf.nn.relu(tf.matmul(self.episode_state_as_vector, W1) + b1)

        #output layer
        W2 = tf.Variable(tf.zeros([64, 4]))
        self.b2 = tf.Variable(tf.zeros([1, 4]))
        self.O2 = tf.nn.softmax(tf.matmul(self.O1, W2) + self.b2)
        #self.O2 = tf.nn.softmax(self.b2)

        self.actions = tf.placeholder(tf.uint8, [None], 'actions') 
        self.actions_one_hot = tf.one_hot(self.actions, 4, on_value = 1.0, off_value = 0.0, axis = -1)
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'rewards')

        self.per_action_log_prob = tf.reduce_sum(self.O2 * self.actions_one_hot, reduction_indices = [1])
        #regularised_loss = -tf.reduce_sum((rewards - tau - tau * tf.log(action_probs)) * tf.log(action_probs * (1 - actions) + actions * (1 - action_probs)), reduction_indices=[0])
        self.loss = -tf.reduce_sum(self.rewards * tf.log(self.per_action_log_prob))
        self.train_step = tf.train.AdamOptimizer(0.01).minimize(self.loss)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def start_episode(self):
        self.episode_rewards = []
        self.episode_observations = []
        self.episode_actions = []

    def feedback(self, reward):
        self.episode_rewards.append(reward)

    def end_episode(self):
        rewards = np.cumsum(np.array(self.episode_rewards)[::-1])[::-1]
        rewards = rewards.reshape((rewards.shape[0], 1))
        observations = np.array(self.episode_observations)
        actions = np.array(self.episode_actions)
        #actions = actions.reshape((actions.shape[0], 1))

        _, loss, o2, b2 = self.sess.run([self.train_step, self.loss, self.O2, self.b2], feed_dict={self.episode_states_rgb: observations, self.rewards: rewards, self.actions: actions})
        #loss, o2, t = self.sess.run([self.loss, self.O2, self.per_action_log_prob], feed_dict={self.episode_states_rgb: observations, self.rewards: rewards, self.actions: actions})
        print 'loss', loss
        #print 'o2', o2 
        #print 'b2', b2

    def act(self, ob):
        ob_as_batch = ob.reshape((1, Y, X, 3))
        [o2, grey_image] = self.sess.run([self.O2, self.episode_state_grey], feed_dict={self.episode_states_rgb: ob_as_batch})
        grey_image = grey_image.reshape((Y, X))
        #print ob.shape
        #print grey_image.shape
        #img = Image.fromarray(grey_image, "F")
        #img.show()
        #exit(0)
        #save('my.png')
        #print self.O2
        action = npr.multinomial(1, o2[0, :]).argmax()
        self.episode_actions.append(action)
        self.episode_observations.append(ob)

        return action

def learn(episodes, max_steps_per_episode, visualisation_freq):
    outdir = './output/'
    agent = DoomAgent()

    for episode in xrange(episodes):
        if (episode + 1) % visualisation_freq == 0:
            ENV.monitor.start(outdir, force=True)
        done = False
        ob = ENV.reset()
        rewards = []
        agent.start_episode()
        j = 0
        while not done and j < 5 * 35: #5 sec
            j += 1
            action = agent.act(ob)
            new_ob, reward, done, info = ENV.step(action)
            agent.feedback(reward)
            ob = new_ob
            if done:
                break
        print info['TOTAL_REWARD']
        agent.end_episode()
        if (episode + 1) % visualisation_freq == 0:
            ENV.monitor.close()

if __name__ == '__main__':
    learn(500, None, 100)
