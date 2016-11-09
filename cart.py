import numpy as np
import numpy.random as npr
import gym
import tensorflow as tf
ENV = gym.make('CartPole-v0')

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def play_visualise(W):
    total_reward = 0
    observation = ENV.reset()
    j = 0
    while j < 200:
        j += 1
        ENV.render()
        p_0 = sigmoid(np.dot(observation, W))
        action = 0 if p_0 > 0.5 else 1
        observation, reward, done, info = ENV.step(action)
        total_reward += reward
        if done:
            break
    print("Episode finished with reward {}".format(total_reward))

def do_episode(W):
    rewards = []
    observations = []
    observation = ENV.reset()
    for i in xrange(250):
        p_0 = tf.softmax(np.dot(W, observation))
        if npr.uniform() < p_0:
            action = 0
            pi.append(p_0)
        else:
            action = 1
            pi.append(1 - p_0)
        observation, reward, done, info = ENV.step(action)
        rewards.append(reward)
        if done:
            break
    rewards = np.array(rewards)
    pi = np.array(pi)
    return np.dot(rewards, np.log(pi))

def learn(n_learning_iterations, visualisation_freq):

    states   = tf.placeholder(tf.float32, [None, 4], 'states')
    actions  = tf.placeholder(tf.float32, [None, 1], 'actions')
    rewards  = tf.placeholder(tf.float32, [None, 1], 'rewards')

    W = tf.Variable(tf.zeros([4, 1]), tf.float32)
    tau = tf.constant(0.1, tf.float32, [1])

    action_probs = tf.sigmoid(tf.matmul(states, W))
    regularised_loss = -tf.reduce_sum((rewards - tau - tau * tf.log(action_probs)) * tf.log(action_probs * (1 - actions) + actions * (1 - action_probs)), reduction_indices=[0])

    train_step = tf.train.AdamOptimizer().minimize(regularised_loss)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    for j in xrange(10000): #episodes
        observed_states = []
        observed_rewards = []
        observation = np.array(ENV.reset(), dtype = np.float32)
        observation.resize((1, 4))
        actions_taken = []
        with sess.as_default():
            _W = W.eval()
        for i in xrange(200):
            observed_states.append(observation)
            with sess.as_default():
                p_0 = sigmoid(np.dot(observation, _W))
            action = 0 if npr.uniform() < p_0 else 1
            actions_taken.append(action)
            observation, reward, done, info = ENV.step(action)
            observation.resize((1, 4))
            observed_rewards.append(reward)
            if done:
                break

        observed_states = np.array(observed_states)
        observed_states = observed_states.reshape((observed_states.shape[0], 4))
        observed_rewards = np.array(observed_rewards)
        observed_rewards = np.cumsum(observed_rewards[::-1])[::-1] - 9
        observed_rewards = observed_rewards.reshape((observed_rewards.shape[0], 1))
        actions_taken = np.array(actions_taken)
        actions_taken = actions_taken.reshape((actions_taken.shape[0], 1))
        _, loss, probs = sess.run([train_step, regularised_loss, action_probs], feed_dict={states: observed_states, rewards: observed_rewards, actions: actions_taken})
	if (1 + j) % visualisation_freq == 0:
            with sess.as_default():
                play_visualise(W.eval())

if __name__ == '__main__':
    n_learning_iterations = 100000
    visualisation_freq = 500
    learn(n_learning_iterations, visualisation_freq)
