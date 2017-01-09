# rl
Some experiments with RL, using tensorflow:
 - cart.py -- vanilla policy gradient learning + entropy regularisation for openAI Gym's cartpole problem
 - doom.py -- vanilla policy gradient learning + entropy regularisation for openAI Gym's Doom Basic-0 problem. Two layers of CNN before a fully connected layer. As I learn everything on a MacBook without GPU, I allowed myself to crop the input image to increase the learning speed.

  Here are some examples:

    ![one](https://raw.githubusercontent.com/eugene-kharitonov/rl/master/3.gif)

    ![two](https://raw.githubusercontent.com/eugene-kharitonov/rl/master/4.gif)


