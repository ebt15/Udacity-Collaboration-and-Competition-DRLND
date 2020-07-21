# **REPORT**

# Project III : Collaboration & Competition

## Project Description

### 1. Environment

In this environment called Tennis, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space of an agent consists of 8 variables. Each action is a 2D vector with entries corresponding to movement toward (or away from) the net, and jumping.

- **State space** is a `2x24` dimensional continuous tensor, consisting of positions, and velocities of each agent across 3 frames.

- **Action space** is `2x2` dimentional continuous tensor, corresponding actions for each agent. Every entry in the action tensor should be a number between -1 and 1.


## Approach

To train the Tennis agents I used MDDGP algorithm implementing centralized training and decentralized execution. Each agent receives its own, local observation. Thus it will be possible to simultaneously train both agents through self-play. During training, the critic for each agent uses extra information like states observed and actions
taken by all other agents. For the actor, there is one for each agent. Each actor has access to only its agents observations and actions. During execution time only actors are present and all observations and actions are used.

MADDPG (Multi-agent DDPG) class uses 2 DDPG agents similar to what was used in Udacity classroom for previous projects. Also ReplayBuffer is used as a shared buffer between agents. MADDPG combines states, actions, rewards, next_states, dones from both agents and adds them to shared ReplayBuffer. MADDPG act calls act for 2 DDPG agents.


### 1. Establish Baseline

Before building agents that learn, I started by testing ones that select actions (uniformly) at random at each time step.

Running the random agents a few times resulted in scores from 0 to 0.02. Obviously, if these agents need to achieve an average score of 0.5 over 100 consecutive episodes, then choosing actions at random won't work. However, when you watch the agents acting randomly, it becomes clear that these types of sporadic actions can be useful early in the training process. That is, they can help the agents explore the action space to find some signal of good vs. bad actions. This insight will come into play later when we implement the Ornstein-Uhlenbeck process and epsilon noise decay.



### 2. Implement Learning Algorithm
To get started, there are a few high-level architecture decisions we need to make. First, we need to determine which types of algorithms are most suitable for the Tennis environment.

#### Policy-based vs Value-based Methods
There are two key differences in the Tennis environment compared to the ['Navigation'](https://github.com/tommytracey/DeepRL-P1-Navigation) environment from two projects ago:
1. **Multiple agents** &mdash; The Tennis environment has 2 different agents, whereas the Navigation project had only a single agent.
2. **Continuous action space** &mdash; The action space is now _continuous_, which allows each agent to execute more complex and precise movements. Even though each tennis agent can only move forward, backward, or jump, there's an unlimited range of possible action values that control these movements. Whereas, the agent in the Navigation project was limited to four _discrete_ actions: left, right, forward, backward.

Given the additional complexity of this environment, the **value-based method** we used for the Navigation project is not suitable &mdash; i.e., the Deep Q-Network (DQN) algorithm. Most importantly, we need an algorithm that allows the tennis agent to utilize its full range and power of movement. For this, we'll need to explore a different class of algorithms called **policy-based methods**.

Here are some advantages of policy-based methods:
- **Continuous action spaces** &mdash; Policy-based methods are well-suited for continuous action spaces.
- **Stochastic policies** &mdash; Both value-based and policy-based methods can learn deterministic policies. However, policy-based methods can also learn true stochastic policies.
- **Simplicity** &mdash; Policy-based methods directly learn the optimal policy, without having to maintain a separate value function estimate. With value-based methods, the agent uses its experience with the environment to maintain an estimate of the optimal action-value function, from which an optimal policy is derived. This intermediate step requires the storage of lots of additional data since you need to account for all possible action values. Even if you discretize the action space, the number of possible actions can get quite large. And, using DQN to determine the action that maximizes the action-value function within a continuous or high-dimensional space requires a complex optimization process at every timestep.

##### &nbsp;

#### Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
The original DDPG algorithm from which I extended to create the MADDPG version, is outlined in [this paper](https://arxiv.org/pdf/1509.02971.pdf), _Continuous Control with Deep Reinforcement Learning_, by researchers at Google Deepmind. In this paper, the authors present "a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces." They highlight that DDPG can be viewed as an extension of Deep Q-learning to continuous tasks.

For the DDPG foundation, I used [this vanilla, single-agent DDPG](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) as a template. Then, to make this algorithm suitable for the multiple competitive agents in the Tennis environment, I implemented components discussed in [this paper](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf), _Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments_, by Lowe and Wu, along with other researchers from OpenAI, UC Berkeley, and McGill University. Most notable, I implemented their variation of the actor-critic method (see Figure 1), which I discuss in the following section.

Lastly, I further experimented with components of the DDPG algorithm based on other concepts covered in Udacity's classroom and lessons. My implementation of this algorithm (including various customizations) are discussed below.

<img src="assets/multi-agent-actor-critic.png" width="40%" align="top-left" alt="" title="Multi-Agent Actor-Critic" />

> _Figure 1: Multi-agent decentralized actor with centralized critic ([Lowe and Wu et al](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf))._

##### &nbsp;

#### Actor-Critic Method
Actor-critic methods leverage the strengths of both policy-based and value-based methods.

Using a policy-based approach, the agent (actor) learns how to act by directly estimating the optimal policy and maximizing reward through gradient ascent. Meanwhile, employing a value-based approach, the agent (critic) learns how to estimate the value (i.e., the future cumulative reward) of different state-action pairs. Actor-critic methods combine these two approaches in order to accelerate the learning process. Actor-critic agents are also more stable than value-based agents, while requiring fewer training samples than policy-based agents.

What makes this implementation unique is the **decentralized actor with centralized critic** approach from [the paper by Lowe and Wu](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). Whereas traditional actor-critic methods have a separate critic for each agent, this approach utilizes a single critic that receives as input the actions and state observations from all agents. This extra information makes training easier and allows for centralized training with decentralized execution. Each agent still takes actions based on its own unique observations of the environment.

You can find the actor-critic logic implemented as part of the `Agent()` class [here](https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition/blob/master/maddpg_agent.py#L110) in `maddpg_agent.py` of the source code. The actor-critic models can be found via their respective `Actor()` and `Critic()` classes [here](https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition/blob/master/model.py#L12) in `models.py`.

Note: As we did with Double Q-Learning in the last project, we're again leveraging local and target networks to improve stability. This is where one set of parameters `w` is used to select the best action, and another set of parameters `w'` is used to evaluate that action. In this project, local and target networks are implemented separately for both the actor and the critic.

### The Actor-Network is shown below

     # Actor Network (w/ Target Network)
      self.actor_local = Actor(state_size, action_size, random_seed).to(device)
      self.actor_target = Actor(state_size, action_size, random_seed).to(device)
      self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
    
### The Critic-Network is shown below

    # Critic Network (w/ Target Network)
      self.critic_local = Critic(state_size, action_size, random_seed).to(device)
      self.critic_target = Critic(state_size, action_size, random_seed).to(device)
      self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)


#### Gradient Clipping

Gradient clipping is a technique that tackles exploding gradients. The idea of gradient clipping is very simple: If the gradient gets too large, we rescale it to keep it small.
Initially, I added batch normalization between every layer in both the actor and critic models. However, this may have been overkill, and seemed to prolong training time. I eventually reduced the use of batch normalization to just the outputs of the first fully-connected layers of both the actor and critic models.

Gradient clipping ensures the gradient vector g has norm at most c. This helps gradient descent to have a reasonable behaviour even if the loss landscape of the model is irregular. The following figure shows an example with an extremely steep cliff in the loss landscape. Without clipping, the parameters take a huge descent step and leave the “good” region. With clipping, the descent step size is restricted and the parameters stay in the “good” region.

Note that this function is applied after the backward pass, but before the optimization step.


      # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
      # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()


##### &nbsp;

#### Experience Replay
Experience replay allows the RL agent to learn from past experience.

As with the [previous project](https://github.com/tommytracey/DeepRL-P2-Continuous-Control), the algorithm employs a replay buffer to gather experiences. Experiences are stored in a single replay buffer as each agent interacts with the environment. These experiences are then utilized by the central critic, therefore allowing the agents to learn from each others' experiences.

The replay buffer contains a collection of experience tuples with the state, action, reward, and next state `(s, a, r, s')`. The critic samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive algorithm could otherwise become biased by correlations between sequential experience tuples.

Also, experience replay improves learning through repetition. By doing multiple passes over the data, our agents have multiple opportunities to learn from a single experience tuple. This is particularly useful for state-action pairs that occur infrequently within the environment.

The implementation of the replay buffer can be found [here](https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition/blob/master/maddpg_agent.py#L196) in the `maddpg_agent.py` file of the source code.

##### &nbsp;

## Results
Once all of the above components were in place, the agents were able to solve the Tennis environment. Again, the performance goal is an average reward of at least +0.5 over 100 episodes, taking the best score from either agent for a given episode.

[Here](https://youtu.be/jOWWzygOi1A) is a video showing the trained agents playing a few points.

<a href="https://youtu.be/jOWWzygOi1A"><img src="assets/video-thumbnail.png" width="30%" align="top-left" alt="" title="Tennis Agent Video" /></a>

The graph below shows the final training results. The best-performing agents were able to solve the environment in 607 episodes, with a top score of 5.2 and a top moving average of 0.927. The complete set of results and steps can be found in [this notebook](Tennis.ipynb).

<img src="assets/model-graph.png" width="70%" align="top-left" alt="" title="Results Graph" />

<img src="assets/training-output.png" width="70%" align="top-left" alt="" title="Training Output" />


## Further Improvements
The following techniques can be tried out to further improve the performance of the network
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952): This technique prioritizes the experiences and chooses the best experience for further training when sampling from the buffer. This is known to reduce the training time and make the training more efficient.
- [Asynchornous Actor Critic Agent](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2): This technique trains multiple worker agents that interact with a glocal network asynchronously to optimize the policy and value function. This way, each of these agents interacts with it’s own copy of the environment at the same time as the other agents are interacting with their environments.
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347): This technique modifies the parameters of the network in such a way that the new set of parameters is looked for in the immediate neighbourhood of the parameters in the previous iteration of the training. This is shown also to be an efficient way of training the network so the search space is more optimal. 
