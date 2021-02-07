# Markov Decision Process - (MDP) The Bellman Equation adapted to Reinforcement
# Learning.
import numpy as np

# R is the reward matrix for each state
R = np.matrix([ [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 100, 1, 0, 0],
                [0, 1, 1, 0, 1, 0],
                [1, 0, 0, 1, 0 ,0],
                [0, 1, 0, 0, 0, 0] ])

# Q is our learning matrix
Q = np.matrix(np.zeros([6, 6]))

gamma = 0.8

agent_s_state = 1

def possible_actions(state):
  current_state_row = R[state,]
  possible_act = np.where(current_state_row > 0)[1]
  return possible_act

# get possible actions in the current state
possible_action = possible_actions(agent_s_state)

def action_choice(available_actions_range):
  next_action = int(np.random.choice(possible_action, 1))
  return next_action

# sample next action to be performed
action = action_choice(possible_action)

def reward(current_state, action, gamma):
  max_state = np.where(Q[action,] == np.max(Q[action,]))[1]

  if max_state.shape[0] > 1:
    max_state = int(np.random.choice(max_state, size=1))
  else:
    max_state = int(max_state)
  
  max_value = Q[action, max_state]
  Q[current_state, action] = R[current_state, action] + gamma * max_value

# rewarding q matrix
reward(agent_s_state, action, gamma)

for i in range(50000):
  current_state = np.random.randint(0, int(Q.shape[0]))
  possible_action = possible_actions(current_state)
  action = action_choice(possible_action)
  reward(current_state, action, gamma)
  
print('Q:')
print(Q)

# normalize q
print('Q Normed:')
print(Q/np.max(Q) * 100)
