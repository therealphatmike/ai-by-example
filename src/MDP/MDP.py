import numpy as np

# Reward Matrix
#                A. B. C. D  E. F. G. H. I. J. K. L
R = np.matrix([ [0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0, 1, 100, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0] ])

# Q matrix -- matrix to hold output of Q function
Q = np.matrix(np.zeros([12, 12]))

agent_start = np.random.randint(0, 12)
gamma = 0.8


def possible_actions(current_state):
  return np.where(R[current_state] > 0)[1]


def decide_next_action(possible_transitions):
  return np.random.choice(possible_transitions, 1)


def determine_max_reward_for_prime_state(action):
  max_state = np.where(Q[action,] == np.max(Q[action,]))[1]

  if max_state.shape[0] > 1:
    max_state = int(np.random.choice(max_state, size=1))
  else:
    max_state = int(max_state)
  
  return Q[action, max_state]


def calculate_reward(current_state, action):
  max_s_prime = determine_max_reward_for_prime_state(action)
  Q[current_state, action] = R[current_state, action] + gamma * max_s_prime


def train():
  for i in range(50000):
    current_state = np.random.randint(0, 12)
    possible_transitions = possible_actions(current_state)
    action = decide_next_action(possible_transitions)
    calculate_reward(current_state, action)

def normalize(matrix):
  return matrix / np.max(matrix) * 100


def main():
  print('========================================')
  print('Initializing Reward and Q-value matrices')
  print('R:')
  print(R)
  print('Q:')
  print(Q)
  print('========================================')
  print('Training model...')
  print('========================================')
  train()
  print('Training step done. Outputting Q value matrix:')
  print('Q:')
  print(Q)
  print('========================================')
  print('Normalizing Q matrix')
  normalize(Q)
  print('Ouputting Normalized Q Matrix')
  print(Q)
  print('========================================')


if __name__ == '__main__':
  main()