import random
import numpy as np
import collections
from KMeansModel import KMeans

class QLearning:
  def __init__(self, state_space_shape, num_actions, discount_rate):
    self._discount_rate = discount_rate
    self._num_actions = num_actions

    # The learned (state, action) reward pairs: Q^
    self._rewards = np.zeros((*state_space_shape, num_actions))

    # The number of times this agent has seen this (state, action) pair.
    self._num_visits = np.zeros((*state_space_shape, num_actions))

  def ObserveAction(self, old_state, action, new_state, reward, learning_rate_scale):
    self._num_visits[tuple(old_state)][action] += 1
    
    # Interpolation factor
    alpha = (1 / (1 + (self._num_visits[tuple(old_state)][action] * learning_rate_scale)))

    # The previous reward at the given (state, action): Q^(s, a)
    old_q = self._rewards[tuple(old_state)][action]

    # The new to give to the given (state, action): Q^(s', a')
    new_q = reward + self._discount_rate * np.max(self._rewards[tuple(new_state)])

    # Interpolate to create a decaying weighted average.
    self._rewards[tuple(old_state)][action] = (1 - alpha) * old_q + alpha * new_q 

  def GetAction(self, current_state, learning_mode, random_action_rate, action_probability_base):
    # An array where each index is the reward for taking an action at the current_state.
    rewards_at_state = self._rewards[tuple(current_state)]
    
    # Calculate a "likelihood" of the reward over each action for the current_state.
    state_likelihood = np.power(action_probability_base, rewards_at_state)

    # Calculates the probability of taken an action given the current_state: P(a_i | s).
    action_probabilities = state_likelihood / np.sum(state_likelihood)

    # The best action is the highest probability to take.
    if learning_mode:
      best_action = np.random.choice(self._num_actions, p=action_probabilities)

      # Every so often, take a random action to explore the state space.
      if np.random.uniform() < random_action_rate:
        best_action = random.randint(0, self._num_actions - 1)
    else:
      best_action = np.argmax(action_probabilities)
   
    return best_action

class Trajectory:
  def __init__(self):
    self._states = []
    self._hash = None
    self._is_loop = False

  def append(self, v):
    if self._is_loop:
      return

    new_states = self._states + [v]
    new_hash = self._do_hash(new_states)

    if self._hash == new_hash:
      self._is_loop = True
      return False

    if not self._is_loop:
      self._states = new_states
      self._hash = new_hash
    return True

  def _do_hash(self, states):
    return hash(frozenset(states))

  def __hash__(self):
    return self._do_hash(self._states)

  def __len__(self):
    return len(self._states)

  def __repr__(self):
    return str(self._states)

class SymbolicLearning:
  def __init__(self, state_space_shape, num_actions, discount_rate):
    self._discount_rate = discount_rate
    self._num_actions = num_actions

    # The learned (state, action) reward pairs: Q^
    self._rewards = np.zeros((*state_space_shape, num_actions))

    # The number of times this agent has seen this (state, action) pair.
    self._num_visits = np.zeros((*state_space_shape, num_actions))

    self._current_trajectory = Trajectory()
    self._trajectories = set()
    self._trajectories_visited = collections.defaultdict(lambda: 0)
    self._symbols = set()
    self._states = set()

  def _symbolize(self, state, action):
    #s = tuple(state + [action]) # (tuple(state), action)
    #return s

    self._states.add(tuple(state))
    new_symbols = self._symbols.copy()
    new_symbols.add(s)

    new_kmeans = KMeans()
    new_kmeans.fit(list(self._states), len(new_symbols), 10)
    print(new_kmeans.means())
    means = new_kmeans.means()
    #self._symbols = set(map(tuple, means))
    return s


  def ObserveAction(self, old_state, action, new_state, reward, learning_rate_scale):
    self._num_visits[tuple(old_state)][action] += 1
    
    s = self._symbolize(old_state, action)

    if not self._current_trajectory.append(s):
      self._trajectories.add(self._current_trajectory)
      self._trajectories_visited[self._current_trajectory] += 1
      self._current_trajectory = Trajectory()

    # Interpolation factor
    alpha = (1 / (1 + (self._num_visits[tuple(old_state)][action] * learning_rate_scale)))

    # The previous reward at the given (state, action): Q^(s, a)
    old_q = self._rewards[tuple(old_state)][action]

    # The new to give to the given (state, action): Q^(s', a')
    new_q = reward + self._discount_rate * np.max(self._rewards[tuple(new_state)])

    # Interpolate to create a decaying weighted average.
    self._rewards[tuple(old_state)][action] = (1 - alpha) * old_q + alpha * new_q 

  def GetAction(self, current_state, learning_mode, random_action_rate, action_probability_base):
    # An array where each index is the reward for taking an action at the current_state.
    rewards_at_state = self._rewards[tuple(current_state)]
    
    # Calculate a "likelihood" of the reward over each action for the current_state.
    state_likelihood = np.power(action_probability_base, rewards_at_state)

    # Calculates the probability of taken an action given the current_state: P(a_i | s).
    action_probabilities = state_likelihood / np.sum(state_likelihood)

    # The best action is the highest probability to take.
    if learning_mode:
      best_action = np.random.choice(self._num_actions, p=action_probabilities)

      # Every so often, take a random action to explore the state space.
      if np.random.uniform() < random_action_rate:
        best_action = random.randint(0, self._num_actions - 1)
    else:
      best_action = np.argmax(action_probabilities)
   
    return best_action