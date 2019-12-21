import gym
import random
import QLearning # your implementation goes here...
import GymSupport

def parameter_tuning(discount_rate, action_probability_base,
                     training_iterations, bins_per_dimension,
                     verbose=False,
                     render=False):
  randomActionRate = 0.01    # Percent of time the next action selected by GetAction is totally random
  learningRateScale = 0.01   # Should be multiplied by visits_n from 13.11.

  env = gym.make('MountainCar-v0')
  qlearner = QLearning.QLearning(state_space_shape=GymSupport.MountainCarStateSpaceShapeWithDim(bins_per_dimension),
                                   num_actions=env.action_space.n,
                                   discount_rate=discount_rate)
  
  for trialNumber in range(training_iterations):
    observation = env.reset()
    reward = 0
    for i in range(201):
      env.render()
      currentState = GymSupport.MountainCarObservationToStateSpaceWithDim(observation, bins_per_dimension)
      action = qlearner.GetAction(currentState, learning_mode=True,
                                  random_action_rate=randomActionRate,
                                  action_probability_base=action_probability_base)

      oldState = GymSupport.MountainCarObservationToStateSpaceWithDim(observation, bins_per_dimension)
      observation, reward, isDone, info = env.step(action)
      newState = GymSupport.MountainCarObservationToStateSpaceWithDim(observation, bins_per_dimension)

      # learning rate scale 
      qlearner.ObserveAction(oldState, action, newState, reward, learning_rate_scale=learningRateScale)

      if isDone:
        if(trialNumber%1000) == 0 and verbose:
          print(trialNumber, i, reward)
        break

  n = 10
  totalRewards = []
  for runNumber in range(n):
    observation = env.reset()
    totalReward = 0
    reward = 0
    for i in range(201):
      if render:
        env.render()

      currentState = GymSupport.MountainCarObservationToStateSpaceWithDim(observation, bins_per_dimension)
      observation, reward, isDone, info = env.step(qlearner.GetAction(currentState, learning_mode=False,
                                  random_action_rate=randomActionRate,
                                  action_probability_base=action_probability_base))

      totalReward += reward

      if isDone:
        totalRewards.append(totalReward)
        if verbose:
          print(i, totalReward)
        break

  avg_score = sum(totalRewards) / float(len(totalRewards))

  if verbose:
    print(totalRewards)
    print("Your score:", avg_score)

  return (discount_rate, action_probability_base, training_iterations, bins_per_dimension, avg_score)

def run_default():
  env = gym.make('MountainCar-v0')

  discountRate = 0.98      # Controls the discount rate for future rewards -- this is gamma from 13.10
  actionProbabilityBase = 1.8  # This is k from the P(a_i|s) expression from section 13.3.5 and influences how random exploration is
  randomActionRate = 0.01    # Percent of time the next action selected by GetAction is totally random
  learningRateScale = 0.01   # Should be multiplied by visits_n from 13.11.
  trainingIterations = 20000  

  qlearner = QLearning.QLearning(state_space_shape=GymSupport.MountainCarStateSpaceShape(),
                                 num_actions=env.action_space.n,
                                 discount_rate=discountRate)

  for trialNumber in range(20000):
    observation = env.reset()
    reward = 0
    for i in range(201):
      #env.render()

      currentState = GymSupport.MountainCarObservationToStateSpace(observation)
      action = qlearner.GetAction(currentState, learning_mode=True,
                                  random_action_rate=randomActionRate,
                                  action_probability_base=actionProbabilityBase)

      oldState = GymSupport.MountainCarObservationToStateSpace(observation)
      observation, reward, isDone, info = env.step(action)
      newState = GymSupport.MountainCarObservationToStateSpace(observation)

      # learning rate scale 
      qlearner.ObserveAction(oldState, action, newState, reward, learning_rate_scale=learningRateScale)

      if isDone:
        if(trialNumber%1000) == 0:
          print(trialNumber, i, reward)
        break

  ## Now do the best n runs I can
  #input("Enter to continue...")

  n = 20
  totalRewards = []
  for runNumber in range(n):
    observation = env.reset()
    totalReward = 0
    reward = 0
    for i in range(201):
      renderDone = env.render()

      currentState = GymSupport.MountainCarObservationToStateSpace(observation)
      observation, reward, isDone, info = env.step(qlearner.GetAction(currentState, learning_mode=False,
                                  random_action_rate=randomActionRate,
                                  action_probability_base=actionProbabilityBase))

      totalReward += reward

      if isDone:
        renderDone = env.render()
        print(i, totalReward)
        totalRewards.append(totalReward)
        break

  print(totalRewards)
  print("Your score:", sum(totalRewards) / float(len(totalRewards)))

def tune_parameters():
  default_discount_rate = 0.98
  default_action_probability_base = 2.7183
  default_training_iterations = 5000
  default_bins_per_dimension = 20

  from joblib import Parallel, delayed
  print('discount_rate,action_probability_base,training_iterations,bins_per_dimension,avg_score')
  results = Parallel(n_jobs=10)(delayed(parameter_tuning)(discount_rate,
                                                          default_action_probability_base,
                                                          default_training_iterations,
                                                          default_bins_per_dimension)
                                for discount_rate in [0.999, 0.9999, 0.99999, 0.999999])
  for result in results:
    print(result)

  results = Parallel(n_jobs=10)(delayed(parameter_tuning)(default_discount_rate,
                                                           action_probability_base,
                                                           default_training_iterations,
                                                           default_bins_per_dimension)
                                for action_probability_base in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
  for result in results:
    print(result)

  results = Parallel(n_jobs=10)(delayed(parameter_tuning)(default_discount_rate,
                                                          default_action_probability_base,
                                                          default_training_iterations,
                                                          bins_per_dimension)
                                for bins_per_dimension in [10, 20, 50, 100, 200, 500])
  for result in results:
    print(result)

  results = Parallel(n_jobs=10)(delayed(parameter_tuning)(default_discount_rate,
                                                           default_action_probability_base,
                                                           training_iterations,
                                                           default_bins_per_dimension)
                                for training_iterations in [100, 1000, 2000, 5000, 10000, 20000])
  for result in results:
    print(result)

def run():
  parameter_tuning(discount_rate=0.999,
                   action_probability_base=10.0,
                   training_iterations=20000,
                   bins_per_dimension=20, verbose=True, render=True)