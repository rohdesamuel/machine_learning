import gym
import random
import QLearning # Your implementation goes here...
import GymSupport


def run():
  env = gym.make('CartPole-v0')

  discountRate = 0.98      # Controls the discount rate for future rewards -- this is gamma from 13.10
  actionProbabilityBase = 1.8  # This is k from the P(a_i|s) expression from section 13.3.5 and influences how random exploration is
  randomActionRate = 0.01    # Percent of time the next action selected by GetAction is totally random
  learningRateScale = 0.01   # Should be multiplied by visits_n from 13.11.
  trainingIterations = 2000

  qlearner = QLearning.SymbolicLearning(state_space_shape=GymSupport.CartPoleStateSpaceShape(),
                                 num_actions=env.action_space.n,
                                 discount_rate=discountRate)

  for trialNumber in range(trainingIterations):
    observation = env.reset()
    reward = 0
    for i in range(300):
      #env.render()

      currentState = GymSupport.CartPoleObservationToStateSpace(observation)
      action = qlearner.GetAction(currentState, learning_mode=True,
                                  random_action_rate=randomActionRate,
                                  action_probability_base=actionProbabilityBase)
      
      oldState = GymSupport.CartPoleObservationToStateSpace(observation)
      observation, reward, isDone, info = env.step(action)
      newState = GymSupport.CartPoleObservationToStateSpace(observation)

      qlearner.ObserveAction(oldState, action, newState, reward, learning_rate_scale=learningRateScale)

      if isDone:
        if(trialNumber%1000) == 0:
          print(trialNumber, i, reward)
        break
  print(len(qlearner._trajectories))
  ## Now do the best n runs I can
  #input("Enter to continue...")

  n = 20
  totalRewards = []
  for runNumber in range(n):
    observation = env.reset()
    totalReward = 0
    reward = 0
    for i in range(300):
      renderDone = env.render()

      currentState = GymSupport.CartPoleObservationToStateSpace(observation)
      observation, reward, isDone, info = env.step(qlearner.GetAction(currentState,
                                                                      learning_mode=False,
                                                                      random_action_rate=randomActionRate,
                                                                      action_probability_base=actionProbabilityBase))

      totalReward += reward

      if isDone:
        renderDone = env.render()
        print(i, totalReward)
        totalRewards.append(totalReward)
        break

  print(totalRewards)
  print("Your Score:", sum(totalRewards) / float(len(totalRewards)))