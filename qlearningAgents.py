# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

identifier = '1'

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from memory import SequentialMemory
from featureExtractors import closestFood, closestGhost
from game import Actions

from keras.models import model_from_json

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input
from keras.optimizers import RMSprop, Adam
from keras.layers.merge import Add, Concatenate
import keras.backend as K

# from keras.layers import Dense, Dropout, Input
from keras.callbacks import TensorBoard

from collections import deque
import numpy as np
import random,util,math
import copy


TIME_PENALTY = -1 # Number of points lost each round
FOOD_REWARD = 10
DIE_PENALTY = -20
EAT_ALL_FOOD_REWARD = 0
PUDDLE_PENALTY = .8
steps = 0

class ActionMapping:
    NORTH = 'North'
    SOUTH = 'South'
    EAST = 'East'
    WEST = 'West'
    STOP = 'Stop'

    ActionToNumber = {
        NORTH :  0,
        SOUTH :  1,
        EAST  :  2,
        WEST  :  3,
        STOP  :  4
    }

    NumbertoAction = {
        0 : NORTH,
        1 : SOUTH,
        2 : EAST,
        3 : WEST,
        4 : STOP
    }


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()
        self.totalTrainingSteps = 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.getLegalActions(state)
        if possibleActions:
            maxv = float("-inf")
            for action in possibleActions:
                q = self.getQValue(state, action)
                if q >= maxv:
                    maxv = q
            return maxv
        return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.getLegalActions(state)
        if possibleActions:
            maxv = float("-inf")
            bestAction = None
            for action in possibleActions:
                q = self.getQValue(state, action)
                if q >= maxv:
                    maxv = q
                    bestAction = action
            return bestAction
        return None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        possibleActions = self.getLegalActions(state)
        action = None
        # print self.foodAgent.getQValues(state)
        "*** YOUR CODE HERE ***"
        if possibleActions:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.getLegalActions(nextState)
        R = reward
        if possibleActions:
            Q = []
            for a in possibleActions:
                Q.append(self.getQValue(nextState, a))
            R = reward + self.discount * max(Q)
        self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (R - self.getQValue(state, action))

    def getPolicy(self, state):
        if not self.selfTesting:
            self.totalTrainingSteps += 1

        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.decay = .9999


    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        f = self.featExtractor.getFeatures(state, action)
        qv = 0
        for feature in f:
            qv = qv + self.weights[feature] * f[feature]
        return qv

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay

        R = reward
        f = self.featExtractor.getFeatures(state, action)
        alphadiff = self.alpha * ((R + self.discount * self.getValue(nextState)) - self.getQValue(state, action))
        for feature in f.keys():
            self.weights[feature] = self.weights[feature] + alphadiff * f[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


class DqnModule():
    '''
        This class only deals with numerical actions
    '''
    def __init__(self, name, nb_features, featureExtractor, batch_size = 32, start_epsilon = 1, min_epsilon = 0.01, decay = 0.99, discount = 0.95, nb_actions = 5):
        self.batch_size = batch_size
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.discount = discount
        self.tau = 0.01
        self.model = NeuralNetwork(input_size = nb_features, nb_actions = nb_actions).model
        self.target_model = NeuralNetwork(input_size = nb_features, nb_actions = nb_actions).model
        # self.model = self.loadModel(name)
        self.replay_memory_buffer = deque(maxlen=10000)
        self.extractor = featureExtractor


        print '----------'
        print '### DqnModule ###'
        print 'Epsilon Decay = %s, Discount Factor = %.2f' % (self.decay, self.discount)
        print 'Input Features = %d' % (nb_features)
        print '----------'


    def getQValue(self, state, action):
        qValues = self.model.predict(self.extractor(state))
        return qValues[action]

    def getAction(self, state, legalActions):
        qValues = self.model.predict(np.array([self.extractor(state)]), batch_size=1)[0]
        maxQ, bestAction = float('-inf'), None
        for action in legalActions:
            if qValues[action] > maxQ:
                maxQ, bestAction = qValues[action], action
        return bestAction

    def getQValues(self, state):
        return self.model.predict(np.array([self.extractor(state)]), batch_size=1)[0]

    def update(self, state, action, nextState, reward, done):
        self.add_to_replay_memory(state, action, reward, nextState, done)
        self.replayExperience()
        self._update_target()

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((self.extractor(state),
            action, reward, self.extractor(next_state), done))

    def replayExperience(self):
        # replay_memory_buffer size check
        if len(self.replay_memory_buffer) < self.batch_size:
        # if len(self.replay_memory_buffer) < 5:
            return
        random_sample = self.get_random_sample_from_replay_mem()
        indexes = np.array([i for i in range(self.batch_size)])
        states, actions, rewards, next_states, done_list = self.get_attribues_from_sample(random_sample)





        targets = rewards + self.discount * (np.amax(self.target_model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)

        target_vec[[indexes], [actions]] = targets


        
        self.model.fit(states, target_vec, epochs=1, verbose=0)



    def get_random_sample_from_replay_mem(self):
        return random.sample(self.replay_memory_buffer, self.batch_size)

    def get_attribues_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list
    
    def get_policy(self, state):
        # qvalues = self.getQValues(state)
        # qvalues = tf.convert_to_tensor(self.getQValues(state), dtype=tf.float32)
        # sm_layer = tf.keras.layers.Softmax(axis=0)
        # policy = sm_layer(qvalues).np()
        # policy_tensor = tf.nn.softmax(qvalues)
        # policy = tf.Session().run(policy_tensor)
        qvalues = self.getQValues(state)
        # legalActions = [ActionMapping.ActionToNumber[action] for action in self.getLegalActions(state)]
        # legalActionNumbers  = np.asarray([1 if number in legalActions else 0 for number in range(0, 5)])
        legalActionNumbers  = np.asarray([1]*5)
        policy = np.exp(qvalues)*legalActionNumbers/sum(np.exp(qvalues)*legalActionNumbers)

        # print "++++++++++++"
        # print policy, sum(policy)
        # print "++++++++++++"
        return policy

    def get_epsgreedy_dist(self, state, steps):
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 10000
        q_vals = self.getQValues(state)
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)
        a_star = np.argmax(q_vals)
        pi_s_a = [eps/5 if a!=a_star else 1.-4*eps/5 for a in range(5)]
        return np.array(pi_s_a)
    
    def loadModel(self, file_name):
        json_file = open('weights/' + file_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('weights/' + file_name + '.h5')
        return loaded_model

    def _update_target(self):
        model_weights  = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = model_weights[i]*self.tau + target_weights[i]*(1-self.tau)
        self.target_model.set_weights(target_weights)



class HierarchicalQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 9
        self.nb_actions = 5
        self.nb_features = 13
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.decay = .999
        self.arbitratorDecay = .9995
        self.arbitratorEpsilon = 1
        self.ghostAgent = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        self.foodAgent = DqnModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        self.arbitrator = DqnModule(nb_features = self.nb_features, featureExtractor = CustomizedExtractor().getFeatures, nb_actions = 2)
        self.subModules = [self.ghostAgent, self.foodAgent]
        self.lastSavedWeights = -1
        # self.foodAgent.model = self.loadModel(name)
        # self.ghostAgent.model = self.loadModel(name)
        self.isSaved = 0
        print '----------'
        print '############ HierarchicalQAgent ############'
        print 'Epsilon Decay = %f, Arbitrator Epsilon Decay = %f, Discount Factor = %.2f' % (self.decay, self.arbitratorDecay, self.discount)
        print 'Feature Count: Arbitrator = %d, Ghost = %d, Food = %d' % (self.nb_features, self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for Arbitrator: (Eat ghost) = %.2f, (Eat Food) = %.2f, (Death Penalty) = %.2f, (Time Penalty) = %.2f' % \
        (self.getArbitratorReward(50), self.getArbitratorReward(10), self.getArbitratorReward(-500), self.getArbitratorReward(-1))
        print 'Rewards for foodAgent: Time Penalty = %.2f, (Food Reward + Time Penalty) = %.2f, \
        (Food Reward + Time Penalty + LastReward) = %.2f' % (self.getFoodReward(TIME_PENALTY), \
        self.getFoodReward(TIME_PENALTY + FOOD_REWARD), self.getFoodReward(TIME_PENALTY + FOOD_REWARD + EAT_ALL_FOOD_REWARD))
        print 'Rewards for ghostAgent: Time Penalty = %.2f, (Death Penalty) = %.2f' % (self.getGhostReward(TIME_PENALTY), \
        self.getGhostReward(TIME_PENALTY + DIE_PENALTY))
        print '----------'


    def computeActionFromQValues(self, state):
        # legalActions = self.getLegalActions(state)
        if np.random.rand() < self.arbitratorEpsilon:
            self.arbitratorAction = random.randrange(2)
        else:
            self.arbitratorAction = self.arbitrator.getAction(state, [0, 1])

        legalActions = [ActionMapping.ActionToNumber[action] for action in self.getLegalActions(state)]

        action = self.subModules[self.arbitratorAction].getAction(state, legalActions)


        return ActionMapping.NumbertoAction[action]

    def getFoodReward(self, reward):
        MODIFIED_TIME_PENALTY = -10. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        # MODIFIED_PUDDLE_PENALTY = 10

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY

        return reward / 10.0

    def getGhostReward(self, reward):
        MODIFIED_TIME_PENALTY = -1 # Number of points lost each round
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_DIE_PENALTY

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_DIE_PENALTY

        return reward / 10.0

    def getArbitratorReward(self, reward):

        MODIFIED_TIME_PENALTY = -10. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        if reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY
        if reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD + MODIFIED_DIE_PENALTY

        return reward / 10.0

    def saveModel(self, model, file_name):
        model_json = model.to_json()
        with open('weights/' + file_name + '.json', "w") as json_file:
            json_file.write(model_json)
        model.save_weights('weights/' + file_name + '.h5')

    def loadModel(self, file_name):
        json_file = open('weights/' + file_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('weights/' + file_name + '.h5')
        return loaded_model

    def update(self, state, action, nextState, reward):
        if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
            self.saveModel(self.ghostAgent.model, 'ghostAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.saveModel(self.foodAgent.model, 'foodAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.saveModel(self.arbitrator.model, 'arbitrator_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.lastSavedWeights = self.currentTrainingEpisode
        if self.alpha < 0.0001:
            return
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay

        if self.arbitratorEpsilon > self.min_epsilon:
            self.arbitratorEpsilon = self.arbitratorEpsilon * self.arbitratorDecay

        done = 1 if abs(reward) > 100 else 0
        # if done:
        #     print '------------------------------------------------------------'

        self.arbitrator.update(state, self.arbitratorAction, nextState, self.getArbitratorReward(reward), done)
        self.ghostAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getGhostReward(reward), done)
        self.foodAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getFoodReward(reward), done)


class HierarchicalDDPGAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 9
        # self.nb_puddleFeatures = 15
        self.nb_actions = 5
        self.nb_features = 13
        self.arbitrator_actions = 2
        # self.epsilon = 1
        # self.min_epsilon = 0.01
        self.decay = .999
        # self.puddleAgent = DqnModule(nb_features = self.nb_puddleFeatures, featureExtractor = CustomizedExtractor().getPuddleFeatures)
        self.ghostAgent = DqnModule(name = "ghostAgent", nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        self.foodAgent = DqnModule(name = "foodAgent", nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        self.arbitrator = DDPGModule(nb_features = self.nb_features, featureExtractor = CustomizedExtractor().getFeatures, nb_actions = self.arbitrator_actions, decay = self.decay)
        self.subModules = [self.ghostAgent, self.foodAgent]
        self.lastSavedWeights = -1
        # self.foodAgent.model = self.loadModel(name)
        # self.ghostAgent.model = self.loadModel(name)
        self.isSaved = 0
        print '----------'
        print '############ HierarchicalDDPGAgent ############'
        print 'Arbitrator Epsilon Decay = %f, Discount Factor = %.2f' % (self.arbitrator.decay, self.discount)
        print 'Feature Count: Arbitrator = %d, Ghost = %d, Food = %d' % (self.nb_features, self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for Arbitrator: (Eat ghost) = %.2f, (Eat Food) = %.2f, (Death Penalty) = %.2f, (Time Penalty) = %.2f' % \
        (self.getArbitratorReward(50), self.getArbitratorReward(10), self.getArbitratorReward(-500), self.getArbitratorReward(-1))
        print 'Rewards for foodAgent: Time Penalty = %.2f, (Food Reward + Time Penalty) = %.2f, \
        (Food Reward + Time Penalty + LastReward) = %.2f' % (self.getFoodReward(TIME_PENALTY), \
        self.getFoodReward(TIME_PENALTY + FOOD_REWARD), self.getFoodReward(TIME_PENALTY + FOOD_REWARD + EAT_ALL_FOOD_REWARD))
        print 'Rewards for ghostAgent: Time Penalty = %.2f, (Death Penalty) = %.2f' % (self.getGhostReward(TIME_PENALTY), \
        self.getGhostReward(TIME_PENALTY + DIE_PENALTY))
        print '----------'


    def computeActionFromQValues(self, state):
        self.arbitratorAction = self.arbitrator.getAction(state)[0]
        scaleParameters = self.arbitratorAction

        # if self.currentTrainingEpisode > 300:
        # print state
        # print 'action = ', scaleParameter

        # puddleQValues = self.puddleAgent.getQValues(state)
        ghostQValues = self.ghostAgent.getQValues(state)
        foodQValues = self.foodAgent.getQValues(state)
        scalarizedQValues = scaleParameters[0] * (ghostQValues) + scaleParameters[1] * foodQValues
        # scalarizedQValues = scaleParameters[0] * ghostQValues + scaleParameters[1] * foodQValues + scaleParameters[2] * puddleQValues
        # scalarizedQValues = scaleParameter * ghostQValues + (1 - scaleParameter) * (foodQValues)

        bestAction = ActionMapping.NumbertoAction[np.argmax(scalarizedQValues)]
        return bestAction

    def getPuddleReward(self, reward):

        MODIFIED_PUDDLE_PENALTY = -3.0
        MODIFIED_TIME_PENALTY = -1. # Number of points lost each round

        if reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_PUDDLE_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_PUDDLE_PENALTY
        else:
            reward = MODIFIED_TIME_PENALTY

        return reward / 10.0

    def getFoodReward(self, reward):
        MODIFIED_TIME_PENALTY = -10. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        # MODIFIED_PUDDLE_PENALTY = 10

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY

        return reward / 10.0

    def getGhostReward(self, reward):
        MODIFIED_TIME_PENALTY = -1 # Number of points lost each round
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_DIE_PENALTY

        return reward / 10.0

    def getArbitratorReward(self, reward):

        MODIFIED_PUDDLE_PENALTY = .8
        MODIFIED_TIME_PENALTY = -1. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD + MODIFIED_DIE_PENALTY

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_PUDDLE_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY + MODIFIED_PUDDLE_PENALTY

        return reward / 10.0

    def saveModel(self, model, file_name):
        model_json = model.to_json()
        with open('weights/' + file_name + '.json', "w") as json_file:
            json_file.write(model_json)
        model.save_weights('weights/' + file_name + '.h5')

    def loadModel(self, file_name):
        json_file = open('weights/' + file_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('weights/' + file_name + '.h5')
        return loaded_model

    def update(self, state, action, nextState, reward):        
        # if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
        # # if self.selfTesting and self.currentTrainingEpisode == 5:    
        #     self.saveModel(self.ghostAgent.model, 'ghostAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
        #     self.saveModel(self.foodAgent.model, 'foodAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
        #     # self.saveModel(self.puddleAgent.model, 'puddleAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
        #     self.saveModel(self.arbitrator.actor_model, 'actor_' + identifier + '_' + str(self.currentTrainingEpisode))
        #     self.saveModel(self.arbitrator.critic_model, 'critic_' + identifier + '_' + str(self.currentTrainingEpisode))
        #     # self.saveModel(self.arbitrator.model, 'arbitrator_' + identifier + '_' + str(self.currentTrainingEpisode))
        #     self.lastSavedWeights = self.currentTrainingEpisode
        if self.alpha < 0.0001:
            return
        # if self.epsilon > self.min_epsilon:
        #     self.epsilon = self.epsilon * self.decay

        done = 1 if abs(reward) > 100 else 0
        # if done:
        #     print '------------------------------------------------------------'

        self.arbitrator.update(state, self.arbitratorAction, nextState, self.getArbitratorReward(reward), done)
        # self.ghostAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getGhostReward(reward), done)
        # self.foodAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getFoodReward(reward), done)
        # self.puddleAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getPuddleReward(reward), done)

class DDPGModule:
    def __init__(self, nb_features, featureExtractor, nb_actions, decay):
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.alpha = 0.0005
        self.epsilon = .9
        self.min_epsilon = .01
        self.decay = decay
        self.gamma = .8
        self.tau   = .01
        self.batch_size = 32
        self.extractor = featureExtractor
        self.nb_features = nb_features
        self.nb_actions = nb_actions
        print '----------'
        print '### DDPG Module ###'
        print 'Epsilon Decay = %s, Discount Factor = %.2f, alpha = %f' % (self.decay, self.gamma, self.alpha)
        print 'Input Features = %d' % (self.nb_features)
        print '----------'

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.replay_memory_buffer = deque(maxlen=50000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32,
           [None, self.nb_actions]) # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
            actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.alpha).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, \
            self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
            self.critic_action_input) # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        state_input = Input(shape=(self.nb_features,))
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(128, activation='relu')(h1)
        h3 = Dense(64, activation='relu')(h2)
        output = Dense(self.nb_actions, activation='tanh')(h3)

        model = Model(input=state_input, output=output)
        adam  = Adam(lr=self.alpha)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=(self.nb_features,))
        state_h1 = Dense(64, activation='relu')(state_input)
        state_h2 = Dense(128)(state_h1)

        action_input = Input(shape=(self.nb_actions,))
        action_h1 = Dense(64)(action_input)

        merged = Concatenate()([state_h2, action_h1])
        merged_h1 = Dense(64, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model = Model(input=[state_input,action_input], output=output)

        adam = Adam(lr=self.alpha)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #
    def update(self, state, action, nextState, reward, done):
        if self.alpha < 0.000001:
            return
        # action = (action - 0.5) * 2
        self.add_to_replay_memory(state, action, reward, nextState, done)
        self.train()
        self.update_target()

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((self.extractor(state),
            action, reward, self.extractor(next_state), done))

    def _train_actor(self, samples):

        cur_states, actions, rewards, new_states, _ =  self.get_attributes_from_sample(samples)
        predicted_actions = self.actor_model.predict(cur_states)
        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_state_input:  cur_states,
            self.critic_action_input: predicted_actions
        })[0]

        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: cur_states,
            self.actor_critic_grad: grads
        })

    def _train_critic(self, samples):

        cur_states, actions, rewards, new_states, dones = self.get_attributes_from_sample(samples)
        target_actions = self.target_actor_model.predict(new_states)
        future_rewards = self.target_critic_model.predict([new_states, target_actions])

        rewards += self.gamma * future_rewards * (1 - dones)

        evaluation = self.critic_model.fit([cur_states, actions], rewards, verbose=0)
        #print(evaluation.history)

    def train(self):
        if len(self.replay_memory_buffer) < self.batch_size:
            return

        rewards = []
        samples = random.sample(self.replay_memory_buffer, self.batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]*self.tau + actor_target_weights[i]*(1-self.tau)
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def get_attributes_from_sample(self, random_sample):
        array = np.array(random_sample)

        current_states = np.stack(array[:,0]).reshape((array.shape[0],-1))
        actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
        rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
        new_states = np.stack(array[:,3]).reshape((array.shape[0],-1))
        dones = np.stack(array[:,4]).reshape((array.shape[0],-1))

        return current_states, actions, rewards, new_states, dones

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def getAction(self, state):
        state = self.extractor(state).reshape((1, self.nb_features))
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
        if np.random.random() < self.epsilon:
            noise = np.random.uniform(-1, 1, size = self.nb_actions)
            action = self.actor_model.predict(state) + noise
            return action
        action = self.actor_model.predict(state)
        return action


class PolMixModule:
    def __init__(self, nb_features, featureExtractor, nb_actions, decay, nb_module_a=5, batch_size=16):
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.alpha_actor = 0.001
        self.alpha_critic = 0.01
        self.epsilon = 0.9
        self.min_epsilon = 0.01
        self.decay = decay
        self.gamma = 0.95
        self.tau = 0.01
        self.batch_size = 32
        self.extractor = featureExtractor
        self.nb_features = nb_features
        self.nb_actions = nb_actions
        self.replay_memory_buffer = deque(maxlen=10000)
        self.nb_module_a = nb_module_a
        self.batch_size = batch_size
        self.onpolicy_buffer = []
        self.reward_buffer = []

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.critic_state_input, self.critic_model = self.create_critic_model()
        _, self.target_critic_model = self.create_critic_model()

        self._actor_train_fn()

        # self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input) 
        
        self.sess.run(tf.initialize_all_variables())

    
    def _actor_train_fn(self):
        q_arb = K.placeholder(shape=(None,1),name="q_arb")
        pol_1 = K.placeholder(shape=(None,5),name="pol_1")
        pol_2 = K.placeholder(shape=(None,5),name="pol_2")
        actions = K.placeholder(shape=(None,5),name="actions")
        # q_arb = K.placeholder(shape=(1,1),name="q_arb")
        # pol_1 = K.placeholder(shape=(1,5),name="pol_1")
        # pol_2 = K.placeholder(shape=(1,5),name="pol_2")
        # actions = K.placeholder(dtype=tf.int32, shape=(1,1),name="actions")
        lambdas = self.actor_model.output

    
        loss = -tf.reshape(q_arb,[-1]) * tf.log(lambdas[:,0]*tf.reduce_sum(pol_1*actions, axis = 1) + lambdas[:,1]*tf.reduce_sum(pol_2*actions, axis = 1) + 0.00001)
        print(loss)


        loss = tf.reduce_sum(loss)

        adam = Adam(lr=self.alpha_actor)

        updates = adam.get_updates(params=self.actor_model.trainable_weights, loss=loss)

        self.train_fn = K.function(inputs=[self.actor_model.input, actions, q_arb, pol_1, pol_2], outputs=[], updates=updates)


    def create_actor_model(self):
        state_input = Input(shape=(self.nb_features,))
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(128, activation='relu')(h1)
        h3 = Dense(64, activation='relu')(h2)
        # output = Dense(self.nb_actions, activation='tanh')(h3)
        output = Dense(self.nb_actions, activation='softmax')(h3)

        model = Model(input=state_input, output=output)
        
        return state_input, model


    def create_critic_model(self):
        state_input = Input(shape=(self.nb_features,))
        state_h1 = Dense(64, activation='relu')(state_input)
        state_h2 = Dense(128, activation='relu')(state_h1)
        state_h3 = Dense(64, activation='relu')(state_h2)
        output = Dense(1, activation='linear')(state_h3)
        model = Model(input=state_input, output=output)

        adam = Adam(lr=self.alpha_critic)
        model.compile(loss="mse", optimizer=adam)
        
        return state_input, model


    # def update(self, state, action, nextState, reward, done, pi1, pi2):
    def update(self):
        #pi1 and pi2 are distributions of each module
        # if self.alpha_actor < 0.000001:
        #     return
        # self.add_to_replay_memory(state, action, reward, nextState, done, pi1, pi2)
        if len(self.onpolicy_buffer) > 0:
            self.train()
            self.update_target()
            self.onpolicy_buffer[:] = []
            self.reward_buffer[:] = []


    
    def add_to_onpolicy_buffer(self, state, action, reward, next_state, done, pi1, pi2):
        self.onpolicy_buffer.append((self.extractor(state),
            np.asarray([1 if i == action else 0 for i in range(5)]), reward, self.extractor(next_state), done, pi1, pi2))

    # def add_to_replay_memory(self, state, action, reward, next_state, done, pi1, pi2):
    #     self.replay_memory_buffer.append((self.extractor(state),
    #         action, reward, self.extractor(next_state), done, pi1, pi2))

 
    def _train_actor(self, samples):
        cur_states, actions, rewards, new_states, dones, pi1, pi2 = self.get_attributes_from_sample(samples)
        # predicted_actions = self.actor_model.predict(cur_states)
        V_next = self.critic_model.predict(new_states)
        V_curr = self.critic_model.predict(cur_states)
        predicted_q = rewards + self.gamma * V_next * (1-dones) - V_curr
        
        # #compute empirical return stored in predicted_q
        # empirical_return = []
        # episodes = []
        # episode_reward = []
        # for i in range(len(self.reward_buffer)-1):
        #     episode_reward.append(self.reward_buffer[i][1])
        #     if self.reward_buffer[i][0] != self.reward_buffer[i+1][0]:
        #         episodes.append(copy.deepcopy(episode_reward))
        #         episode_reward = []
        #     if i+1 == len(self.reward_buffer)-1:
        #         episode_reward.append(self.reward_buffer[i+1][1])
        #         episodes.append(copy.deepcopy(episode_reward))
        #         episode_reward = []
 
        # for episode_reward in episodes:
        #     rets = []
        #     return_so_far = 0
        #     for t in range(len(episode_reward) - 1, -1, -1):
        #         return_so_far = episode_reward[t] + self.gamma * return_so_far
        #         rets.append(return_so_far)
        #     # The returns are stored backwards in time, so we need to revert it
        #     rets = np.array(rets[::-1])
        #     # normalise returns
        #     rets = (rets - np.mean(rets)) / (np.std(rets) + 1e-8)
        #     empirical_return.extend(copy.deepcopy(rets))



        # predicted_q = np.asarray(empirical_return).reshape((-1,1))

        self.train_fn([cur_states, actions, predicted_q, pi1, pi2])


    def _train_critic(self, samples):
        cur_states, actions, rewards, new_states, dones, pi1, pi2 = self.get_attributes_from_sample(samples)
        # target_actions = self.target_actor_model.predict(new_states)
        # print(new_states.shape)
        V_next = self.critic_model.predict(new_states)
        # V_next = self.target_critic_model.predict(new_states)
        # V_curr = self.critic_model.predict(cur_states)

        # td_error = rewards + self.gamma * V_next * (1 - dones) - V_curr
        target = rewards + self.gamma * V_next * (1 - dones) 

        evaluation = self.critic_model.fit(cur_states, target, verbose=0)


    def train(self):
        # if len(self.replay_memory_buffer) < self.batch_size:
        #     return
        # rewards = []
        # samples = random.sample(self.replay_memory_buffer, self.batch_size)
        # samples = self.replay_memory_buffer.pop()
        samples = self.onpolicy_buffer[:]
        self._train_critic(samples)
        self._train_actor(samples)


    # def _update_actor_target(self):
    #     actor_model_weights  = self.actor_model.get_weights()
    #     actor_target_weights = self.target_actor_model.get_weights()
    #     for i in range(len(actor_target_weights)):
    #         actor_target_weights[i] = actor_model_weights[i]*self.tau + actor_target_weights[i]*(1-self.tau)
    #     self.target_actor_model.set_weights(actor_target_weights)


    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
        self.target_critic_model.set_weights(critic_target_weights)


    def update_target(self):
        # self._update_actor_target()
        self._update_critic_target()


    def get_attributes_from_sample(self, random_sample):
        array = np.array(random_sample)
        # print(array)
        current_states = np.stack(array[:,0]).reshape((array.shape[0],-1))
        actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
        rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
        new_states = np.stack(array[:,3]).reshape((array.shape[0],-1))
        dones = np.stack(array[:,4]).reshape((array.shape[0],-1))
        pi1 = np.stack(array[:,5]).reshape((array.shape[0],-1))
        pi2 = np.stack(array[:,6]).reshape((array.shape[0],-1))

        # current_states = np.asarray(array[0]).reshape((-1,len(array[0])))
        # actions = np.asarray(array[1]).reshape((-1,1))
        # rewards = np.asarray(array[2]).reshape((-1,1))
        # new_states = np.asarray(array[3]).reshape((-1,len(array[3])))
        # dones = np.asarray(array[4]).reshape((-1,1))
        # pi1 = np.asarray(array[5]).reshape((-1,len(array[5])))
        # pi2 = np.asarray(array[6]).reshape((-1,len(array[6])))


        return current_states, actions, rewards, new_states, dones, pi1, pi2


    def getAction(self, state):
        state = self.extractor(state).reshape((1, self.nb_features))
        # if self.epsilon > self.min_epsilon:
        #     self.epsilon *= self.decay
        # if np.random.random() < self.epsilon:
        #     noise = np.random.uniform(-1, 1, size = self.nb_actions)
        #     action = self.actor_model.predict(state) + noise
        #     return action
        action = self.actor_model.predict(state)
        return action

        
class PolMixAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', batch_size = 16, **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 9
        self.nb_actions = 5
        self.nb_features = 13
        self.arbitrator_actions = 2
        self.decay = 0.999
        self.ghostAgent = DqnModule(name = "ghostAgent", nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        self.foodAgent = DqnModule(name = "foodAgent", nb_features=self.nb_foodFeatures, featureExtractor=CustomizedExtractor().getFoodFeatures)
        
        self.arbitrator = PolMixModule(nb_features=self.nb_features, featureExtractor=CustomizedExtractor().getFeatures, 
        nb_actions=self.arbitrator_actions, decay=self.decay, nb_module_a=self.nb_actions)
        self.batch_size = batch_size

        self.subModules = [self.ghostAgent, self.foodAgent]
        self.lastSavedWeights = -1
        self.isSaved = 0
        print '----------'
        print '############ PolMixAgent ############'
        print 'Arbitrator Epsilon Decay = %f, Discount Factor = %.2f' % (self.arbitrator.decay, self.discount)
        print 'Feature Count: Arbitrator = %d, Ghost = %d, Food = %d' % (self.nb_features, self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for Arbitrator: (Eat ghost) = %.2f, (Eat Food) = %.2f, (Death Penalty) = %.2f, (Time Penalty) = %.2f' % \
        (self.getArbitratorReward(50), self.getArbitratorReward(10), self.getArbitratorReward(-500), self.getArbitratorReward(-1))
        print 'Rewards for foodAgent: Time Penalty = %.2f, (Food Reward + Time Penalty) = %.2f, \
        (Food Reward + Time Penalty + LastReward) = %.2f' % (self.getFoodReward(TIME_PENALTY), \
        self.getFoodReward(TIME_PENALTY + FOOD_REWARD), self.getFoodReward(TIME_PENALTY + FOOD_REWARD + EAT_ALL_FOOD_REWARD))
        print 'Rewards for ghostAgent: Time Penalty = %.2f, (Death Penalty) = %.2f' % (self.getGhostReward(TIME_PENALTY), \
        self.getGhostReward(TIME_PENALTY + DIE_PENALTY))
        print '----------'

    ############### change here ##############
    def computeActionFromQValues(self, state):

        #random exploration        

        self.arbitratorAction = self.arbitrator.getAction(state)[0]
        scaleParameters = self.arbitratorAction
        print(scaleParameters)

        global steps
        steps += 1
        ghost_policy = self.ghostAgent.get_epsgreedy_dist(state, steps)
        food_policy = self.foodAgent.get_epsgreedy_dist(state, steps)

        # print(steps)
        # print(ghost_policy)
        # print(food_policy)

        mix_policy = scaleParameters[0] * ghost_policy + scaleParameters[1] * food_policy
        # print(mix_policy)

        # mix_policy_tensor = tf.convert_to_tensor(mix_policy)
        # norm_mix_policy_tensor = tf.nn.softmax(mix_policy_tensor)
        # mix_policy_np = tf.Session().run(norm_mix_policy_tensor) 
        # legalActions = [ActionMapping.ActionToNumber[action] for action in self.getLegalActions(state)]
        # legalActionNumbers  = np.asarray([1 if number legalActions else 0 in  for number in range(0, 5)])
        # policy = np.exp(mix_policy)*legalActionNumbers/sum(np.exp(mix_policy)*legalActionNumbers)
        
        #how about invalid actions???
        legalActions = [ActionMapping.ActionToNumber[action] for action in self.getLegalActions(state)]
        legalActionNumbers  = np.asarray([1 if number in legalActions else 0 for number in range(0, 5)])
        policy = mix_policy*legalActionNumbers/sum(mix_policy*legalActionNumbers)
        

        # print((scaleParameters[0],scaleParameters[1]))
        # print(self.ghostAgent.getQValues(state))
        # print(ghost_policy)
        # print(policy)
        # mix_policy = mix_policy/sum(mix_policy)
        
        action_number = np.random.choice(np.arange(0,5), p=policy)
        action = ActionMapping.NumbertoAction[action_number]
        return action

    def update(self, state, action, nextState, reward):
        # if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
        #     self.saveModel(self.arbitrator.model, 'arbitrator_' + identifier + '_' + str(self.currentTrainingEpisode))
        #     self.lastSavedWeights = self.currentTrainingEpisode
        
        # if self.alpha_actor < 0.0001:
        #     return
        # print(reward)
        done = 1 if abs(reward) > 100 else 0

        ####### changes here ########

        # print(self.episodesSoFar)
        if self.episodesSoFar % self.batch_size == 0:
            # arguments to the update function not being used: remove later
            # self.arbitrator.update(state, ActionMapping.ActionToNumber[action], nextState, self.getArbitratorReward(reward), done,
            #                         self.ghostAgent.get_policy(state), self.foodAgent.get_policy(state))
            # print(self.episodesSoFar)
            self.arbitrator.update()
        else:
            ArbReward = self.getArbitratorReward(reward)
            self.arbitrator.reward_buffer.append((self.episodesSoFar, ArbReward))
            self.arbitrator.add_to_onpolicy_buffer(state, ActionMapping.ActionToNumber[action], ArbReward,
                                                nextState, done, self.ghostAgent.get_policy(state), 
                                                self.foodAgent.get_policy(state))
        self.ghostAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getGhostReward(reward), done)
        self.foodAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getFoodReward(reward), done)

        
    def getFoodReward(self, reward):
        MODIFIED_TIME_PENALTY = -10. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        # MODIFIED_PUDDLE_PENALTY = 10

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY

        return reward / 10.0

    def getGhostReward(self, reward):
        MODIFIED_TIME_PENALTY = -1 # Number of points lost each round
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_DIE_PENALTY

        return reward / 10.0

    def getArbitratorReward(self, reward):
        MODIFIED_PUDDLE_PENALTY = .8
        MODIFIED_TIME_PENALTY = -1. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD + MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_PUDDLE_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY + MODIFIED_PUDDLE_PENALTY

        return reward / 10.0
 
    def saveModel(self, model, file_name):
        model_json = model.to_json()
        with open('weights/' + file_name + '.json', "w") as json_file:
            json_file.write(model_json)
        model.save_weights('weights/' + file_name + '.h5')

    def loadModel(self, file_name):
        json_file = open('weights/' + file_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('weights/' + file_name + '.h5')
        return loaded_model

class NeuralNetwork:
    def __init__(self, input_size, nb_actions):
        self.input_size = input_size
        self.nb_actions = nb_actions

        self.model = Sequential()
        self.model.add(Dense(64, init='lecun_uniform', input_shape=(self.input_size,)))
        self.model.add(Activation('relu'))

        self.model.add(Dense(64, init='lecun_uniform'))
        self.model.add(Activation('relu'))

        self.model.add(Dense(32, init='lecun_uniform'))
        self.model.add(Activation('relu'))

        self.model.add(Dense(self.nb_actions, init='lecun_uniform'))
        self.model.add(Activation('linear'))

        # rms = RMSprop(lr=0.000001, rho=0.6)
        adamOptimizer = Adam(lr=0.01)

        self.model.compile(loss='mse', optimizer=adamOptimizer)
