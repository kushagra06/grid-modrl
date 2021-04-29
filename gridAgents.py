import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import tensorflow as tf
# from keras.models import Sequential, Model
# from keras.layers.core import Dense, Dropout, Activation
# from keras.layers import Input
# from keras.optimizers import RMSprop, Adam
# from keras.layers.merge import Add, Concatenate
# import keras.backend as K


# from grid import GridModule

# config = tf.compat.v1.ConfigProto(device_count = {'GPU':0, 'CPU': 6} ) 

class DQN:
    pass


class QModule:
    def __init__(self, env, eps=1., alpha=0.2, gamma=0.9):
        self.env = env
        self._height = len(self.env.gridworld)
        self._width = len(self.env.gridworld[0])
        self.n_states = self.env.observation_space.n
        self.n_acts = self.env.action_space.n
        self.q = np.zeros([self.n_states, self.n_acts])
        self.q[self.env.goal] = np.full(self.n_acts, 100.)
        self.cumulative_r = 0.
        # for x in range(self._height):
        #     for y in range(self._width):
        #         self.q[(x,y)] = {'LEFT':0, 'UP':0, 'RIGHT':0, 'DOWN':0}
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
    
    def get_action(self, state):
        if np.random.uniform(0,1) < self.eps and self.eps > 0.001:
            a = np.random.randint(self.n_acts)
        else:
            a = np.argmax(self.q[state]) # multiple opt actions??
        return a
    
    def update(self, s, a, new_s, r):
        self.q[s,a] = (1-self.alpha) * self.q[s,a] + self.alpha * (r + self.gamma * np.max(self.q[new_s]))

    
    def get_policy(self, state):
        policy = self._softmax(self.q[state])
        return policy

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def reset(self):
        self.cumulative_r = 0.


# class PolMixModule:
#     def __init__(self, env, n_acts=2, eps=0., alpha=0.1, gamma=0.9):
#         self.sess = tf.Session(config=config)
#         K.set_session(self.sess)
#         self.env = env
#         self.n_acts = n_acts
#         self.n_states = self.env.observation_space.n
#         self.n_acts_module = self.env.action_space.n
#         self.alpha_actor = alpha
#         self.alpha_critic = alpha
#         self.gamma = gamma
#         self.eps = eps
#         state = np.zeros((1,self.env.observation_space.n,))
#         state[0][self.env.cur_state] = 1
#         # self.cur_state = tf.convert_to_tensor(state, np.float)
#         # self.cur_state = np.expand_dims(state,axis=0)
#         self.cur_state = state

#         # self.actor = np.ones([self.n_states, self.n_acts])/self.n_actions
#         # self.critic = np.zeros([self.n_states, self.n_acts])
#         self.actor_in, self.actor_model = self.create_actor()

#         self.critic_in, self.critic_model = self.create_critic()
#         _, self.target_critic_model = self.create_critic()

#         # self._actor_train_fn()
#         self.train_actor = self._actor_train_fn()
#         self.onpolicy_buffer = []

#         self.sess.run(tf.compat.v1.initialize_all_variables())


#     def update_state(self, state_val):
#         state_array = np.zeros((1,self.env.observation_space.n))
#         state_array[0][state_val] = 1
#         # self.cur_state = tf.convert_to_tensor(state_array, np.float)
#         # self.cur_state = np.expand_dims(state_array, axis=0)
#         self.cur_state = state_array

    
#     def _actor_train_fn(self):
#         q_arb = K.placeholder(shape=(None,1),name="q_arb")
#         pol_1 = K.placeholder(shape=(None,self.n_acts_module),name="pol_1")
#         pol_2 = K.placeholder(shape=(None,self.n_acts_module),name="pol_2")
#         actions = K.placeholder(shape=(None,self.n_acts_module),name="actions")

#         lambdas = self.actor_model.output
#         loss = -tf.reshape(q_arb,[-1]) * tf.log(lambdas[:,0]*tf.reduce_sum(pol_1*actions, axis = 1) +
#                                                 lambdas[:,1]*tf.reduce_sum(pol_2*actions, axis = 1) + 0.00001)
#         loss = tf.reduce_sum(loss)
#         params_grad = tf.gradients(lambdas, self.actor_model.trainable_weights)
#         grads = zip(params_grad, self.actor_model.trainable_weights)

#         return K.function([self.actor_model.input, actions, q_arb, pol_1, pol_2], [tf.train.AdamOptimizer(self.alpha_actor).apply_gradients(grads)])
        
#         # adam = Adam(lr=self.alpha_actor)
#         # updates = adam.get_updates(params=self.actor_model.trainable_weights, loss=loss)

#         # self.train_fn = K.function(inputs=[self.actor_model.input, actions, q_arb, pol_1, pol_2], outputs=[], updates=updates)


#     def create_actor(self):
#         state_input = Input(shape=(self.n_states,))
#         h1 = Dense(64, activation='relu')(state_input)
#         h2 = Dense(128, activation='relu')(h1)
#         # output = Dense(self.nb_actions, activation='tanh')(h3)
#         output = Dense(self.n_acts, activation='softmax')(h2)
#         model = Model(input=state_input, output=output)
        
#         return state_input, model


#     def create_critic(self):
#         state_input = Input(shape=(self.n_states,))
#         state_h1 = Dense(64, activation='relu')(state_input)
#         state_h2 = Dense(128, activation='relu')(state_h1)
#         output = Dense(1, activation='linear')(state_h2)
#         model = Model(input=state_input, output=output)

#         adam = Adam(lr=self.alpha_critic)
#         model.compile(loss="mse", optimizer=adam)
        
#         return state_input, model


#     def update(self):
#         if len(self.onpolicy_buffer) > 0:
#             self.train()
#             self.update_target()
#             self.onpolicy_buffer[:] = []

    
#     def add_to_onpolicy_buffer(self, state, action, reward, next_state, done, pi1, pi2):
#         self.onpolicy_buffer.append((state,
#             np.asarray([1 if i == action else 0 for i in range(4)]), reward, next_state, done, pi1, pi2))

 
#     def _train_actor(self, samples):
#         cur_states, actions, rewards, new_states, dones, pi1, pi2 = self.get_attributes_from_sample(samples)
#         # print("cur_states, actions, rewards, new_states, dones, pi1, pi2: ", cur_states, actions, rewards, new_states, dones, pi1, pi2)
#         V_next = self.critic_model.predict(new_states) ## target model??
#         V_curr = self.critic_model.predict(cur_states)
#         predicted_q = rewards + self.gamma * V_next * (1-dones) - V_curr
        
#         self.train_actor([cur_states, actions, predicted_q, pi1, pi2])


#     def _train_critic(self, samples):
#         cur_states, actions, rewards, new_states, dones, pi1, pi2 = self.get_attributes_from_sample(samples)
#         V_next = self.target_critic_model.predict(new_states)
#         target = rewards + self.gamma * V_next * (1 - dones) 

#         evaluation = self.critic_model.fit(cur_states, target, verbose=0)


#     def train(self):
#         samples = self.onpolicy_buffer[:]
#         self._train_critic(samples)
#         self._train_actor(samples)


#     def _update_critic_target(self):
#         critic_model_weights  = self.critic_model.get_weights()
#         critic_target_weights = self.target_critic_model.get_weights()

#         for i in range(len(critic_target_weights)):
#             critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
#         self.target_critic_model.set_weights(critic_target_weights)


#     def update_target(self):
#         # self._update_actor_target()
#         self._update_critic_target()


#     def get_attributes_from_sample(self, random_sample):
#         array = np.array(random_sample)
#         current_states = np.stack(array[:,0]).reshape((array.shape[0],-1))
#         actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
#         rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
#         new_states = np.stack(array[:,3]).reshape((array.shape[0],-1))
#         dones = np.stack(array[:,4]).reshape((array.shape[0],-1))
#         pi1 = np.stack(array[:,5]).reshape((array.shape[0],-1))
#         pi2 = np.stack(array[:,6]).reshape((array.shape[0],-1))
        
#         return current_states, actions, rewards, new_states, dones, pi1, pi2


#     def get_action(self, state):
#         action = self.actor_model.predict(state)
#         return action


# class PolMixAgent:
#     def __init__(self, env1, env2):
#         self.n_arb_acts = 2
#         self.n_acts = 4
#         self.n_states = 16
#         self.batch_size = 8
#         self.module1 = QModule(env1)
#         self.module2 = QModule(env2)
#         self.arbitrator = PolMixModule(env1, self.n_arb_acts)


#     def compute_action(self, state):
#         arb_act = self.arbitrator.getAction(state)[0]
#         m1_policy = self.module1.get_policy(state)
#         m2_policy = self.module2.get_policy(state)

#         mix_policy = arb_act[0] * m1_policy + arb_act[1] * m2_policy
#         policy = mix_policy/sum(mix_policy) ## normalization??
#         action = np.random.choice(np.arange(0,self.n_acts), p=policy)
#         return action
    

#     def update(self, state, action, nextState, reward, epi_so_far):
#         done = 1 if abs(reward) > 100 else 0

#         if epi_so_far % self.batch_size == 0:
#             self.arbitrator.update()
#         else:
#             self.arbitrator.add_to_onpolicy_buffer(state, action, reward, nextState, done, 
#                                                     self.module1.get_policy(state), self.module2.get_policy(state))
    


# env1, env2 = GridModule(15), GridModule(12)
# agent = QModule(env1)

# for i in range(10):
#     print(agent.get_action(0))