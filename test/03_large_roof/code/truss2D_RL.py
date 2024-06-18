
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from collections import deque
import tensorflow as tf
import tensorflow.keras.backend as K
from spektral.layers import GCNConv, GlobalSumPool, GATConv, GCNConv
from tensorflow.keras.layers import Dense, Dropout, Activation, LayerNormalization, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from spektral.utils import gcn_filter, degree_matrix, degree_power
from keras import backend as K
from set_seed_global import seedThis
#set GPU fraction ---------------------------------------------------

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# -------------------------------------------------------------------
import random
import numpy as np

initializer = tf.keras.initializers.GlorotNormal(seedThis)

random.seed(seedThis)
np.random.seed(seedThis)
tf.random.set_seed(seedThis)

attention_head = 1
concat_heads = True
gat_dropout = 0.0

#--------------------------------
# DDPG_Actor_Critic class with keras
#--------------------------------
# Ornstein-Uhlenbeck noise
class OUNoise():
    def __init__(self, mu, theta, sigma):
        self.mu = mu #mean
        self.theta = theta
        self.sigma = sigma #std
        self.dt = 0.0001
    def gen_noise(self,x):
        return self.theta*(self.mu-x)*self.dt + self.sigma*np.random.randn(1)
class multimodes_actor(Model):
    def __init__(self, n_hidden, n_action1, n_action2):
        super().__init__()

        self.gcn_l1_1 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l1_2 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l1_3 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l1_4 = GCNConv(n_hidden, kernel_initializer=initializer)

        self.gcn_l2_1 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_2 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_3 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_4 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_5 = GCNConv(n_hidden, kernel_initializer=initializer)

        self.gcn_l3_1 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l3_2 = GCNConv(n_hidden, kernel_initializer=initializer)

        self.gcn_l4_1 = GCNConv(n_action1, kernel_initializer=initializer)
        self.gcn_l4_2 = GCNConv(n_action2, kernel_initializer=initializer)

        self.sigmoid = Activation('sigmoid')
        self.relu = Activation('relu')
        self.pool = GlobalSumPool()


    def call(self, inputs):
        x_n, A_n, A_s, A_n_ts, A_n_cs, x_p, A_p = inputs

        x_1_1 = self.gcn_l1_1([x_n,A_n])
        x_1_1 = self.relu(x_1_1)

        x_1_2 = self.gcn_l1_2([x_n,A_n])
        x_1_2 = self.relu(x_1_2)

        x_1_3 = self.gcn_l1_3([x_n,A_n])
        x_1_3 = self.relu(x_1_3)

        x_1_4  = self.gcn_l1_4([x_p,A_p])
        x_1_4 = self.relu(x_1_4)
        x_1_4 = self.pool(x_1_4)
        x_1_4 = tf.ragged.stack([x_1_4 for i in range(x_1_1.shape[1])], axis=-1)
        #x3 = tf.transpose(x3)
        batch_shape  = x_1_4.shape[0]
        x_p_shape    = x_1_4.shape[2]
        hidden_shape = x_1_4.shape[1]
        x_1_4 = tf.reshape(x_1_4,(batch_shape,x_p_shape,hidden_shape))

        x_2_1 = self.gcn_l2_1([x_1_1,A_n])
        x_2_1 = self.relu(x_2_1)

        x_2_2 = self.gcn_l2_2([x_1_2,A_n_ts])
        x_2_2 = self.relu(x_2_2)

        x_2_3 = self.gcn_l2_3([x_1_2,A_n_cs])
        x_2_3 = self.relu(x_2_3)

        x_2_4 = self.gcn_l2_4([x_1_3,A_s])
        x_2_4 = self.relu(x_2_4)

        x_2_5 = self.gcn_l2_5([x_1_4,A_n])
        x_2_5 = self.relu(x_2_5)

        x_3_1 = x_2_1 + x_2_2 + x_2_3 + x_2_4 + x_2_5
        x_3_2 = x_2_1 + x_2_2 + x_2_3 + x_2_4 + x_2_5

        x_3_1 = self.gcn_l3_1([x_3_1,A_n])
        x_3_1 = self.relu(x_3_1)

        x_3_2 = self.gcn_l3_2([x_3_2,A_s])
        x_3_2 = self.relu(x_3_2)

        out_1 = self.gcn_l4_1([x_3_1,A_n])
        out_1 = self.sigmoid(out_1)

        out_2 = self.gcn_l4_2([x_3_2,A_n])
        out_2 = self.sigmoid(out_2)

        return out_1, out_2


class multimodes_critic(Model):
    def __init__(self, n_hidden, n_q):
        super().__init__()

        self.gcn_l1_1 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l1_2 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l1_3 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l1_4 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l1_5 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l1_6 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l1_7 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l1_8 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l1_9 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l1_10 = GCNConv(n_hidden, kernel_initializer=initializer)

        self.gcn_l2_1 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_2 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_3 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_4 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_5 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_6 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_7 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_8 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_9 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_10 = GCNConv(n_hidden, kernel_initializer=initializer)
        self.gcn_l2_11 = GCNConv(n_hidden, kernel_initializer=initializer)

        self.sigmoid = Activation('sigmoid')
        self.relu = Activation('relu')

        self.pool = GlobalSumPool()
        self.norm = LayerNormalization()
        self.conc = Concatenate()
        self.bnorm = BatchNormalization()
        #self.dropout = Dropout(0.5)
        self.dense_1 = Dense(n_q, activation="relu", kernel_initializer=initializer)
        self.dense_2 = Dense(n_q, activation="relu", kernel_initializer=initializer)
        self.dense_out = Dense(1)

    def call(self, inputs):
        x_n, A_n, A_s, A_n_ts, A_n_cs, mask, x_p, A_p, self_g, self_t, other_g1, other_t1, other_g2, other_t2 = inputs


        x_1_1 = self.gcn_l1_1([x_n,A_n])
        x_1_1 = self.relu(x_1_1)

        x_1_2 = self.gcn_l1_2([x_n,A_n])
        x_1_2 = self.relu(x_1_2)

        x_1_3 = self.gcn_l1_3([x_n,A_n])
        x_1_3 = self.relu(x_1_3)

        x_1_4  = self.gcn_l1_4([x_p,A_p])
        x_1_4 = self.relu(x_1_4)
        x_1_4 = self.pool(x_1_4)
        x_1_4 = tf.ragged.stack([x_1_4 for i in range(x_1_1.shape[1])], axis=-1)
        batch_shape  = x_1_4.shape[0]
        x_p_shape    = x_1_4.shape[2]
        hidden_shape = x_1_4.shape[1]
        x_1_4 = tf.reshape(x_1_4,(batch_shape,x_p_shape,hidden_shape))

        x_1_5 = self.gcn_l1_5([self_g,A_n])
        x_1_5 = self.relu(x_1_5)

        x_1_6 = self.gcn_l1_6([self_t,A_n])
        x_1_6 = self.relu(x_1_6)

        x_1_7 = self.gcn_l1_7([other_g1,A_n])
        x_1_7 = self.relu(x_1_7)

        x_1_8 = self.gcn_l1_8([other_t1,A_n])
        x_1_8 = self.relu(x_1_8)

        x_1_9 = self.gcn_l1_9([other_g2,A_n])
        x_1_9 = self.relu(x_1_9)

        x_1_10 = self.gcn_l1_10([other_t2,A_n])
        x_1_10 = self.relu(x_1_10)


        x_2_1 = self.gcn_l2_1([x_1_1,A_n])
        x_2_1 = self.relu(x_2_1)

        x_2_2 = self.gcn_l2_2([x_1_2,A_n_ts])
        x_2_2 = self.relu(x_2_2)

        x_2_3 = self.gcn_l2_3([x_1_2,A_n_cs])
        x_2_3 = self.relu(x_2_3)

        x_2_4 = self.gcn_l2_4([x_1_3,A_s])
        x_2_4 = self.relu(x_2_4)

        x_2_5 = self.gcn_l2_5([x_1_5,A_n])
        x_2_5 = self.relu(x_2_5)

        x_2_6 = self.gcn_l2_6([x_1_6,A_n])
        x_2_6 = self.relu(x_2_6)

        x_2_7 = self.gcn_l2_7([x_1_7,A_n])
        x_2_7 = self.relu(x_2_7)

        x_2_8 = self.gcn_l2_8([x_1_8,A_n])
        x_2_8 = self.relu(x_2_8)


        x_2_9 = self.gcn_l2_9([x_1_9,A_n])
        x_2_9 = self.relu(x_2_9)

        x_2_10 = self.gcn_l2_10([x_1_10,A_n])
        x_2_10 = self.relu(x_2_10)

        x_2_11 = self.gcn_l2_11([x_1_4,A_n])
        x_2_11 = self.relu(x_2_11)



        x_3_1 = self.pool(x_2_1)
        x_3_2 = self.pool(x_2_2)
        x_3_3 = self.pool(x_2_3)
        x_3_4 = self.pool(x_2_4)
        x_3_5 = self.pool(x_2_5)
        x_3_6 = self.pool(x_2_6)
        x_3_7 = self.pool(x_2_7)
        x_3_8 = self.pool(x_2_8)
        x_3_9 = self.pool(x_2_9)
        x_3_10 = self.pool(x_2_10)
        x_3_11 = self.pool(x_2_11)

        # concate operation
        q_pred = self.conc([x_3_1,x_3_2,x_3_3,x_3_4,x_3_5,x_3_6,x_3_7,x_3_8,x_3_9,x_3_10,x_3_11])
        #q_pred = self.norm(q_pred)
        # NEURAL NETWORKS
        q_pred = self.dense_1(q_pred)
        q_pred = self.dense_2(q_pred)
        q_pred = self.dense_out(q_pred)

        return q_pred


class multimodals_OneAgent:
    def __init__(self,lr,ep,epd,gamma,a_nn,c_nn,num_action1,num_action2,mu_s,theta_s,sigma_s,mu_t,theta_t,sigma_t, all_agent, batch):
        self.number = 1
        self.lr = lr
        self.epmin=0.05
        self.gamma = gamma
        self.a_nn = a_nn
        self.c_nn = c_nn

        #self.num_state = num_state
        self.num_action1 = num_action1
        self.num_action2 = num_action2
        self.batch_size = 32
        self.tau = 0.005 # soft update
        self.var_actor = None
        self.var_critic= None
        self.noise_geo = []#[NoiseofAction1,NoiseofAction2,...]
        self.noise_topo = []#[NoiseofAction1,NoiseofAction2,...]
        self.update_num =0
        self.update_lr = 0
        self.c_loss = []

        self.create_noise(mu_s,theta_s,sigma_s,mu_t,theta_t,sigma_t)
        # Actor Model
        '''
        During training
          Actor           : Trained
          Actor_target    : Not Trained, but updated
          Critic          : Trained
          Critic_target   : Not Train, but updated
          Prediction during training
            1. In ENV
                Actor_target
            2. In Memory
                Critic Training : Q = R+max(Qt+1)
                  Q(t+1) = Critic(St+1,A1+1)
                  A(t+1) = Actor_target(St+1)
                Actor Training : Q = Critic(S,A)
                  A = Actor(S)
        '''
        self.actor_model = self.create_actor_model()
        self.target_actor_model = self.create_actor_model()

        # Critic Model
        self.critic_model = self.create_critic_model()
        self.target_critic_model = self.create_critic_model()

        # Multiagent properties
        self.all_agent = all_agent
        self.batch_size = batch

        self.update_init()

    def create_noise(self,mu_s,theta_s,sigma_s,mu_t,theta_t,sigma_t):
        for i in range(self.num_action1):
            self.noise_geo.append(OUNoise(mu_s[i], theta_s[i], sigma_s[i]))
        for i in range(self.num_action2):
            self.noise_topo.append(OUNoise(mu_t[i], theta_t[i], sigma_t[i]))

    def act(self,x_n, A_n, A_s, A_n_ts, A_n_cs, x_p, A_p):
        x_n = tf.convert_to_tensor([x_n])
        #A_e = tf.convert_to_tensor(A_e)
        #A_e = tf.sparse.from_dense(A_e)
        A_n = tf.convert_to_tensor([A_n])
        A_s = tf.convert_to_tensor([A_s])
        A_n_ts = tf.convert_to_tensor([A_n_ts])
        A_n_cs = tf.convert_to_tensor([A_n_cs])
        x_p = tf.convert_to_tensor([x_p])
        A_p = tf.convert_to_tensor([A_p])

        out_geo, out_topo = self.actor_model([x_n, A_n, A_s, A_n_ts, A_n_cs, x_p, A_p])
        action_geo = np.array(out_geo)[0]
        if self.noise_geo != None:
            for i in range(len(action_geo)):
                for j in range(len(action_geo[i])):
                    action_geo[i][j] += self.noise_geo[j].gen_noise(action_geo[i][j])

        action_topo = np.array(out_topo)[0]
        if self.noise_topo != None:
            for i in range(len(action_topo)):
                for j in range(len(action_topo[i])):
                    action_topo[i][j] += self.noise_topo[j].gen_noise(action_topo[i][j])

        self.update_num += 1

        return action_geo, action_topo

    def create_actor_model(self):
        model = multimodes_actor(self.a_nn,self.num_action1,self.num_action2)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr,clipnorm=1.))

        return model

    def create_critic_model(self):
        model  = multimodes_critic(self.a_nn,self.c_nn)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr,clipnorm=1.))

        return model

    def _update_actor_target(self,init=None):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        if init==1:
            for i in range(len(actor_target_weights)):
                actor_target_weights[i] = actor_model_weights[i]
        # Soft update using tau
        else:
            for i in range(len(actor_target_weights)):
                actor_target_weights[i] = (actor_model_weights[i]*self.tau) + (actor_target_weights[i]*(1-self.tau))
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self,init=None):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        if init==1:
            for i in range(len(critic_target_weights)):
                critic_target_weights[i] = critic_model_weights[i]
        # Softupdate using tau
        else:
            for i in range(len(critic_target_weights)):
                critic_target_weights[i] = (critic_model_weights[i]*self.tau) + (critic_target_weights[i]*(1-self.tau))
        self.target_critic_model.set_weights(critic_target_weights) #use for train critic_model_weights

    def update(self):
        # Softupdate using tau every self.update_num interval
        if self.update_num == 1000:
            self._update_actor_target()
            self._update_critic_target()
            self.update_num = 0
            print('update target')
        else:
            pass

    def update_init(self):
        self._update_actor_target(1)
        self._update_critic_target(1)


class MADDPG:
    def __init__(self,lr,ep,epd,gamma,a_nn,c_nn,max_mem,num_agents,num_action,mu,theta,sigma,max_poss_n_num=1):
        self.num_agents = num_agents
        self.lr = lr
        self.epint = ep
        self.ep = ep
        self.epd = epd
        self.epmin=0.05
        self.gamma = gamma
        self.a_nn = a_nn
        self.c_nn = c_nn
        self.mu = mu
        self.theta =theta
        self.sigma = sigma
        self.temprp = deque(maxlen = max_mem)
        for i in range(max_poss_n_num):
            self.temprp.append(deque(maxlen = max_mem))
        self.agents = []
        self.update_counter=[]
        self.num_state = [0,0] #[GEO,TOPO] actually does not use in GCN
        self.num_action = num_action #[GEO,TOPO]
        self.batch_size = 32
        self.max_poss_n_num = max_poss_n_num
        self.gen_agents()

    def gen_agents(self):
        '''
        agent = 'agent'
        for i in range(self.num_agents):
            self.agents.append(agent+str(i+1))
            self.agents[-1] = OneAgent(self.lr,self.ep,self.epd,self.gamma,self.a_nn,self.c_nn,self.num_state,self.num_action,self.mu,self.theta,self.sigma,self.num_agents,self.batch_size)
            self.agents[-1].number = i+1
            self.update_counter.append(0)
        '''
        agent = 'agent'
        self.agents.append(agent+str(1))
        self.agents[-1] = multimodals_OneAgent(self.lr,self.ep,self.epd,self.gamma,self.a_nn,self.c_nn,self.num_action[0],self.num_action[1],self.mu[0],self.theta[0],self.sigma[0],self.mu[1],self.theta[1],self.sigma[1],self.num_agents,self.batch_size)

        self.agents[-1].number = 1
        self.update_counter.append(0)

        self.agents.append(agent+str(2))
        self.agents[-1] = multimodals_OneAgent(self.lr,self.ep,self.epd,self.gamma,self.a_nn,self.c_nn,self.num_action[0],self.num_action[1],self.mu[0],self.theta[0],self.sigma[0],self.mu[1],self.theta[1],self.sigma[1],self.num_agents,self.batch_size)
        self.agents[-1].number = 2
        self.update_counter.append(0)

        self.agents.append(agent+str(3))
        self.agents[-1] = multimodals_OneAgent(self.lr,self.ep,self.epd,self.gamma,self.a_nn,self.c_nn,self.num_action[0],self.num_action[1],self.mu[0],self.theta[0],self.sigma[0],self.mu[1],self.theta[1],self.sigma[1],self.num_agents,self.batch_size)
        self.agents[-1].number = 3
        self.update_counter.append(0)

    def remember(self, state, a0_g, a0_t , a1_g, a1_t , a2_g, a2_t, reward, next_state1,next_state2,next_state3, done, n_node):
        #self.temprp[n_node-1].append([state, action_s, action_t, reward, next_state, done]) # use it for now training on same structure
        self.temprp[0].append([state, a0_g, a0_t , a1_g, a1_t , a2_g, a2_t, reward, next_state1,next_state2,next_state3, done])

    # working here 2021 10 18!!
    def train(self):
        '''
        New training function for actor-critic
        Train both Critic and Actor using tf.GradientTape()

        temprp of MADDPG consists of
        [O],[Ag,At],[Rg,Rt],[O]...
        where
        g = geo agent
        t = topo agent

        O are the same for both agent since the agent can observe the whole structure

        '''
        # Batch
        batch_size = self.batch_size
        '''
        if len(self.temprp) < batch_size:
            return
        '''
        rewards = []
        #samples = random.sample(self.temprp, batch_size)
        '''
        temprp_idx = []
        for i in range(len(self.temprp)):
            if len(self.temprp[i]) >= batch_size:
                temprp_idx.append(i)

        if len(temprp_idx) ==0:
            return

        sample_idx = random.choice(temprp_idx)

        samples = random.sample(self.temprp[int(sample_idx)], batch_size)
        '''

        if len(self.temprp[0]) >= batch_size:
            samples = random.sample(self.temprp[0], batch_size)
        else:
            return

        states = [val[0] for val in samples]
        s_x_n = np.array([val[0] for val in states])
        s_A_n = np.array([val[1] for val in states])
        s_A_s = np.array([val[2] for val in states])
        s_A_n_ts = np.array([val[3] for val in states])
        s_A_n_cs = np.array([val[4] for val in states])
        s_mask = np.array([val[5] for val in states])
        s_x_p = np.array([val[6] for val in states])
        s_A_p = np.array([val[7] for val in states])

        next_states_0 = [val[8] for val in samples]
        ns0_x_n = np.array([val[0] for val in next_states_0])
        ns0_A_n = np.array([val[1] for val in next_states_0])
        ns0_A_s = np.array([val[2] for val in next_states_0])
        ns0_A_n_ts = np.array([val[3] for val in next_states_0])
        ns0_A_n_cs = np.array([val[4] for val in next_states_0])
        ns0_mask = np.array([val[5] for val in next_states_0])
        ns0_x_p = np.array([val[6] for val in next_states_0])
        ns0_A_p = np.array([val[7] for val in next_states_0])

        next_states_1 = [val[9] for val in samples]
        ns1_x_n = np.array([val[0] for val in next_states_1])
        ns1_A_n = np.array([val[1] for val in next_states_1])
        ns1_A_s = np.array([val[2] for val in next_states_1])
        ns1_A_n_ts = np.array([val[3] for val in next_states_1])
        ns1_A_n_cs = np.array([val[4] for val in next_states_1])
        ns1_mask = np.array([val[5] for val in next_states_1])
        ns1_x_p = np.array([val[6] for val in next_states_1])
        ns1_A_p = np.array([val[7] for val in next_states_1])

        next_states_2 = [val[10] for val in samples]
        ns2_x_n = np.array([val[0] for val in next_states_2])
        ns2_A_n = np.array([val[1] for val in next_states_2])
        ns2_A_s = np.array([val[2] for val in next_states_2])
        ns2_A_n_ts = np.array([val[3] for val in next_states_2])
        ns2_A_n_cs = np.array([val[4] for val in next_states_2])
        ns2_mask = np.array([val[5] for val in next_states_2])
        ns2_x_p = np.array([val[6] for val in next_states_2])
        ns2_A_p = np.array([val[7] for val in next_states_2])

        #action_0 = [val[1] for val in samples]
        a_0_g    = np.array([val[1] for val in samples])
        a_0_t    = np.array([val[2] for val in samples])

        #action_1 = [val[2] for val in samples]
        a_1_g    = np.array([val[3] for val in samples])
        a_1_t    = np.array([val[4] for val in samples])

        #action_2 = [val[2] for val in samples]
        a_2_g    = np.array([val[5] for val in samples])
        a_2_t    = np.array([val[6] for val in samples])

        rewards = np.array([val[7] for val in samples])
        reward_0 = np.array([val[0] for val in rewards])
        reward_1 = np.array([val[1] for val in rewards])
        reward_2 = np.array([val[2] for val in rewards])

        q_s_a_0 = self.agents[0].critic_model.predict_on_batch([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_mask, s_x_p, s_A_p, a_0_g, a_0_t, a_1_g, a_1_t, a_2_g, a_2_t]) # agent  1
        q_s_a_1 = self.agents[1].critic_model.predict_on_batch([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_mask, s_x_p, s_A_p, a_1_g, a_1_t, a_0_g, a_0_t, a_2_g, a_2_t]) # agent  2
        q_s_a_2 = self.agents[2].critic_model.predict_on_batch([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_mask, s_x_p, s_A_p, a_2_g, a_2_t, a_0_g, a_0_t, a_1_g, a_1_t]) # agent  3

        # future 1 _ f0
        na_0_g_f0, na_0_t_f0 = self.agents[0].target_actor_model.predict_on_batch([ns0_x_n, ns0_A_n, ns0_A_s, ns0_A_n_ts, ns0_A_n_cs, ns0_x_p, ns0_A_p])
        na_1_g_f0, na_1_t_f0 = self.agents[1].target_actor_model.predict_on_batch([ns0_x_n, ns0_A_n, ns0_A_s, ns0_A_n_ts, ns0_A_n_cs, ns0_x_p, ns0_A_p])
        na_2_g_f0, na_2_t_f0 = self.agents[2].target_actor_model.predict_on_batch([ns0_x_n, ns0_A_n, ns0_A_s, ns0_A_n_ts, ns0_A_n_cs, ns0_x_p, ns0_A_p])
        # future 2 _ f1
        na_0_g_f1, na_0_t_f1 = self.agents[0].target_actor_model.predict_on_batch([ns1_x_n, ns1_A_n, ns1_A_s, ns1_A_n_ts, ns1_A_n_cs, ns1_x_p, ns1_A_p])
        na_1_g_f1, na_1_t_f1 = self.agents[1].target_actor_model.predict_on_batch([ns1_x_n, ns1_A_n, ns1_A_s, ns1_A_n_ts, ns1_A_n_cs, ns1_x_p, ns1_A_p])
        na_2_g_f1, na_2_t_f1 = self.agents[2].target_actor_model.predict_on_batch([ns1_x_n, ns1_A_n, ns1_A_s, ns1_A_n_ts, ns1_A_n_cs, ns1_x_p, ns1_A_p])
        # future 3 _ f2
        na_0_g_f2, na_0_t_f2 = self.agents[0].target_actor_model.predict_on_batch([ns2_x_n, ns2_A_n, ns2_A_s, ns2_A_n_ts, ns2_A_n_cs, ns2_x_p, ns2_A_p])
        na_1_g_f2, na_1_t_f2 = self.agents[1].target_actor_model.predict_on_batch([ns2_x_n, ns2_A_n, ns2_A_s, ns2_A_n_ts, ns2_A_n_cs, ns2_x_p, ns2_A_p])
        na_2_g_f2, na_2_t_f2 = self.agents[2].target_actor_model.predict_on_batch([ns2_x_n, ns2_A_n, ns2_A_s, ns2_A_n_ts, ns2_A_n_cs, ns2_x_p, ns2_A_p])

        # future 1 _ f0
        q_s_a_d_0_f0 = self.agents[0].target_critic_model.predict_on_batch([ns0_x_n, ns0_A_n, ns0_A_s, ns0_A_n_ts, ns0_A_n_cs, ns0_mask, ns0_x_p, ns0_A_p,
                                                                            na_0_g_f0, na_0_t_f0, na_1_g_f0, na_1_t_f0, na_2_g_f0, na_2_t_f0])
        q_s_a_d_1_f0 = self.agents[1].target_critic_model.predict_on_batch([ns0_x_n, ns0_A_n, ns0_A_s, ns0_A_n_ts, ns0_A_n_cs, ns0_mask, ns0_x_p, ns0_A_p,
                                                                            na_1_g_f0, na_1_t_f0, na_0_g_f0, na_0_t_f0, na_2_g_f0, na_2_t_f0])
        q_s_a_d_2_f0 = self.agents[2].target_critic_model.predict_on_batch([ns0_x_n, ns0_A_n, ns0_A_s, ns0_A_n_ts, ns0_A_n_cs, ns0_mask, ns0_x_p, ns0_A_p,
                                                                            na_2_g_f0, na_2_t_f0, na_0_g_f0, na_0_t_f0, na_1_g_f0, na_1_t_f0])

        # future 2 _ f1
        q_s_a_d_0_f1 = self.agents[0].target_critic_model.predict_on_batch([ns1_x_n, ns1_A_n, ns1_A_s, ns1_A_n_ts, ns1_A_n_cs, ns1_mask, ns1_x_p, ns1_A_p,
                                                                            na_0_g_f1, na_0_t_f1, na_1_g_f1, na_1_t_f1, na_2_g_f1, na_2_t_f1])
        q_s_a_d_1_f1 = self.agents[1].target_critic_model.predict_on_batch([ns1_x_n, ns1_A_n, ns1_A_s, ns1_A_n_ts, ns1_A_n_cs, ns1_mask, ns1_x_p, ns1_A_p,
                                                                            na_1_g_f1, na_1_t_f1, na_0_g_f1, na_0_t_f1, na_2_g_f1, na_2_t_f1])
        q_s_a_d_2_f1 = self.agents[2].target_critic_model.predict_on_batch([ns1_x_n, ns1_A_n, ns1_A_s, ns1_A_n_ts, ns1_A_n_cs, ns1_mask, ns1_x_p, ns1_A_p,
                                                                            na_2_g_f1, na_2_t_f1, na_0_g_f1, na_0_t_f1, na_1_g_f1, na_1_t_f1])

        # future 3 _ f2
        q_s_a_d_0_f2 = self.agents[0].target_critic_model.predict_on_batch([ns2_x_n, ns2_A_n, ns2_A_s, ns2_A_n_ts, ns2_A_n_cs, ns2_mask, ns2_x_p, ns2_A_p,
                                                                            na_0_g_f2, na_0_t_f2, na_1_g_f2, na_1_t_f2, na_2_g_f2, na_2_t_f2])
        q_s_a_d_1_f2 = self.agents[1].target_critic_model.predict_on_batch([ns2_x_n, ns2_A_n, ns2_A_s, ns2_A_n_ts, ns2_A_n_cs, ns2_mask, ns2_x_p, ns2_A_p,
                                                                            na_1_g_f2, na_1_t_f2, na_0_g_f2, na_0_t_f2, na_2_g_f2, na_2_t_f2])
        q_s_a_d_2_f2 = self.agents[2].target_critic_model.predict_on_batch([ns2_x_n, ns2_A_n, ns2_A_s, ns2_A_n_ts, ns2_A_n_cs, ns2_mask, ns2_x_p, ns2_A_p,
                                                                            na_2_g_f2, na_2_t_f2, na_0_g_f2, na_0_t_f2, na_1_g_f2, na_1_t_f2])

        # Training agent 1-------------------------------------------
        y = np.zeros((len(samples), 1))
        for i, b in enumerate(samples):
            _, _, _, _, done ,_ = b[0], b[1], b[2], b[3], b[4], b[5]
            reward = reward_0[i]
            current_q = q_s_a_0[i]
            if done is 1:
                current_q[0] = reward
            else:
                current_q[0] = reward + self.gamma * (np.amax(q_s_a_d_0_f0[i]) + np.amax(q_s_a_d_0_f1[i]) + np.amax(q_s_a_d_0_f2[i]))/3

            y[i] = current_q

        self.agents[0].critic_model.train_on_batch([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_mask, s_x_p, s_A_p, a_0_g, a_0_t, a_1_g, a_1_t, a_2_g, a_2_t], y)

        with tf.GradientTape() as tape:
            pred_a_0_g, pred_a_0_t = self.agents[0].actor_model([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_x_p, s_A_p], training=True) # Prediction
            pred_a_1_g, pred_a_1_t = self.agents[1].actor_model([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_x_p, s_A_p], training=True) # Prediction
            pred_a_2_g, pred_a_2_t = self.agents[2].actor_model([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_x_p, s_A_p], training=True) # Prediction
            #tape.watch(actor_preds)
            critic_preds_2 = self.agents[0].critic_model([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_mask, s_x_p, s_A_p,
                                                         pred_a_0_g, pred_a_0_t, pred_a_1_g, pred_a_1_t, pred_a_2_g, pred_a_2_t], training=True) # Prediction
            #tape.watch(critic_preds_2)
            actor_loss = -tf.math.reduce_mean(critic_preds_2) # Maximizing value function
        # Optimize critic
        # Train Actor
        actor_grads = tape.gradient(actor_loss, self.agents[0].actor_model.trainable_weights) # Gradient to train actor
        tf.keras.optimizers.Adam(learning_rate=self.agents[0].lr*0.1,clipnorm=1.).apply_gradients(zip(actor_grads,self.agents[0].actor_model.trainable_weights)) # Optimize actor

        # Training agent 2-------------------------------------------
        y = np.zeros((len(samples), 1))
        for i, b in enumerate(samples):
            _, _, _, _, done ,_ = b[0], b[1], b[2], b[3], b[4], b[5]
            reward = reward_1[i]
            current_q = q_s_a_1[i]
            if done is 1:
                current_q[0] = reward
            else:
                current_q[0] = reward + self.gamma * (np.amax(q_s_a_d_1_f0[i]) + np.amax(q_s_a_d_1_f1[i]) + np.amax(q_s_a_d_1_f2[i]))/3

            y[i] = current_q

        self.agents[1].critic_model.train_on_batch([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_mask, s_x_p, s_A_p, a_1_g, a_1_t, a_0_g, a_0_t, a_2_g, a_2_t], y)

        with tf.GradientTape() as tape:
            pred_a_0_g, pred_a_0_t = self.agents[0].actor_model([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_x_p, s_A_p], training=True) # Prediction
            pred_a_1_g, pred_a_1_t = self.agents[1].actor_model([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_x_p, s_A_p], training=True) # Prediction
            pred_a_2_g, pred_a_2_t = self.agents[2].actor_model([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_x_p, s_A_p], training=True) # Prediction
            #tape.watch(actor_preds)
            critic_preds_2 = self.agents[1].critic_model([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_mask, s_x_p, s_A_p,
                                                         pred_a_1_g, pred_a_1_t, pred_a_0_g, pred_a_0_t, pred_a_2_g, pred_a_2_t], training=True) # Prediction
            #tape.watch(critic_preds_2)
            actor_loss = -tf.math.reduce_mean(critic_preds_2) # Maximizing value function
        # Optimize critic
        # Train Actor
        actor_grads = tape.gradient(actor_loss, self.agents[1].actor_model.trainable_weights) # Gradient to train actor
        tf.keras.optimizers.Adam(learning_rate=self.agents[1].lr*0.1,clipnorm=1.).apply_gradients(zip(actor_grads,self.agents[1].actor_model.trainable_weights)) # Optimize actor


        # Training agent 3-------------------------------------------
        y = np.zeros((len(samples), 1))
        for i, b in enumerate(samples):
            _, _, _, _, done ,_ = b[0], b[1], b[2], b[3], b[4], b[5]
            reward = reward_2[i]
            current_q = q_s_a_2[i]
            if done is 1:
                current_q[0] = reward
            else:
                current_q[0] = reward + self.gamma * (np.amax(q_s_a_d_2_f0[i]) + np.amax(q_s_a_d_2_f1[i]) + np.amax(q_s_a_d_2_f2[i]))/3

            y[i] = current_q

        self.agents[2].critic_model.train_on_batch([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_mask, s_x_p, s_A_p, a_2_g, a_2_t, a_0_g, a_0_t, a_1_g, a_1_t], y)

        with tf.GradientTape() as tape:
            pred_a_0_g, pred_a_0_t = self.agents[0].actor_model([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_x_p, s_A_p], training=True) # Prediction
            pred_a_1_g, pred_a_1_t = self.agents[1].actor_model([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_x_p, s_A_p], training=True) # Prediction
            pred_a_2_g, pred_a_2_t = self.agents[2].actor_model([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_x_p, s_A_p], training=True) # Prediction
            #tape.watch(actor_preds)
            critic_preds_2 = self.agents[2].critic_model([s_x_n, s_A_n, s_A_s, s_A_n_ts, s_A_n_cs, s_mask, s_x_p, s_A_p,
                                                         pred_a_2_g, pred_a_2_t, pred_a_0_g, pred_a_0_t, pred_a_1_g, pred_a_1_t], training=True) # Prediction
            #tape.watch(critic_preds_2)
            actor_loss = -tf.math.reduce_mean(critic_preds_2) # Maximizing value function
        # Optimize critic
        # Train Actor
        actor_grads = tape.gradient(actor_loss, self.agents[2].actor_model.trainable_weights) # Gradient to train actor
        tf.keras.optimizers.Adam(learning_rate=self.agents[2].lr*0.1,clipnorm=1.).apply_gradients(zip(actor_grads,self.agents[2].actor_model.trainable_weights)) # Optimize actor
        #print('train ok')


    def update(self):
        # Softupdate using tau every self.update_num interval
        interval = len(self.agents)*100
        if self.agents[0].update_num%interval==0:
            #if self.update_counter[0] == 0:
            for i in range(len(self.agents)):
                self.agents[i].update()
                print('update target agent{}'.format(i+1))


    def update_init(self):
        for num in range(len(self.agents)):
            self.agents[num].update_init()









