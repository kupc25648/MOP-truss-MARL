'''
=====================================================================
Master file for DDPG for training truss structure
=========================== ==========================================
'''

#====================================================================
#Import Part
#インポート部
#====================================================================
from set_seed_global import seedThis
from FEM_2Dtruss import *
from truss2D_GEN import *
from truss2D_RL import *
from truss2D_ENV import *
from tensorflow.keras import backend as K
from utils import dominates, simple_cull, simple_cull_final, CoverQuery, union_rectangles_fastest
import ast
import gc

import pandas as pd
import datetime

import random
import numpy as np
random.seed(seedThis)
np.random.seed(seedThis)


#====================================================================
#Parameter Part
#パラメータ部
#====================================================================

#----------------------------------
# ENV parameters 研究室
#----------------------------------
env_end_step    = 500
env_num_agents  = 2

#----------------------------------
# RL parameters (AC) 研究室
#----------------------------------
lr                   = 0.0000001  # neural network agent learning rate / ＃ニューラルネットワークエージェントの学習率
ep                   = 1   # initial epsilon value / 初期イプシロン値
epd                  = 0.95  # epsilon decay value / イプシロン減衰値
gamma                = 0.99     # reward discount factor / 報酬割引係数
# how neural network is built(neuron in each layer) / ニューラルネットワークの構築方法（各層のニューロン）
a_nn                 = 200      # Hidden layer dim-space
c_nn                 = 200      # Hidden Neuron in critic model
max_mem              = 100000  # maximum length of replay buffer / 再生バッファの最大長
num_agents           = 3
num_action           = [2,3]        # neural network output(action) / ニューラルネットワーク出力（行動）


theta_s = []
mu_s    = []
sigma_s = []
for i in range(num_action[0]):
    theta_s.append(0.1)
    mu_s.append(0.1)
    sigma_s.append(0.1)

theta_t = []
mu_t    = []
sigma_t = []
for i in range(num_action[1]):
    theta_t.append(0.1)
    mu_t.append(0.1)
    sigma_t.append(0.1)



theta = [theta_s,theta_t] #len(theta) == num_action # multiplier
mu = [mu_s,mu_t] #len(mu) == num_action #means
sigma = [sigma_s,sigma_t] #len(sigma) == num_action # noise

train_period         = 0        #(0/1) 0=練習なし　, 1=練習あり
#----------------------------------
# How many game to train 研究室
# トレーニングするゲームの数
#----------------------------------
num_episodes = 10
#----------------------------------
# Load reinforcement learning model 研究室
# 強化学習モデルの読み込み
#----------------------------------
base_num = 2000
#----------------------------------
# How many game to save neural network 研究室
# ニューラルネットワークを保存するゲームの数
#----------------------------------
save_iter = 100

#====================================================================
# Function Part
# 機能部
#====================================================================

#----------------------------------
# Plot function: Function to plot a graph of objective-step in each game
# プロット関数：各ゲームの目的ステップのグラフをプロットする関数
#----------------------------------

game_reward = [0,0,0,0] # reward_g, reward_t, global_r
Utility = []
hyperS = [] # Size of hypervolume
numHV  = [] # Numer of solutions
now = str(datetime.datetime.now().strftime("_%Y-%m-%d_%H_%M_%S"))
def plotforgame():
    # PLOT HYPER VOLUME
    name_of_file = 'Game_{}_finH_{}_finN_{}_R0_{}_R1_{}_R2_{}_Gr_{}.png'.format(
        counter+1,
        round(hyperS[-1],3),
        round(numHV[-1],3),
        round(game_reward[0],3),
        round(game_reward[1],3),
        round(game_reward[2],3),
        round(game_reward[3],3))
    # hypervolume graph
    save_path = 'LogHV_{}-Step'.format('Hyper')+now+'/'
    if not os.path.exists('LogHV_{}-Step'.format('Hyper')+now):
        os.makedirs('LogHV_{}-Step'.format('Hyper')+now)
    name = os.path.join(save_path, name_of_file)
    plt.ylabel('{}'.format('Hyper'))
    plt.xlabel("step")
    plt.plot(hyperS)
    plt.savefig(name)
    plt.close("all")

    # PLOT REWARD
    name_of_file = 'Game_{}_finH_{}_finN_{}_R0_{}_R1_{}_R2_{}_Gr_{}.png'.format(
        counter+1,
        round(hyperS[-1],3),
        round(numHV[-1],3),
        round(game_reward[0],3),
        round(game_reward[1],3),
        round(game_reward[2],3),
        round(game_reward[3],3))
    # hypervolume graphç
    save_path = 'LogUT_{}-Step'.format('Hyper')+now+'/'
    if not os.path.exists('LogUT_{}-Step'.format('Hyper')+now):
        os.makedirs('LogUT_{}-Step'.format('Hyper')+now)
    name = os.path.join(save_path, name_of_file)
    plt.ylabel('{}'.format('Utility'))
    plt.xlabel("step")
    plt.plot(Utility)
    plt.savefig(name)
    plt.close("all")

#----------------------------------
# Reinforcement Learning function: Function to run reinforcement learning loop for 1 game
# 強化学習機能：1ゲームの強化学習ループを実行する機能
#----------------------------------
super_step = 0 # for_countion computational cost

def run(game,train_period=1,savedata=1):
    env1_test.reset()
    intcounter = 0
    x_n, A_n, A_s, A_n_ts, A_n_cs, mask, x_p, A_p, nN_x_n, nN_x_e, nC_e = env1_test.game._game_get_1_state()
    S0 = [x_n, A_n, A_s, A_n_ts, A_n_cs, mask, x_p, A_p, nN_x_n, nN_x_e, nC_e] # Make state data S_structure0
    Pf = [[1,1,0,0,S0,None,None,None,None,None,None,0,0,0,S0,S0,S0,S0]] # Initialize a list of pareto front Pf

    Pf_HV = [[1,1,0,0]] # Initialize a list of pareto front Pf for computing HV
    cur_ref_point_x = 1
    cur_ref_point_y = 1
    my_step = 0

    for_out  = [] # for tracking front in each iteration

    ref_points = [1,1]

    sizeHV_HVl = {0.0:0,
                 0.1:0,
                 0.2:0,
                 0.3:0,
                 0.4:0,
                 0.5:0,
                 0.6:0,
                 0.7:0,
                 0.8:0,
                 0.9:0,
                 10:0,} # dictionary that stroe key:value of HV_size(1decimal):front_lenght
    front_lenght_comp = 0
    new_sum_distance = 0
    new_p_norm_cd = 0
    new_std_cd = 1

    front_no = [[1,1,0,0]]

    while env1_test.over != 1:


        sorted(Pf, key=lambda x: x[0])
        Ppf = Pf # Initialize a list of prospective pareto front Pf
        Ppf_no = [Pf[i][:4] for i in range(len(Pf))]
        for i in range(len(Ppf_no)):
            Ppf_no[i].append(i)
        HV_Bound = [1,1] # Tracking bound on HV in each game to set the reeference points
        R_S = 0
        R_T = 0
        hyperV = 0
        hv_margin = 0.2
        for sol in range(len(Pf)):
            if savedata==1:
                #-----------------------------
                #save file as .txt for initial model / 初期モデルのファイルを.txtとして保存
                name_of_file = 'Game{}_Step{}_Sol_{}.txt'.format(counter+1,env1_test.game.game_step,sol)
                save_path = 'MADDPG_Model_data_txt_Game{}/'.format(counter+1)
                if not os.path.exists('MADDPG_Model_data_txt_Game{}/'.format(counter+1)):
                    os.makedirs('MADDPG_Model_data_txt_Game{}/'.format(counter+1))
                name = os.path.join(save_path, name_of_file)
                env1_test.game.gen_model.savetxt(name)
                #----------------------------------

            # REFERENCE POINT IS THE SAME FOR R_G R_T and G_U
            for_xmax  = [Pf[sol][0]]
            for_ymax  = [Pf[sol][1]]
            # Agent S ----------------------------------------------------


            state = Pf[sol][4] # Use S_structure0 →1 as state data
            #print(state[4].shape)
            #print(state[5].shape)
            '''
            print(state[0])
            print('-----------')
            print(state[1])
            print('-----------')
            print(state[2])
            print('-----------')
            print(state[3])
            print('-----------')
            print(state[4])
            print('-----------')
            print(state[6])
            print('-----------')
            print(state[7])
            print('--------------------------------------------')
            '''
            # Agent 1 ----------------------------------------------------
            a0_geo, a0_topo = reinforcement_learning.agents[0].act(state[0],state[1],state[2],state[3],state[4],state[6],state[7]) # Agent S computes policies
            point_a0, St0_S = env1_test.game._game_modify(state[-3],state[-2],state[-1],[a0_geo, a0_topo]) # modify structure using state[0] for node
            for_front_0 = [ele for ele in front_no]#[[Pf[sol][0],Pf[sol][1],0,0]]

            # Agent 2 ----------------------------------------------------
            a1_geo, a1_topo = reinforcement_learning.agents[1].act(state[0],state[1],state[2],state[3],state[4],state[6],state[7]) # Agent S computes policies
            point_a1, St1_S = env1_test.game._game_modify(state[-3],state[-2],state[-1],[a1_geo, a1_topo]) # modify structure using state[0] for node
            for_front_1 = [ele for ele in front_no]#[[Pf[sol][0],Pf[sol][1],0,0]]

            # Agent 3 ----------------------------------------------------
            a2_geo, a2_topo = reinforcement_learning.agents[2].act(state[0],state[1],state[2],state[3],state[4],state[6],state[7]) # Agent S computes policies
            point_a2, St2_S = env1_test.game._game_modify(state[-3],state[-2],state[-1],[a2_geo, a2_topo]) # modify structure using state[0] for node
            for_front_2 = [ele for ele in front_no]#[[Pf[sol][0],Pf[sol][1],0,0]]

            # DIFFERENCE REWARD COMPONENT
            # DIFFERENCE REWARD FOR AGENT i = REWARD(ALL AGENTS) - REWARD(ALL AGENTS - Agent i)
            # AGENT 1
            # PUT SOLUTIONS FROM AGENT 2,3 to AGENT 1's front
            if point_a1[0] <= 1 and point_a1[1] <= 1  and point_a1[2] <= 1 and point_a1[3] <= 1:
                for_front_0.append(point_a1)
            if point_a2[0] <= 1 and point_a2[1] <= 1 and point_a2[2] <= 1 and point_a2[3] <= 1:
                for_front_0.append(point_a2)
            front_0, max_d_0, dis_d_0, p_cd_0, sum_distance_0, std_cd_0   = simple_cull(for_front_0) # find front from obj 1 and obj2 and c1, c2

            # AGENT 2
            # PUT SOLUTIONS FROM AGENT 1,3 to AGENT 2's front
            if point_a0[0] <= 1 and point_a0[1]  <= 1 and point_a0[2] <= 1 and point_a0[3] <= 1:
                for_front_1.append(point_a0)
            if point_a2[0] <= 1 and point_a2[1] <= 1 and point_a2[2] <= 1 and point_a2[3] <= 1:
                for_front_1.append(point_a2)
            front_1, max_d_1, dis_d_1,p_cd_1, sum_distance_1, std_cd_1 = simple_cull(for_front_1) # find front from obj 1 and obj2 and c1, c2

            # AGENT 3
            # PUT SOLUTIONS FROM AGENT 1,2 to AGENT 3's front
            if point_a0[0] <= 1 and point_a0[1]  <= 1 and point_a0[2] <= 1 and point_a0[3] <= 1:
                for_front_2.append(point_a0)
            if point_a1[0] <= 1 and point_a1[1] <= 1 and point_a1[2] <= 1 and point_a1[3] <= 1:
                for_front_2.append(point_a1)
            front_2, max_d_2, dis_d_2,p_cd_2, sum_distance_2, std_cd_2 = simple_cull(for_front_2) # find front from obj 1 and obj2 and c1, c2



            for_front = [ele for ele in front_no]#[[Pf[sol][0],Pf[sol][1],0,0]]
            if point_a0[0] <= 1 and point_a0[1] <= 1 and point_a0[2] <= 1 and point_a0[3] <= 1:
                for_front.append(point_a0)
                for_xmax.append(point_a0[0])
                for_ymax.append(point_a0[1])
            if point_a1[0] <= 1 and point_a1[1] <= 1 and point_a1[2] <= 1 and point_a1[3] <= 1:
                for_front.append(point_a1)
                for_xmax.append(point_a1[0])
                for_ymax.append(point_a1[1])
            if point_a2[0] <= 1 and point_a2[1] <= 1 and point_a2[2] <= 1 and point_a2[3] <= 1:
                for_front.append(point_a2)
                for_xmax.append(point_a2[0])
                for_ymax.append(point_a2[1])

            front, max_d, dis_d, p_cd, sum_distance, std_cd = simple_cull(for_front) # find front from obj 1 and obj2 and c1, c2

            # COMPUTE REWARD COMPONENTS ------------------------------------
            OPENING = +1  # constants for events
            CLOSING = -1  # -1 has higher priority

            hyperV_0 = union_rectangles_fastest(front_0,OPENING,CLOSING,ref_point=ref_points) # obtain size of HV of agent 1
            hyperV_1 = union_rectangles_fastest(front_1,OPENING,CLOSING,ref_point=ref_points) # obtain size of HV of agent 2
            hyperV_2 = union_rectangles_fastest(front_2,OPENING,CLOSING,ref_point=ref_points) # obtain size of HV of agent 2
            hyperV   = union_rectangles_fastest(front,OPENING,CLOSING,ref_point=ref_points) # obtain size of HV of agents S and T

            compareV = union_rectangles_fastest(Pf_HV,OPENING,CLOSING,ref_point=ref_points)
            Real_compareV = union_rectangles_fastest(Pf_HV,OPENING,CLOSING,ref_point=[1,1])
            RealHV_0 = union_rectangles_fastest(front_0,OPENING,CLOSING,ref_point=[1,1]) # obtain size of HV of agent S
            RealHV_1 = union_rectangles_fastest(front_1,OPENING,CLOSING,ref_point=[1,1]) # obtain size of HV of agent T
            RealHV_2 = union_rectangles_fastest(front_2,OPENING,CLOSING,ref_point=[1,1]) # obtain size of HV of agent T
            RealHV   = union_rectangles_fastest(front,OPENING,CLOSING,ref_point=[1,1]) # obtain size of HV of agents S and T

            # REWARD COMPONENT OF AGENT 1
            hyperV_0 = max([0,hyperV_0-compareV]) # measure improvement of HV
            max_distance_0 = max([0,max_d_0 - env1_test.game.front_max_distance])

            # REWARD COMPONENT OF AGENT 2
            hyperV_1 = max([0,hyperV_1-compareV]) # measure improvement of HV
            max_distance_1 = max([0,max_d_1 - env1_test.game.front_max_distance])

            # REWARD COMPONENT OF AGENT 3
            hyperV_2 = max([0,hyperV_2-compareV]) # measure improvement of HV
            max_distance_2 = max([0,max_d_2 - env1_test.game.front_max_distance])

            # REWARD COMPONENT OF BOTH AGENT S and T
            hyperV  = max([0,hyperV-compareV]) # measure improvement of HV
            max_distance = max([0,max_d - env1_test.game.front_max_distance])

            front.extend(Pf_HV)


            R_0_weight_reward = 0
            R_1_weight_reward = 0
            R_2_weight_reward = 0


            if point_a0[0] <= 1 and point_a0[1] <= 1 and point_a0[2] <= 1 and point_a0[3] <= 1:
                R_0_weight_reward = (1)*max([0,(Pf[sol][0]-point_a0[0])]) + (0)*max([0,(Pf[sol][1]-point_a0[0])])
            if point_a1[0] <= 1 and point_a1[1] <= 1 and point_a1[2] <= 1 and point_a1[3] <= 1:
                R_1_weight_reward = (1/2)*max([0,(Pf[sol][0]-point_a1[0])]) + (1/2)*max([0,(Pf[sol][1]-point_a1[0])])
            if point_a2[0] <= 1 and point_a2[1] <= 1 and point_a2[2] <= 1 and point_a2[3] <= 1:
                R_2_weight_reward = (0)*max([0,(Pf[sol][0]-point_a2[0])]) + (1)*max([0,(Pf[sol][1]-point_a2[0])])



            #print('weight check -------------------------------------------')
            '''
            print('weighted sum {} HV_improvement {} HV {} std {} length {}'.format(round(R_0_weight_reward,3),round((hyperV - hyperV_0),3),round(Real_compareV,3),round(std_cd,3),round(sum_distance,3)))
            print('weighted sum {} HV_improvement {} HV {} std {} length {}'.format(round(R_1_weight_reward,3),round((hyperV - hyperV_1),3),round(Real_compareV,3),round(std_cd,3),round(sum_distance,3)))
            print('weighted sum {} HV_improvement {} HV {} std {} length {}'.format(round(R_2_weight_reward,3),round((hyperV - hyperV_2),3),round(Real_compareV,3),round(std_cd,3),round(sum_distance,3)))
            '''


            weigth = 100
            R_0 = 0.25*R_0_weight_reward/(max([0.25,Real_compareV])*len(Pf)) + 0.25*(hyperV - hyperV_0)/(max([0.25,Real_compareV])*len(Pf)) + 10*(Real_compareV/len(Pf)) - 0.05*max([0,min([1,std_cd])])/len(Pf) + 0.05*sum_distance/(2*(max([0.25,Real_compareV])**0.5)*len(Pf)) #+ 0.005*len(Pf)/20
            R_1 = 0.25*R_1_weight_reward/(max([0.25,Real_compareV])*len(Pf)) + 0.25*(hyperV - hyperV_1)/(max([0.25,Real_compareV])*len(Pf)) + 10*(Real_compareV/len(Pf)) - 0.05*max([0,min([1,std_cd])])/len(Pf) + 0.05*sum_distance/(2*(max([0.25,Real_compareV])**0.5)*len(Pf)) #+ 0.005*len(Pf)/20
            R_2 = 0.25*R_2_weight_reward/(max([0.25,Real_compareV])*len(Pf)) + 0.25*(hyperV - hyperV_2)/(max([0.25,Real_compareV])*len(Pf)) + 10*(Real_compareV/len(Pf)) - 0.05*max([0,min([1,std_cd])])/len(Pf) + 0.05*sum_distance/(2*(max([0.25,Real_compareV])**0.5)*len(Pf)) #+ 0.005*len(Pf)/20
            G_U = (20*Real_compareV/len(Pf)) - (1*std_cd)/len(Pf) + 1*sum_distance/len(Pf) #+ 0.005*len(Pf)/20 # dont want MADDPG agent to make 2 solutions

            cur_ref_point_x = min([(max(for_xmax)),1])
            cur_ref_point_y = min([(max(for_ymax)),1])


            if ((point_a0[2]<= 1) and (point_a0[3]<= 1)) and ((point_a1[2]<= 1) and (point_a1[3]<= 1)) and ((point_a2[2]<= 1) and (point_a2[3]<= 1)):
                Ppf.append([point_a0[0], point_a0[1], point_a0[2], point_a0[3], St0_S, a0_geo, a0_topo ,a1_geo, a1_topo, a2_geo, a2_topo, R_0, R_1, R_2, St0_S, St1_S, St2_S, state])
                Ppf_no.append([point_a0[0], point_a0[1], point_a0[2], point_a0[3],len(Ppf_no)])

                Ppf.append([point_a1[0], point_a1[1], point_a1[2], point_a1[3], St1_S, a0_geo, a0_topo ,a1_geo, a1_topo, a2_geo, a2_topo, R_0, R_1, R_2, St0_S, St1_S, St2_S, state])
                Ppf_no.append([point_a1[0], point_a1[1], point_a1[2], point_a1[3],len(Ppf_no)])

                Ppf.append([point_a2[0], point_a2[1], point_a2[2], point_a2[3], St2_S, a0_geo, a0_topo ,a1_geo, a1_topo, a2_geo, a2_topo, R_0, R_1, R_2, St0_S, St1_S, St2_S, state])
                Ppf_no.append([point_a2[0], point_a2[1], point_a2[2], point_a2[3],len(Ppf_no)])

            elif ((point_a0[2]<= 1) and (point_a0[3]<= 1)) and ((point_a1[2]<= 1) and (point_a1[3]<= 1)):
                Ppf.append([point_a0[0], point_a0[1], point_a0[2], point_a0[3], St0_S, a0_geo, a0_topo ,a1_geo, a1_topo, a2_geo, a2_topo, R_0, R_1, R_2, St0_S, St1_S, random.choice([St0_S, St1_S]), state])
                Ppf_no.append([point_a0[0], point_a0[1], point_a0[2], point_a0[3],len(Ppf_no)])

                Ppf.append([point_a1[0], point_a1[1], point_a1[2], point_a1[3], St1_S, a0_geo, a0_topo ,a1_geo, a1_topo, a2_geo, a2_topo, R_0, R_1, R_2, St0_S, St1_S, random.choice([St0_S, St1_S]), state])
                Ppf_no.append([point_a1[0], point_a1[1], point_a1[2], point_a1[3],len(Ppf_no)])

            elif ((point_a0[2]<= 1) and (point_a0[3]<= 1)) and  ((point_a2[2]<= 1) and (point_a2[3]<= 1)):
                Ppf.append([point_a0[0], point_a0[1], point_a0[2], point_a0[3], St0_S, a0_geo, a0_topo ,a1_geo, a1_topo, a2_geo, a2_topo, R_0, R_1, R_2, St0_S, random.choice([St0_S, St2_S]), St2_S, state])
                Ppf_no.append([point_a0[0], point_a0[1], point_a0[2], point_a0[3],len(Ppf_no)])

                Ppf.append([point_a2[0], point_a2[1], point_a2[2], point_a2[3], St2_S, a0_geo, a0_topo ,a1_geo, a1_topo, a2_geo, a2_topo, R_0, R_1, R_2, St0_S, random.choice([St0_S, St2_S]), St2_S, state])
                Ppf_no.append([point_a2[0], point_a2[1], point_a2[2], point_a2[3],len(Ppf_no)])

            elif ((point_a1[2]<= 1) and (point_a1[3]<= 1)) and ((point_a2[2]<= 1) and (point_a2[3]<= 1)):
                Ppf.append([point_a1[0], point_a1[1], point_a1[2], point_a1[3], St1_S, a0_geo, a0_topo ,a1_geo, a1_topo, a2_geo, a2_topo, R_0, R_1, R_2,  random.choice([St1_S, St2_S]), St1_S, St2_S, state])
                Ppf_no.append([point_a1[0], point_a1[1], point_a1[2], point_a1[3],len(Ppf_no)])

                Ppf.append([point_a2[0], point_a2[1], point_a2[2], point_a2[3], St2_S, a0_geo, a0_topo ,a1_geo, a1_topo, a2_geo, a2_topo, R_0, R_1, R_2,  random.choice([St1_S, St2_S]), St1_S, St2_S, state])
                Ppf_no.append([point_a2[0], point_a2[1], point_a2[2], point_a2[3],len(Ppf_no)])

            elif ((point_a0[2]<= 1) and (point_a0[3]<= 1)):
                Ppf.append([point_a0[0], point_a0[1], point_a0[2], point_a0[3], St0_S, a0_geo, a0_topo ,a1_geo, a1_topo, a2_geo, a2_topo, R_0, R_1, R_2, St0_S, St0_S, St0_S, state])
                Ppf_no.append([point_a0[0], point_a0[1], point_a0[2], point_a0[3],len(Ppf_no)])

            elif ((point_a1[2]<= 1) and (point_a1[3]<= 1)):
                Ppf.append([point_a1[0], point_a1[1], point_a1[2], point_a1[3], St1_S, a0_geo, a0_topo ,a1_geo, a1_topo, a2_geo, a2_topo, R_0, R_1, R_2, St1_S, St1_S, St1_S, state])
                Ppf_no.append([point_a1[0], point_a1[1], point_a1[2], point_a1[3],len(Ppf_no)])

            elif ((point_a2[2]<= 1) and (point_a2[3]<= 1)):
                Ppf.append([point_a2[0], point_a2[1], point_a2[2], point_a2[3], St2_S, a0_geo, a0_topo ,a1_geo, a1_topo, a2_geo, a2_topo, R_0, R_1, R_2, St2_S, St2_S, St2_S, state])
                Ppf_no.append([point_a2[0], point_a2[1], point_a2[2], point_a2[3],len(Ppf_no)])



            #print(R_S)
            #print(R_T)
            #print(hyperV)
            #print(env1_test.game.current_hv)
            game_reward[0] += R_0 # reward_g
            game_reward[1] += R_1 # reward_t
            game_reward[2] += R_2 # reward_t
            game_reward[3] += G_U# global_r max(hyperV) is already 1, max(Length) is 2 so normailze with 2

            my_step += 3 # using analysis


        # Game move to nextstep and check if the end condition is met
        if env1_test.game.game_step == env1_test.game.end_step:
            env1_test.game.done_counter = 1
        env1_test.check_over()

        if env1_test.game.done_counter != 1:
            front_no, new_max_distance, new_dis_distance, new_p_norm_cd, new_sum_distance, new_std_cd, front_no_edges = simple_cull(Ppf_no,True) # find front data in Ppf

            OPENING = +1  # constants for events
            CLOSING = -1  # -1 has higher priority
            for i in range(len(front_no)):
                if front_no[i][0] > 1:
                    front_no[i][0] = 1
                if front_no[i][1] > 1:
                    front_no[i][1] = 1

            for i in range(len(front_no_edges)):
                if front_no_edges[i][0] > 1:
                    front_no_edges[i][0] = 1
                if front_no_edges[i][1] > 1:
                    front_no_edges[i][1] = 1
            #print(front_no)
            env1_test.game.current_hv   = union_rectangles_fastest(front_no,OPENING,CLOSING,ref_point=[1,1]) # obtain size of current HV
            #print(env1_test.game.current_hv)
            env1_test.game.front_max_distance = new_max_distance
            env1_test.game.front_dis_distance = new_dis_distance

            #print(area)
            # check length give the same HV
            size_HV_key = round(env1_test.game.current_hv,1)
            #front_lenght_comp = 0 # reward component from front_lenght
            if sizeHV_HVl[size_HV_key] < new_sum_distance:
                #front_lenght_comp = new_sum_distance - sizeHV_HVl[size_HV_key]
                sizeHV_HVl[size_HV_key] = new_sum_distance
            else:
                pass

            front_lenght_comp = sizeHV_HVl[size_HV_key] # reward component from front_lenght
            front = []
            front_forref = []

            for i in range(len(front_no)):
                front_forref.append(Ppf[int(front_no[i][-1])])
                for_out.append('{} {} {}'.format(
                        env1_test.game.game_step,
                        front_no[i][0],
                        front_no[i][1]))
            for i in range(len(front_no_edges)):
                #print(front_no[i][-1])
                front.append(Ppf[int(front_no_edges[i][-1])])

            MAX_PARETO_SIZE = 50
            current_min_x = []
            current_min_y = []
            for sol in range(len(front)):

                current_min_x.append(front[sol][0])
                current_min_y.append(front[sol][1])

                #if front[sol][-1][-4] is None and front[sol][-1][-3] is None:
                #print(front[sol][-1])
                #print('-=========================================================================================================================')
                #print(front[sol][-2])
                front[sol][-1][-5], front[sol][-1][-4] = pareto_state_data(front,index=sol) # Modify P and Ap using new pareto graph
                front[sol][-2][-5], front[sol][-2][-4] = pareto_state_data(front,index=sol) # Modify P and Ap using new pareto graph
                front[sol][-3][-5], front[sol][-3][-4] = pareto_state_data(front,index=sol) # Modify P and Ap using new pareto graph
                front[sol][-4][-5], front[sol][-4][-4] = pareto_state_data(front,index=sol) # Modify P and Ap using new pareto graph

                ## MAKE PARETO BLOCK MEMORY FOR front[sol][-1]
                pareto_size_now = front[sol][-1][-5].shape[0]#max([front[sol][-1][-4].shape[0],front[sol][-2][-4].shape[0],front[sol][4][-4].shape[0]])
                add_block_size  = MAX_PARETO_SIZE - pareto_size_now #front[sol][-1][-4].shape[0]
                if add_block_size > 0:
                    # for node feature (pareto)
                    lower    = np.zeros((add_block_size,4))
                    front[sol][-1][-5] = np.block([[front[sol][-1][-5]],
                                                  [lower]])
                    # for adjacency (pareto)
                    topRight = np.zeros((pareto_size_now,add_block_size))
                    lowLeft  = np.zeros((add_block_size,pareto_size_now))
                    lowRight = np.zeros((add_block_size,add_block_size))
                    front[sol][-1][-4] = np.block([[front[sol][-1][-4],topRight],
                                                  [lowLeft,lowRight]])
                elif add_block_size < 0:
                    front[sol][-1][-5] = front[sol][-1][-5][:MAX_PARETO_SIZE,:]
                    front[sol][-1][-4] = front[sol][-1][-4][:MAX_PARETO_SIZE,:MAX_PARETO_SIZE]
                else:
                    pass

                pareto_size_now = front[sol][-2][-5].shape[0]#max([front[sol][-1][-4].shape[0],front[sol][-2][-4].shape[0],front[sol][4][-4].shape[0]])
                add_block_size  = MAX_PARETO_SIZE - pareto_size_now #front[sol][-2][-4].shape[0]
                if add_block_size > 0:
                    # for node feature (pareto)
                    lower    = np.zeros((add_block_size,4))
                    front[sol][-2][-5] = np.block([[front[sol][-2][-5]],
                                                  [lower]])
                    # for adjacency (pareto)
                    topRight = np.zeros((pareto_size_now,add_block_size))
                    lowLeft  = np.zeros((add_block_size,pareto_size_now))
                    lowRight = np.zeros((add_block_size,add_block_size))
                    front[sol][-2][-4] = np.block([[front[sol][-2][-4],topRight],
                                                  [lowLeft,lowRight]])
                elif add_block_size < 0:
                    front[sol][-2][-5] = front[sol][-2][-5][:MAX_PARETO_SIZE,:]
                    front[sol][-2][-4] = front[sol][-2][-4][:MAX_PARETO_SIZE,:MAX_PARETO_SIZE]
                else:
                    pass

                pareto_size_now = front[sol][-3][-5].shape[0]#max([front[sol][-1][-4].shape[0],front[sol][-2][-4].shape[0],front[sol][4][-4].shape[0]])
                add_block_size  = MAX_PARETO_SIZE - pareto_size_now #front[sol][-2][-4].shape[0]
                if add_block_size > 0:
                    # for node feature (pareto)
                    lower    = np.zeros((add_block_size,4))
                    front[sol][-3][-5] = np.block([[front[sol][-3][-5]],
                                                  [lower]])
                    # for adjacency (pareto)
                    topRight = np.zeros((pareto_size_now,add_block_size))
                    lowLeft  = np.zeros((add_block_size,pareto_size_now))
                    lowRight = np.zeros((add_block_size,add_block_size))
                    front[sol][-3][-4] = np.block([[front[sol][-3][-4],topRight],
                                                  [lowLeft,lowRight]])
                elif add_block_size < 0:
                    front[sol][-3][-5] = front[sol][-3][-5][:MAX_PARETO_SIZE,:]
                    front[sol][-3][-4] = front[sol][-3][-4][:MAX_PARETO_SIZE,:MAX_PARETO_SIZE]
                else:
                    pass


                pareto_size_now = front[sol][-4][-5].shape[0]#max([front[sol][-1][-4].shape[0],front[sol][-2][-4].shape[0],front[sol][4][-4].shape[0]])
                add_block_size  = MAX_PARETO_SIZE - pareto_size_now #front[sol][-2][-4].shape[0]
                if add_block_size > 0:
                    # for node feature (pareto)
                    lower    = np.zeros((add_block_size,4))
                    front[sol][-4][-5] = np.block([[front[sol][-4][-5]],
                                                  [lower]])
                    # for adjacency (pareto)
                    topRight = np.zeros((pareto_size_now,add_block_size))
                    lowLeft  = np.zeros((add_block_size,pareto_size_now))
                    lowRight = np.zeros((add_block_size,add_block_size))
                    front[sol][-4][-4] = np.block([[front[sol][-4][-4],topRight],
                                                  [lowLeft,lowRight]])
                elif add_block_size < 0:
                    front[sol][-4][-5] = front[sol][-4][-5][:MAX_PARETO_SIZE,:]
                    front[sol][-4][-4] = front[sol][-4][-4][:MAX_PARETO_SIZE,:MAX_PARETO_SIZE]
                else:
                    pass






                ## MAKE PARETO BLOCK MEMORY FOR front[sol][4]
                pareto_size_now = front[sol][4][-5].shape[0]
                add_block_size  = MAX_PARETO_SIZE - pareto_size_now #front[sol][4][-4].shape[0]
                if add_block_size > 0:
                    # for node feature (pareto)
                    lower    = np.zeros((add_block_size,4))
                    front[sol][4][-5] = np.block([[front[sol][4][-5]],
                                                  [lower]])
                    # for adjacency (pareto)
                    topRight = np.zeros((pareto_size_now,add_block_size))
                    lowLeft  = np.zeros((add_block_size,pareto_size_now))
                    lowRight = np.zeros((add_block_size,add_block_size))
                    front[sol][4][-4] = np.block([[front[sol][4][-4],topRight],
                                                  [lowLeft,lowRight]])
                elif add_block_size < 0:
                    front[sol][4][-5] = front[sol][4][-5][:MAX_PARETO_SIZE,:]
                    front[sol][4][-4] = front[sol][4][-4][:MAX_PARETO_SIZE,:MAX_PARETO_SIZE]
                else:
                    pass

                if train_period==1:
                    if front[sol][5] is None:
                        '''
                        print('------------------------------------------------')
                        print('s_x_p shape {}'.format(front[sol][4][-4].shape))
                        print('ns0_x_p shape {}'.format(front[sol][-1][-4].shape))
                        print('ns1_x_p shape {}'.format(front[sol][-2][-4].shape))
                        try:
                            print('A0g shape {}'.format(front[sol][5].shape))
                        except:
                            print(front[sol])
                        print('A0t shape {}'.format(front[sol][6].shape))
                        print('A1g shape {}'.format(front[sol][7].shape))
                        print('A1t shape {}'.format(front[sol][8].shape))
                        '''
                        pass
                    else:
                        reinforcement_learning.remember(
                        front[sol][-1],
                        front[sol][5],front[sol][6],
                        front[sol][7],front[sol][8],
                        front[sol][9],front[sol][10],
                        [front[sol][11],front[sol][12],front[sol][13]],
                        front[sol][14],
                        front[sol][15],
                        front[sol][16],
                        env1_test.game.done_counter,1) # Put front in replay buffer
                else:
                    pass


            #----------------------------------
            # SET REF POINT EVEERY STEP USING CURRENT FRONT

            min_edge_on_x_axis = min(current_min_x)
            min_edge_on_y_axis = min(current_min_y)

            edges_on_x_axis = [] # y value
            edges_on_y_axis = [] # x value

            for sol in range(len(front_forref)):
                if round(front_forref[sol][0],2) == round(min_edge_on_x_axis,2):
                    edges_on_x_axis.append(front_forref[sol][1])
                if round(front_forref[sol][1],2) == round(min_edge_on_y_axis,2):
                    edges_on_y_axis.append(front_forref[sol][0])

            #ref_points = [min(ref_points[0],min(edges_on_y_axis)),min(ref_points[1],min(edges_on_x_axis))]
            ref_points = [min([1,ref_points[0]+hv_margin]),min([1,ref_points[1]+hv_margin])]


            #----------------------------------
            # train the neural network agent
            if train_period==1:
                reinforcement_learning.train()
                reinforcement_learning.update()
            else:
                pass
            Pf = front
            Pf_HV = [x[:4] for x in Pf]
            #print(Pf_HV)

            #----------------------------------
            # Add data for the plot
            #print(R_S)
            #print(R_T)
            #game_reward[0] += R_S # reward_g
            #game_reward[1] += R_T # reward_t
            #game_reward[2] += hyperV   # global_r
            hyperS.append(env1_test.game.current_hv)
            Utility.append(game_reward[3])
            numHV.append(len(Pf))
            #----------------------------------
            # Print out result on the console
            print('Step {} || Hypervolumes {} n {} || R0 {} R1 {} R2 {} Gr {}'.format(
                env1_test.game.game_step,
                round(hyperS[-1],3),
                len(Pf),
                round(game_reward[0],6),
                round(game_reward[1],6),
                round(game_reward[2],6),
                round(game_reward[3],6)))
            #----------------------------------
            env1_test.game.step()

        else:
            front_no, new_max_distance, new_dis_distance, new_p_norm_cd, new_sum_distance, new_std_cd, front_no_edges = simple_cull_final(Ppf_no,True) # find front data in Ppf

            OPENING = +1  # constants for events
            CLOSING = -1  # -1 has higher priority
            for i in range(len(front_no)):
                if front_no[i][0] > 1:
                    front_no[i][0] = 1
                if front_no[i][1] > 1:
                    front_no[i][1] = 1

            for i in range(len(front_no_edges)):
                if front_no_edges[i][0] > 1:
                    front_no_edges[i][0] = 1
                if front_no_edges[i][1] > 1:
                    front_no_edges[i][1] = 1
            #print(front_no)
            env1_test.game.current_hv   = union_rectangles_fastest(front_no,OPENING,CLOSING,ref_point=[1,1]) # obtain size of current HV
            #print(env1_test.game.current_hv)
            env1_test.game.front_max_distance = new_max_distance
            env1_test.game.front_dis_distance = new_dis_distance

            #print(area)
            # check length give the same HV
            size_HV_key = round(env1_test.game.current_hv,1)
            #front_lenght_comp = 0 # reward component from front_lenght
            if sizeHV_HVl[size_HV_key] < new_sum_distance:
                #front_lenght_comp = new_sum_distance - sizeHV_HVl[size_HV_key]
                sizeHV_HVl[size_HV_key] = new_sum_distance
            else:
                pass

            front_lenght_comp = sizeHV_HVl[size_HV_key] # reward component from front_lenght
            front = []
            front_forref = []

            for i in range(len(front_no)):
                front_forref.append(Ppf[int(front_no[i][-1])])
                for_out.append('{} {} {}'.format(
                        env1_test.game.game_step,
                        front_no[i][0],
                        front_no[i][1]))
            for i in range(len(front_no_edges)):
                #print(front_no[i][-1])
                front.append(Ppf[int(front_no_edges[i][-1])])

            Pf = front
            Pf_HV = [x[:4] for x in Pf]

            hyperS.append(env1_test.game.current_hv)
            Utility.append(game_reward[3])
            numHV.append(len(Pf))
            #----------------------------------
            # Print out result on the console
            print('Step {} || Hypervolumes {} n {} || R0 {} R1 {} R2 {} Gr {}'.format(
                env1_test.game.game_step,
                round(hyperS[-1],3),
                len(Pf),
                round(game_reward[0],6),
                round(game_reward[1],6),
                round(game_reward[2],6),
                round(game_reward[3],6)))
            #----------------------------------
            env1_test.game.step()

    if savedata==1:
        for sol in range(len(Pf)):
            # modifiy str
            state = Pf[sol][-1] # Use S_structure0 →1 as state data
            env1_test.game._set_model(state[8], state[9])
            #-----------------------------
            #save file as .txt for initial model / 初期モデルのファイルを.txtとして保存
            name_of_file = 'Game{}_Step{}_Sol_{}.txt'.format(counter+1,env1_test.game.game_step,sol)
            save_path = 'MADDPG_Model_data_txt_Game{}/'.format(counter+1)
            if not os.path.exists('MADDPG_Model_data_txt_Game{}/'.format(counter+1)):
                os.makedirs('MADDPG_Model_data_txt_Game{}/'.format(counter+1))
            name = os.path.join(save_path, name_of_file)
            env1_test.game.gen_model.savetxt(name)
            #----------------------------------

    if savedata==1:
        front_iteration = 'MADDPG_Model_data_txt_Game{}/out.txt'.format(counter+1)
        new_file1 = open(front_iteration, "w+")
        for i in range(len(for_out)):
            new_file1.write(" {}\r\n".format(for_out[i]))
        new_file1.close()

    return my_step

# ----------------------------------
# Save and Restore Agents
# ----------------------------------
if base_num!= 0:
    base_pickle_path = '{}pickle_base/'.format(base_num)
    all_base_name_of_Actor_pickle = []
    all_base_name_of_Critic_pickle = []
    for i in range(num_agents):
        all_base_name_of_Actor_pickle.append("Agent{}_Actor_pickle".format(i+1))
        all_base_name_of_Critic_pickle.append("Agent{}_Critic_pickle".format(i+1))

# ----------------------------------
# Main program
# ----------------------------------
reinforcement_learning = MADDPG(lr,ep,epd,gamma,a_nn,c_nn,max_mem,num_agents,num_action,mu,theta,sigma)
if base_num!= 0:
    try:
        for i in range(num_agents):
            base_Actor_picklename = os.path.join(base_pickle_path, all_base_name_of_Actor_pickle[i])
            base_Critic_picklename = os.path.join(base_pickle_path, all_base_name_of_Critic_pickle[i])
            reinforcement_learning.agents[i].actor_model.load_weights(base_Actor_picklename).expect_partial()
            reinforcement_learning.agents[i].critic_model.load_weights(base_Critic_picklename).expect_partial()
            reinforcement_learning.agents[i].target_actor_model.load_weights(base_Actor_picklename).expect_partial()
            reinforcement_learning.agents[i].target_critic_model.load_weights(base_Critic_picklename).expect_partial()
            print("Load model success!")
    except:
        print("No model file to restore")

counter = 0 # Counter for game

env_model = None
game = None
game_choice = None

while counter < num_episodes:
    # ==========================================
    # measure some game scheme
    # ==========================================
    span_x = [5,5,5,5,5,5,5]
    span_y = [8]
    num_x = len(span_x) + 1
    num_y = len(span_y) + 1
    tar_y = [4,3,2.5,2,2,2.5,3,4]

    dmin  = 0.3
    sum_x = [ sum(span_x[:i]) for i in range(len(span_x)) ]
    tar_xy = [ [sum_x[i],tar_y[i]] for i in range(len(span_x))]
    tar_xy.append([sum(span_x), tar_y[-1]])

    loadx = 0
    loady = -120*1000


    truss_type='roof'
    support_case=1
    topo_code=None

    if env_model == None:
        env_model = gen_model(num_x,num_y,span_x,span_y,tar_y,dmin,loadx,loady,truss_type,support_case,topo_code)
    else:
        env_model.re_value(num_x,num_y,span_x,span_y,tar_y,dmin,loadx,loady,truss_type,support_case,topo_code)

    if game == None:
        game = Game_research04(env_end_step,env_model,env_num_agents)
    else:
        game.re_game(env_end_step,env_model,env_num_agents)
    env1_test = ENV(game)
    print('Episode{}'.format(counter+1))

    super_step += run(game,0,savedata=1)
    plotforgame()
    game_reward = [0,0,0,0] # reward_g, reward_t, global_r
    cons_ratio = [0,0] # 2 constraint
    objf_1 = [] # Volume
    objf_2 = [] # Target shape
    hyperS = []
    Utility =[]
    numHV  = []
    counter += 1 # game counter += 1
    #game.gen_model.render('Training_game_{}.png'.format(counter))
    gc.collect() #release unreferenced memory


    print('TOTAL CURRENT ANALYSIS=========================================')
    print(super_step)
    print('=====================================================================')
