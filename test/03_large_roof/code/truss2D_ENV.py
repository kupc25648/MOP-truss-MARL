from set_seed_global import seedThis
from collections import Counter
from spektral.utils import degree_power
from utils import dominates, simple_cull, simple_cull_final, CoverQuery, union_rectangles_fastest
from operator import itemgetter
import numpy as np
import random

import math
import gc
np.random.seed(seedThis)
random.seed(seedThis)

MAX_MEM_NO = 5 # this is the max index start from 0
MAX_FRONT = 50

'''
======================================================
Helper class and functions
======================================================
'''
def pareto_state_data(pf, index=0):
    # pf = [obj1,obj2,0,0,S0,None,None,None,None,None,None,0,0,S0]]
    # obj1 = point[0]
    # obj2 = point[1]

    x_pf = np.zeros((len(pf),4),dtype=np.float32)
    for i in range(len(pf)):
        x_pf[i][0] = pf[i][0]
        x_pf[i][1] = pf[i][1]
        if i == index:
            x_pf[i][2] = 1
        x_pf[i][3] = len(pf)/MAX_FRONT
    A_pf = np.eye(len(pf),dtype=np.float32)
    for i in range(len(pf)-1):
        A_pf[i][i+1] = 1
        A_pf[i+1][i] = 1

    D_pf = degree_power(A_pf,-1/2)            # for GCN CASE
    A_pf = np.matmul(D_pf,np.matmul(A_pf,D_pf)) # for GCN CASE
    return x_pf, A_pf

def state_data(generated):
    '''
    The generated is gen_model
    This function creates
        1. x_n   : [n_node,n_node_features] : Node Feature Matrix of node data
        2. A_n   : [n_node,n_node]          : Adjacency Matrix of node data
        3. x_e   : [n_element,n_element_features] : Node Feature Matrix of element data
        4. A_e   : [n_element,n_element]          : Adjacency Matrix of element data
    STATE FOR BOTH AGENT
    '''
    x_n  = np.zeros((len(generated.model.nodes),13),dtype=np.float32)
    A_n  = np.zeros((len(generated.model.nodes),len(generated.model.nodes)),dtype=np.float32)
    A_s  = np.zeros((len(generated.model.nodes),len(generated.model.nodes)),dtype=np.float32)

    A_n_ts  = np.zeros((len(generated.model.nodes),len(generated.model.nodes)),dtype=np.float32)
    A_n_cs  = np.zeros((len(generated.model.nodes),len(generated.model.nodes)),dtype=np.float32)

    #One_n  = np.ones((len(generated.model.nodes),len(generated.model.nodes)),dtype=np.float32)
    Dia_n  = np.eye(len(generated.model.nodes),dtype=np.float32)

    mask = np.zeros((len(generated.model.nodes),len(generated.model.nodes)),dtype=np.float32)


    for i in range(len(generated.model.nodes)):
        x_n[i][0]  = generated.model.nodes[i].coord[0] # x coord (Norm max-min!=0)
        x_n[i][1]  = generated.model.nodes[i].coord[1] # y coord (Norm max-min!=0)
        x_n[i][2]  = generated.model.nodes[i].res[0] # x support
        x_n[i][3]  = generated.model.nodes[i].res[1] # y support
        x_n[i][4]  = abs(generated.model.nodes[i].has_loady) # y load (Norm max-min!=0)
        x_n[i][5]  = generated.model.nodes[i].top_node # is upper node
        x_n[i][6]  = abs(generated.model.nodes[i].top_node-1) # is lower node
        x_n[i][7]  = generated.model.nodes[i].max_up # move up range (Norm max-min==0)
        x_n[i][8]  = generated.model.nodes[i].max_down # move down range (Norm max-min==0)
        x_n[i][9]  = generated.model.nodes[i].target*generated.model.nodes[i].top_node/(generated.model.nodes[i].coord[1]+1e-6)
        x_n[i][10] = abs(generated.model.nodes[i].global_d[1][0]) # Normalized deformation (Norm max-min==0)
        factor = 0.5
        if x_n[i][10]/generated.max_deformation > 1:
            factor = 1
        x_n[i][11] = min([(x_n[i][10]/generated.max_deformation),1])*factor # is violate deformation constraint
        x_n[i][12] = int((x_n[i][10]/generated.max_deformation)>1) # is violate deformation constraint

    for i in range(len(generated.model.elements)):

        A_n[generated.model.elements[i].nodes[0].name-1][generated.model.elements[i].nodes[1].name-1] = 1
        A_n[generated.model.elements[i].nodes[1].name-1][generated.model.elements[i].nodes[0].name-1] = 1

        A_s[generated.model.elements[i].nodes[0].name-1][generated.model.elements[i].nodes[1].name-1] = generated.model.elements[i].area/(generated.truss[-1][0]*1e-4)
        A_s[generated.model.elements[i].nodes[1].name-1][generated.model.elements[i].nodes[0].name-1] = generated.model.elements[i].area/(generated.truss[-1][0]*1e-4)

        mask[generated.model.elements[i].nodes[0].name-1][generated.model.elements[i].nodes[1].name-1] = 1
        mask[generated.model.elements[i].nodes[1].name-1][generated.model.elements[i].nodes[0].name-1] = 1

        factor = 0.5
        if generated.model.elements[i].prop_yeield > 1:
            factor = 1
        if generated.model.elements[i].iscompress == 0:
            A_n_ts[generated.model.elements[i].nodes[0].name-1][generated.model.elements[i].nodes[1].name-1] = min([generated.model.elements[i].prop_yeield,1])*factor
            A_n_ts[generated.model.elements[i].nodes[1].name-1][generated.model.elements[i].nodes[0].name-1] = min([generated.model.elements[i].prop_yeield,1])*factor
        else:
            A_n_cs[generated.model.elements[i].nodes[0].name-1][generated.model.elements[i].nodes[1].name-1] = min([generated.model.elements[i].prop_yeield,1])*factor
            A_n_cs[generated.model.elements[i].nodes[1].name-1][generated.model.elements[i].nodes[0].name-1] = min([generated.model.elements[i].prop_yeield,1])*factor

    x_n = (x_n-x_n.min(axis=0)) / (x_n.max(axis=0)-x_n.min(axis=0)+1e-6) # Normalize x_n

    A_n = A_n + Dia_n

    D_n = degree_power(A_n,-1/2)            # for GCN CASE
    A_n = np.matmul(D_n,np.matmul(A_n,D_n)) # for GCN CASE

    return x_n, A_n, A_s, A_n_ts, A_n_cs, mask


def state_data_not_norm(generated):
    '''
    The generated is gen_model
    This function creates
        1. x_n   : [n_node,n_node_features] : Node Feature Matrix of node data
        2. A_n   : [n_node,n_node]          : Adjacency Matrix of node data
        3. x_e   : [n_element,n_element_features] : Node Feature Matrix of element data
        4. A_e   : [n_element,n_element]          : Adjacency Matrix of element data
    STATE FOR BOTH AGENT
    '''
    x_n  = np.zeros((len(generated.model.nodes),12),dtype=np.float32)
    A_n  = np.zeros((len(generated.model.nodes),len(generated.model.nodes)),dtype=np.float32)
    #One_n  = np.ones((len(generated.model.nodes),len(generated.model.nodes)),dtype=np.float32)
    Dia_n  = np.eye(len(generated.model.nodes),dtype=np.float32)

    x_e  = np.zeros((len(generated.model.elements),21),dtype=np.float32)
    A_e  = np.zeros((len(generated.model.elements),len(generated.model.elements)),dtype=np.float32)
    #One_e  = np.ones((len(generated.model.elements),len(generated.model.elements)),dtype=np.float32)
    Dia_e  = np.eye(len(generated.model.elements),dtype=np.float32) # not use

    c_e  = np.zeros((len(generated.model.elements),len(generated.model.nodes)),dtype=np.float32)

    for i in range(len(generated.model.nodes)):
        x_n[i][0]  = generated.model.nodes[i].coord[0] # x coord (Norm max-min!=0)
        x_n[i][1]  = generated.model.nodes[i].coord[1] # y coord (Norm max-min!=0)
        x_n[i][2]  = generated.model.nodes[i].res[0] # x support
        x_n[i][3]  = generated.model.nodes[i].res[1] # y support
        x_n[i][4]  = abs(generated.model.nodes[i].has_loady) # y load (Norm max-min!=0)
        x_n[i][5]  = generated.model.nodes[i].top_node # is upper node
        x_n[i][6]  = abs(generated.model.nodes[i].top_node-1) # is lower node
        x_n[i][7]  = generated.model.nodes[i].max_up # move up range (Norm max-min==0)
        x_n[i][8]  = generated.model.nodes[i].max_down # move down range (Norm max-min==0)
        x_n[i][9]  = generated.model.nodes[i].target*generated.model.nodes[i].top_node/(generated.model.nodes[i].coord[1]+1e-6)
        x_n[i][10] = abs(generated.model.nodes[i].global_d[1][0]) # Normalized deformation (Norm max-min==0)
        x_n[i][11] = int((x_n[i][10]/generated.max_deformation)>=1) # is violate deformation constraint

    for i in range(len(generated.model.elements)):
        x_e[i][0]  = generated.model.elements[i].section_no # Section No
        x_e[i][1]  = generated.model.elements[i].area # Area
        x_e[i][2]  = generated.model.elements[i].length # Length
        x_e[i][3]  = abs(generated.model.elements[i].iscompress-1) # is under tension
        x_e[i][4]  = generated.model.elements[i].iscompress # is under compression
        x_e[i][5]  = generated.model.elements[i].e_q[0][0] # Normalized stress (Norm max-min==0)
        x_e[i][6]  = int(generated.model.elements[i].prop_yeield > 1) # is violate stress constraint
        x_e[i][7]  = generated.model.elements[i].nodes[0].coord[0] # p-end: x coord
        x_e[i][8]  = generated.model.elements[i].nodes[0].coord[1] # p-end: y coord
        x_e[i][9]  = generated.model.elements[i].nodes[0].res[0] # p-end: x support
        x_e[i][10] = generated.model.elements[i].nodes[0].res[1] # p-end: y support
        x_e[i][11] = abs(generated.model.elements[i].nodes[0].has_loady) # p-end: y load
        x_e[i][12] = abs(generated.model.elements[i].nodes[0].global_d[1][0]) # p-end: Normalized deformation (Norm max-min==0)
        x_e[i][13] = int((x_e[i][12]/generated.max_deformation)>=1) # p-end: is violate deformation constraint
        x_e[i][14] = generated.model.elements[i].nodes[1].coord[0] # q-end: x coord
        x_e[i][15] = generated.model.elements[i].nodes[1].coord[1] # q-end: y coord
        x_e[i][16] = generated.model.elements[i].nodes[1].res[0] # q-end: x support
        x_e[i][17] = generated.model.elements[i].nodes[1].res[1] # q-end: y support
        x_e[i][18] = abs(generated.model.elements[i].nodes[1].has_loady) # q-end: y load
        x_e[i][19] = abs(generated.model.elements[i].nodes[1].global_d[1][0]) # q-end: Normalized deformation (Norm max-min==0)
        x_e[i][20] = int((x_e[i][19]/generated.max_deformation)>=1) # q-end: is violate deformation constraint

        A_n[generated.model.elements[i].nodes[0].name-1][generated.model.elements[i].nodes[1].name-1] = 1
        A_n[generated.model.elements[i].nodes[1].name-1][generated.model.elements[i].nodes[0].name-1] = 1

        for nod in range(len(generated.model.elements[i].nodes)):
            for conc in range(len(generated.model.elements[i].nodes[nod].adj_ele)):
                A_e[i][generated.model.elements[i].nodes[nod].adj_ele[conc]-1] = 1

        c_e[i][generated.model.elements[i].nodes[0].name-1] = 1
        c_e[i][generated.model.elements[i].nodes[1].name-1] = 1


    #x_n = (x_n-x_n.min(axis=0)) / (x_n.max(axis=0)-x_n.min(axis=0)+1e-6) # Normalize x_n
    #x_e = (x_e-x_e.min(axis=0)) / (x_e.max(axis=0)-x_e.min(axis=0)+1e-6) # Normalize x_e

    A_n = A_n + Dia_n
    D_n = degree_power(A_n,-1/2)            # for GCN CASE
    A_n = np.matmul(D_n,np.matmul(A_n,D_n)) # for GCN CASE


    D_e = degree_power(A_e,-1/2)            # for GCN CASE
    A_e = np.matmul(D_e,np.matmul(A_e,D_e)) # for GCN CASE

    return x_n, A_n, x_e, c_e


'''
======================================================
CLASS PART
======================================================
'''


# Environment class, contain Game class. Do not change / 環境クラス、ゲームクラスを含みます。 変えないで
class ENV:
    def __init__(self,game):
        self.name = 'FRAME_ENV'
        self.game = game
        self.num_agents = game.num_agents
        self.over = 0
        #=======================
        #Output Action
        #=======================
        self.output = [] #output = [St,at,rt,St+1,Done]

    def check_over(self):
        if self.game.done_counter == 1:
            self.over = 1
        else:
            pass

    def reset(self):
        self.over = 0
        self.game.reset()
        self.output = []


#=============================================================================
# GAME MOMARL
class Game_research04:
    def __init__(self,end_step,model,num_agents=2):
        self.name = 'Game_research04'
        self.description = 'There are 2 type of agent \n Agent_s adjust node up and down\n Agent_t adjust element section'
        self.objective = 'min(Weigth),min(Diff_btw_targetShape_and_currentShape)'
        self.num_agents = num_agents
        self.gen_model = model
        self.num_x = model.num_x
        self.num_y = model.num_y

        self.game_step = 1
        self.end_step = end_step # when will the game end

        #=======================
        #State Action'
        # for MADDPG
        #=======================
        self.height_change = []
        self.topology_change = []
        #=======================
        # Game rules
        #=======================

        self.max_y_val = model.y_max
        self.min_y_val = model.y_min

        self.reward_counter = [0,0]
        self.done_counter = 0

        self.current_hv = 0
        self.ref_point = [1,1]
        self.front_max_distance = 0
        self.front_dis_distance = 0

        # Compute objs
        all_v = np.zeros((len(self.gen_model.model.elements)),dtype=np.float32)
        for i in range(len(self.gen_model.model.elements)):
            all_v[i] = self.gen_model.model.elements[i].area * self.gen_model.model.elements[i].length

        all_dt = np.zeros((len(self.gen_model.model.nodes)),dtype=np.float32)
        for i in range(len(self.gen_model.model.nodes)):
            if self.gen_model.model.nodes[i].top_node == 1:
                all_dt[i] = abs(self.gen_model.model.nodes[i].target - self.gen_model.model.nodes[i].coord[1])

        self.int_obj1 = np.sum(all_v) # Truss volume
        self.int_obj2 = np.sum(all_dt) # Difference btw target geometry and current truss geometry

        print('-------------------------------------------------------')
        print(self.description)
        print(self.objective)
        print('GAME WILL BE ENDED AFTER {} STEP'.format(self.end_step))
        print('-------------------------------------------------------')

    def re_game(self,end_step,model,num_agents=2):
        self.name = 'Game_research04'
        self.description = 'There are 2 type of agent \n Agent_s adjust node up and down\n Agent_t adjust element section'
        self.objective = 'min(Weigth),min(Diff_btw_targetShape_and_currentShape)'
        self.num_agents = num_agents
        self.gen_model = model
        self.num_x = model.num_x
        self.num_y = model.num_y

        self.game_step = 1
        self.end_step = end_step # when will the game end

        #=======================
        #State Action'
        # for MADDPG
        #=======================

        self.height_change = []
        self.topology_change = []

        #=======================
        # Game rules
        #=======================

        self.max_y_val = model.y_max
        self.min_y_val = model.y_min

        self.reward_counter = [0,0]
        self.done_counter = 0

        self.current_hv   = 0
        self.ref_point = [1,1]
        self.front_max_distance = 0
        self.front_dis_distance = 0

        # Compute objs
        all_v = np.zeros((len(self.gen_model.model.elements)),dtype=np.float32)
        for i in range(len(self.gen_model.model.elements)):
            all_v[i] = self.gen_model.model.elements[i].area * self.gen_model.model.elements[i].length

        all_dt = np.zeros((len(self.gen_model.model.nodes)),dtype=np.float32)
        for i in range(len(self.gen_model.model.nodes)):
            if self.gen_model.model.nodes[i].top_node == 1:
                all_dt[i] = abs(self.gen_model.model.nodes[i].target - self.gen_model.model.nodes[i].coord[1])

        self.int_obj1 = np.sum(all_v) # Truss volume
        self.int_obj2 = np.sum(all_dt) # Difference btw target geometry and current truss geometry

        print('-------------------------------------------------------')
        print(self.description)
        print(self.objective)
        print('GAME WILL BE ENDED AFTER {} STEP'.format(self.end_step))
        print('-------------------------------------------------------')

    def _game_get_1_state(self, index=0):
        # index in pf
        self.gen_model.set_moveRange()
        self.gen_model.model.restore()
        self.gen_model.model.gen_all()
        x_n, A_n, A_s, A_n_ts, A_n_cs, mask = state_data(self.gen_model)
        nN_x_n, _, nN_x_e, nC_e = state_data_not_norm(self.gen_model)
        # initial pareto graph
        x_pf = np.zeros((1,4),dtype=np.float32)
        x_pf[0][0] = 1
        x_pf[0][1] = 1
        x_pf[0][2] = 1
        x_pf[0][3] = 1/MAX_FRONT
        A_pf = np.eye(1,dtype=np.float32)

        return x_n, A_n, A_s, A_n_ts, A_n_cs, mask, x_pf, A_pf, nN_x_n, nN_x_e, nC_e

    def _game_get_state(self, index=0):
        # index in pf
        x_n, A_n, A_s, A_n_ts, A_n_cs, mask, x_pf, A_pf = self.pf[index][2], self.pf[index][3], self.pf[index][4], self.pf[index][5], self.pf[index][6], self.pf[index][7], self.pf[index][0], self.pf[index][1]
        return x_n, A_n, A_s, A_n_ts, A_n_cs, mask, x_pf, A_pf

    def _set_model(self, set_node, set_element):

        #set model using graph from pareto index
        for i in range(len(self.gen_model.model.nodes)):
            self.gen_model.model.nodes[i].coord[1] = set_node[i][1]

        for i in range(len(self.gen_model.model.elements)):

            self.gen_model.model.elements[i].section_no = int(set_element[i][0])
            self.gen_model.model.elements[i].area       = self.gen_model.truss[int(set_element[i][0])][0]*1e-4
            self.gen_model.model.elements[i].set_i(self.gen_model.truss[int(set_element[i][0])][1]*1e-8)

    def _game_modify(self,set_node, set_element, nC_e, actions):
        # mode = 'S', 'T' ,'ST'
        # actions = [action, None]   for agent S
        # actions = [None, action]   for agent T
        # actions = [action, action] for agent S and agent T

        for i in range(len(actions[0])):
            for j in range(len(actions[0][i])):
                if actions[0][i][j] > 1:
                    actions[0][i][j] = 1
                elif actions[0][i][j] < 0:
                    actions[0][i][j] = 0

        for i in range(len(actions[1])):
            for j in range(len(actions[1][i])):
                if actions[1][i][j] > 1:
                    actions[1][i][j] = 1
                elif actions[1][i][j] < 0:
                    actions[1][i][j] = 0

        #print(actions[0])
        #print('----------------')
        #print(actions[1])
        self._set_model(set_node, set_element)

        factor = 0.25
        threshold_stress = 0.95
        # MODIFY THE GEOMETRY
        for i in range(len(actions[0])):
            adjnum = np.argmax(actions[0][i])
            if adjnum == 0:
                # MOVE UPWARD
                step = min([1,actions[0][i][adjnum]])*self.gen_model.model.nodes[i].max_up*factor
                #print()
                self.gen_model.model.nodes[i].coord[1] += step
            elif adjnum == 1:
                # MOVE DOWNWARD
                step = min([1,actions[0][i][adjnum]])*self.gen_model.model.nodes[i].max_down*factor
                self.gen_model.model.nodes[i].coord[1] -= step

        for i in range(len(self.gen_model.model.nodes)):
            if self.gen_model.model.nodes[i].res[1] == 1:
                self.gen_model.model.nodes[i].coord[1] = 0
            self.gen_model.model.nodes[i].coord[1] = round(self.gen_model.model.nodes[i].coord[1],2)


        modify_element = np.dot(nC_e,actions[1])
        for i in range(len(self.gen_model.model.elements)):
            pred_vals = modify_element[i]
            if np.argmax(pred_vals) == 0:
                self.gen_model.model.elements[i].section_no = max(0,self.gen_model.model.elements[i].section_no-1)
            elif np.argmax(pred_vals) == 1:
                self.gen_model.model.elements[i].section_no = min(len(self.gen_model.truss)-1,self.gen_model.model.elements[i].section_no+1)
            else:
                self.gen_model.model.elements[i].section_no = self.gen_model.model.elements[i].section_no


            self.gen_model.model.elements[i].area       = self.gen_model.truss[self.gen_model.model.elements[i].section_no][0]*1e-4
            self.gen_model.model.elements[i].set_i(self.gen_model.truss[self.gen_model.model.elements[i].section_no][1]*1e-8)
        # FIX NODE HEIGTH
        # IF NODAL HEIGHT IS LESSER THAN min_y_val
        for i in range(len(self.gen_model.model.nodes)):
            if self.gen_model.model.nodes[i].coord[1] < self.min_y_val:
                if self.gen_model.model.nodes[i].top_node == 1:
                    self.gen_model.model.nodes[i].coord[1] = self.gen_model.d_min
                    self.gen_model.model.nodes[i].vertical_pair[0].coord[1] = self.min_y_val
                else:
                    self.gen_model.model.nodes[i].coord[1] = self.min_y_val
        # IF NODAL HEIGHT IS LAERGER THAN y max
        for i in range(len(self.gen_model.model.nodes)):
            if self.gen_model.model.nodes[i].coord[1] > self.gen_model.y_max:
                if self.gen_model.model.nodes[i].top_node == 1:
                    self.gen_model.model.nodes[i].coord[1] = self.gen_model.y_max
                else:
                    self.gen_model.model.nodes[i].coord[1] = self.gen_model.y_max - self.gen_model.d_min
                    self.gen_model.model.nodes[i].vertical_pair[0].coord[1] = self.gen_model.y_max
        # IF depth pair is no good
        for i in range(len(self.gen_model.model.nodes)):
            if abs(self.gen_model.model.nodes[i].coord[1]-self.gen_model.model.nodes[i].vertical_pair[0].coord[1]) < self.gen_model.d_min:
                if self.gen_model.model.nodes[i].top_node == 1:
                    self.gen_model.model.nodes[i].coord[1] = self.gen_model.model.nodes[i].vertical_pair[0].coord[1] + self.gen_model.d_min
                else:
                    pass



        # ASSIGN SYMMETRY NODE
        if random.random()>=0.5:
            self.gen_model.model.nodes[0].coord[1] = self.gen_model.model.nodes[0].coord[1]
            self.gen_model.model.nodes[15].coord[1] = self.gen_model.model.nodes[0].coord[1]

            self.gen_model.model.nodes[1].coord[1] = self.gen_model.model.nodes[1].coord[1]
            self.gen_model.model.nodes[14].coord[1] = self.gen_model.model.nodes[1].coord[1]

            self.gen_model.model.nodes[2].coord[1] = self.gen_model.model.nodes[2].coord[1]
            self.gen_model.model.nodes[13].coord[1] = self.gen_model.model.nodes[2].coord[1]

            self.gen_model.model.nodes[3].coord[1] = self.gen_model.model.nodes[3].coord[1]
            self.gen_model.model.nodes[12].coord[1] = self.gen_model.model.nodes[3].coord[1]

            self.gen_model.model.nodes[4].coord[1] = self.gen_model.model.nodes[4].coord[1]
            self.gen_model.model.nodes[11].coord[1] = self.gen_model.model.nodes[4].coord[1]

            self.gen_model.model.nodes[5].coord[1] = self.gen_model.model.nodes[5].coord[1]
            self.gen_model.model.nodes[10].coord[1] = self.gen_model.model.nodes[5].coord[1]

            self.gen_model.model.nodes[6].coord[1] = self.gen_model.model.nodes[6].coord[1]
            self.gen_model.model.nodes[9].coord[1] = self.gen_model.model.nodes[6].coord[1]

            self.gen_model.model.nodes[7].coord[1] = self.gen_model.model.nodes[7].coord[1]
            self.gen_model.model.nodes[8].coord[1] = self.gen_model.model.nodes[7].coord[1]

            self.gen_model.model.nodes[16].coord[1] = self.gen_model.model.nodes[16].coord[1]
            self.gen_model.model.nodes[31].coord[1] = self.gen_model.model.nodes[16].coord[1]

            self.gen_model.model.nodes[17].coord[1] = self.gen_model.model.nodes[17].coord[1]
            self.gen_model.model.nodes[30].coord[1] = self.gen_model.model.nodes[17].coord[1]

            self.gen_model.model.nodes[18].coord[1] = self.gen_model.model.nodes[18].coord[1]
            self.gen_model.model.nodes[29].coord[1] = self.gen_model.model.nodes[18].coord[1]

            self.gen_model.model.nodes[19].coord[1] = self.gen_model.model.nodes[19].coord[1]
            self.gen_model.model.nodes[28].coord[1] = self.gen_model.model.nodes[19].coord[1]

            self.gen_model.model.nodes[20].coord[1] = self.gen_model.model.nodes[20].coord[1]
            self.gen_model.model.nodes[27].coord[1] = self.gen_model.model.nodes[20].coord[1]

            self.gen_model.model.nodes[21].coord[1] = self.gen_model.model.nodes[21].coord[1]
            self.gen_model.model.nodes[26].coord[1] = self.gen_model.model.nodes[21].coord[1]

            self.gen_model.model.nodes[22].coord[1] = self.gen_model.model.nodes[22].coord[1]
            self.gen_model.model.nodes[25].coord[1] = self.gen_model.model.nodes[22].coord[1]

            self.gen_model.model.nodes[23].coord[1] = self.gen_model.model.nodes[23].coord[1]
            self.gen_model.model.nodes[24].coord[1] = self.gen_model.model.nodes[23].coord[1]

        else:
            self.gen_model.model.nodes[15].coord[1] = self.gen_model.model.nodes[15].coord[1]
            self.gen_model.model.nodes[0].coord[1] = self.gen_model.model.nodes[15].coord[1]

            self.gen_model.model.nodes[14].coord[1] = self.gen_model.model.nodes[14].coord[1]
            self.gen_model.model.nodes[1].coord[1] = self.gen_model.model.nodes[14].coord[1]

            self.gen_model.model.nodes[13].coord[1] = self.gen_model.model.nodes[13].coord[1]
            self.gen_model.model.nodes[2].coord[1] = self.gen_model.model.nodes[13].coord[1]

            self.gen_model.model.nodes[12].coord[1] = self.gen_model.model.nodes[12].coord[1]
            self.gen_model.model.nodes[3].coord[1] = self.gen_model.model.nodes[12].coord[1]

            self.gen_model.model.nodes[11].coord[1] = self.gen_model.model.nodes[11].coord[1]
            self.gen_model.model.nodes[4].coord[1] = self.gen_model.model.nodes[11].coord[1]

            self.gen_model.model.nodes[10].coord[1] = self.gen_model.model.nodes[10].coord[1]
            self.gen_model.model.nodes[5].coord[1] = self.gen_model.model.nodes[10].coord[1]

            self.gen_model.model.nodes[9].coord[1] = self.gen_model.model.nodes[9].coord[1]
            self.gen_model.model.nodes[6].coord[1] = self.gen_model.model.nodes[9].coord[1]

            self.gen_model.model.nodes[8].coord[1] = self.gen_model.model.nodes[8].coord[1]
            self.gen_model.model.nodes[7].coord[1] = self.gen_model.model.nodes[8].coord[1]

            self.gen_model.model.nodes[31].coord[1] = self.gen_model.model.nodes[31].coord[1]
            self.gen_model.model.nodes[16].coord[1] = self.gen_model.model.nodes[31].coord[1]

            self.gen_model.model.nodes[30].coord[1] = self.gen_model.model.nodes[30].coord[1]
            self.gen_model.model.nodes[17].coord[1] = self.gen_model.model.nodes[30].coord[1]

            self.gen_model.model.nodes[29].coord[1] = self.gen_model.model.nodes[29].coord[1]
            self.gen_model.model.nodes[18].coord[1] = self.gen_model.model.nodes[29].coord[1]

            self.gen_model.model.nodes[28].coord[1] = self.gen_model.model.nodes[28].coord[1]
            self.gen_model.model.nodes[19].coord[1] = self.gen_model.model.nodes[28].coord[1]

            self.gen_model.model.nodes[27].coord[1] = self.gen_model.model.nodes[27].coord[1]
            self.gen_model.model.nodes[20].coord[1] = self.gen_model.model.nodes[27].coord[1]

            self.gen_model.model.nodes[26].coord[1] = self.gen_model.model.nodes[26].coord[1]
            self.gen_model.model.nodes[21].coord[1] = self.gen_model.model.nodes[26].coord[1]

            self.gen_model.model.nodes[25].coord[1] = self.gen_model.model.nodes[25].coord[1]
            self.gen_model.model.nodes[22].coord[1] = self.gen_model.model.nodes[25].coord[1]

            self.gen_model.model.nodes[24].coord[1] = self.gen_model.model.nodes[24].coord[1]
            self.gen_model.model.nodes[23].coord[1] = self.gen_model.model.nodes[24].coord[1]


        # ASSIGN SYMMETRY ELEMENT + ADJUSTMENT
        self.gen_model.model.elements[0].section_no = min(self.gen_model.model.elements[0].section_no,self.gen_model.model.elements[14].section_no)
        self.gen_model.model.elements[14].section_no = min(self.gen_model.model.elements[0].section_no,self.gen_model.model.elements[14].section_no)

        self.gen_model.model.elements[1].section_no = min(self.gen_model.model.elements[1].section_no,self.gen_model.model.elements[13].section_no)
        self.gen_model.model.elements[13].section_no = min(self.gen_model.model.elements[1].section_no,self.gen_model.model.elements[13].section_no)

        self.gen_model.model.elements[2].section_no = min(self.gen_model.model.elements[2].section_no,self.gen_model.model.elements[12].section_no)
        self.gen_model.model.elements[12].section_no = min(self.gen_model.model.elements[2].section_no,self.gen_model.model.elements[12].section_no)

        self.gen_model.model.elements[3].section_no = min(self.gen_model.model.elements[3].section_no,self.gen_model.model.elements[11].section_no)
        self.gen_model.model.elements[11].section_no = min(self.gen_model.model.elements[3].section_no,self.gen_model.model.elements[11].section_no)

        self.gen_model.model.elements[4].section_no = min(self.gen_model.model.elements[4].section_no,self.gen_model.model.elements[10].section_no)
        self.gen_model.model.elements[10].section_no = min(self.gen_model.model.elements[4].section_no,self.gen_model.model.elements[10].section_no)

        self.gen_model.model.elements[5].section_no = min(self.gen_model.model.elements[5].section_no,self.gen_model.model.elements[9].section_no)
        self.gen_model.model.elements[9].section_no = min(self.gen_model.model.elements[5].section_no,self.gen_model.model.elements[9].section_no)

        self.gen_model.model.elements[6].section_no = min(self.gen_model.model.elements[6].section_no,self.gen_model.model.elements[8].section_no)
        self.gen_model.model.elements[8].section_no = min(self.gen_model.model.elements[6].section_no,self.gen_model.model.elements[8].section_no)

        self.gen_model.model.elements[15].section_no = min(self.gen_model.model.elements[15].section_no,self.gen_model.model.elements[29].section_no)
        self.gen_model.model.elements[29].section_no = min(self.gen_model.model.elements[15].section_no,self.gen_model.model.elements[29].section_no)

        self.gen_model.model.elements[16].section_no = min(self.gen_model.model.elements[16].section_no,self.gen_model.model.elements[28].section_no)
        self.gen_model.model.elements[28].section_no = min(self.gen_model.model.elements[16].section_no,self.gen_model.model.elements[28].section_no)

        self.gen_model.model.elements[17].section_no = min(self.gen_model.model.elements[17].section_no,self.gen_model.model.elements[27].section_no)
        self.gen_model.model.elements[27].section_no = min(self.gen_model.model.elements[17].section_no,self.gen_model.model.elements[27].section_no)

        self.gen_model.model.elements[18].section_no = min(self.gen_model.model.elements[18].section_no,self.gen_model.model.elements[26].section_no)
        self.gen_model.model.elements[26].section_no = min(self.gen_model.model.elements[18].section_no,self.gen_model.model.elements[26].section_no)

        self.gen_model.model.elements[19].section_no = min(self.gen_model.model.elements[19].section_no,self.gen_model.model.elements[25].section_no)
        self.gen_model.model.elements[25].section_no = min(self.gen_model.model.elements[19].section_no,self.gen_model.model.elements[25].section_no)

        self.gen_model.model.elements[20].section_no = min(self.gen_model.model.elements[20].section_no,self.gen_model.model.elements[24].section_no)
        self.gen_model.model.elements[24].section_no = min(self.gen_model.model.elements[20].section_no,self.gen_model.model.elements[24].section_no)

        self.gen_model.model.elements[21].section_no = min(self.gen_model.model.elements[21].section_no,self.gen_model.model.elements[23].section_no)
        self.gen_model.model.elements[23].section_no = min(self.gen_model.model.elements[21].section_no,self.gen_model.model.elements[23].section_no)

        self.gen_model.model.elements[30].section_no = min(self.gen_model.model.elements[30].section_no,self.gen_model.model.elements[45].section_no)
        self.gen_model.model.elements[45].section_no = min(self.gen_model.model.elements[30].section_no,self.gen_model.model.elements[45].section_no)

        self.gen_model.model.elements[31].section_no = min(self.gen_model.model.elements[31].section_no,self.gen_model.model.elements[44].section_no)
        self.gen_model.model.elements[44].section_no = min(self.gen_model.model.elements[31].section_no,self.gen_model.model.elements[44].section_no)

        self.gen_model.model.elements[32].section_no = min(self.gen_model.model.elements[32].section_no,self.gen_model.model.elements[43].section_no)
        self.gen_model.model.elements[43].section_no = min(self.gen_model.model.elements[32].section_no,self.gen_model.model.elements[43].section_no)

        self.gen_model.model.elements[33].section_no = min(self.gen_model.model.elements[33].section_no,self.gen_model.model.elements[42].section_no)
        self.gen_model.model.elements[42].section_no = min(self.gen_model.model.elements[33].section_no,self.gen_model.model.elements[42].section_no)

        self.gen_model.model.elements[34].section_no = min(self.gen_model.model.elements[34].section_no,self.gen_model.model.elements[41].section_no)
        self.gen_model.model.elements[41].section_no = min(self.gen_model.model.elements[34].section_no,self.gen_model.model.elements[41].section_no)

        self.gen_model.model.elements[35].section_no = min(self.gen_model.model.elements[35].section_no,self.gen_model.model.elements[40].section_no)
        self.gen_model.model.elements[40].section_no = min(self.gen_model.model.elements[35].section_no,self.gen_model.model.elements[40].section_no)

        self.gen_model.model.elements[36].section_no = min(self.gen_model.model.elements[36].section_no,self.gen_model.model.elements[39].section_no)
        self.gen_model.model.elements[39].section_no = min(self.gen_model.model.elements[36].section_no,self.gen_model.model.elements[39].section_no)

        self.gen_model.model.elements[37].section_no = min(self.gen_model.model.elements[37].section_no,self.gen_model.model.elements[38].section_no)
        self.gen_model.model.elements[38].section_no = min(self.gen_model.model.elements[37].section_no,self.gen_model.model.elements[38].section_no)

        self.gen_model.model.elements[46].section_no = min(self.gen_model.model.elements[46].section_no,self.gen_model.model.elements[75].section_no)
        self.gen_model.model.elements[75].section_no = min(self.gen_model.model.elements[46].section_no,self.gen_model.model.elements[75].section_no)

        self.gen_model.model.elements[47].section_no = min(self.gen_model.model.elements[47].section_no,self.gen_model.model.elements[74].section_no)
        self.gen_model.model.elements[74].section_no = min(self.gen_model.model.elements[47].section_no,self.gen_model.model.elements[74].section_no)

        self.gen_model.model.elements[48].section_no = min(self.gen_model.model.elements[48].section_no,self.gen_model.model.elements[73].section_no)
        self.gen_model.model.elements[73].section_no = min(self.gen_model.model.elements[48].section_no,self.gen_model.model.elements[73].section_no)

        self.gen_model.model.elements[49].section_no = min(self.gen_model.model.elements[49].section_no,self.gen_model.model.elements[72].section_no)
        self.gen_model.model.elements[72].section_no = min(self.gen_model.model.elements[49].section_no,self.gen_model.model.elements[72].section_no)

        self.gen_model.model.elements[50].section_no = min(self.gen_model.model.elements[50].section_no,self.gen_model.model.elements[71].section_no)
        self.gen_model.model.elements[71].section_no = min(self.gen_model.model.elements[50].section_no,self.gen_model.model.elements[71].section_no)

        self.gen_model.model.elements[51].section_no = min(self.gen_model.model.elements[51].section_no,self.gen_model.model.elements[70].section_no)
        self.gen_model.model.elements[70].section_no = min(self.gen_model.model.elements[51].section_no,self.gen_model.model.elements[70].section_no)

        self.gen_model.model.elements[52].section_no = min(self.gen_model.model.elements[52].section_no,self.gen_model.model.elements[69].section_no)
        self.gen_model.model.elements[69].section_no = min(self.gen_model.model.elements[52].section_no,self.gen_model.model.elements[69].section_no)

        self.gen_model.model.elements[53].section_no = min(self.gen_model.model.elements[53].section_no,self.gen_model.model.elements[68].section_no)
        self.gen_model.model.elements[68].section_no = min(self.gen_model.model.elements[53].section_no,self.gen_model.model.elements[68].section_no)

        self.gen_model.model.elements[54].section_no = min(self.gen_model.model.elements[54].section_no,self.gen_model.model.elements[67].section_no)
        self.gen_model.model.elements[67].section_no = min(self.gen_model.model.elements[54].section_no,self.gen_model.model.elements[67].section_no)

        self.gen_model.model.elements[55].section_no = min(self.gen_model.model.elements[55].section_no,self.gen_model.model.elements[66].section_no)
        self.gen_model.model.elements[66].section_no = min(self.gen_model.model.elements[55].section_no,self.gen_model.model.elements[66].section_no)

        self.gen_model.model.elements[56].section_no = min(self.gen_model.model.elements[56].section_no,self.gen_model.model.elements[65].section_no)
        self.gen_model.model.elements[65].section_no = min(self.gen_model.model.elements[56].section_no,self.gen_model.model.elements[65].section_no)

        self.gen_model.model.elements[57].section_no = min(self.gen_model.model.elements[57].section_no,self.gen_model.model.elements[64].section_no)
        self.gen_model.model.elements[64].section_no = min(self.gen_model.model.elements[57].section_no,self.gen_model.model.elements[64].section_no)

        self.gen_model.model.elements[58].section_no = min(self.gen_model.model.elements[58].section_no,self.gen_model.model.elements[63].section_no)
        self.gen_model.model.elements[63].section_no = min(self.gen_model.model.elements[58].section_no,self.gen_model.model.elements[63].section_no)

        self.gen_model.model.elements[59].section_no = min(self.gen_model.model.elements[59].section_no,self.gen_model.model.elements[62].section_no)
        self.gen_model.model.elements[62].section_no = min(self.gen_model.model.elements[59].section_no,self.gen_model.model.elements[62].section_no)

        self.gen_model.model.elements[60].section_no = min(self.gen_model.model.elements[60].section_no,self.gen_model.model.elements[61].section_no)
        self.gen_model.model.elements[61].section_no = min(self.gen_model.model.elements[60].section_no,self.gen_model.model.elements[61].section_no)


        for i in range(len(self.gen_model.model.elements)):
            self.gen_model.model.elements[i].area = self.gen_model.truss[self.gen_model.model.elements[i].section_no][0]*1e-4
            self.gen_model.model.elements[i].set_i(self.gen_model.truss[self.gen_model.model.elements[i].section_no][1]*1e-8)


        # Structural analysis
        self.gen_model.set_moveRange()
        self.gen_model.model.restore()
        self.gen_model.model.gen_all()
        # Make graph
        x_n, A_n, A_s, A_n_ts, A_n_cs, mask = state_data(self.gen_model)


        nN_x_n, _, nN_x_e, nC_e = state_data_not_norm(self.gen_model)

        St_S  = [x_n, A_n, A_s, A_n_ts, A_n_cs, mask, None, None, nN_x_n, nN_x_e, nC_e] ########
        # Compute objs and constraints
        all_s = np.zeros((len(self.gen_model.model.elements)),dtype=np.float32)
        all_v = np.zeros((len(self.gen_model.model.elements)),dtype=np.float32)
        for i in range(len(self.gen_model.model.elements)):
            all_s[i] = self.gen_model.model.elements[i].prop_yeield
            all_v[i] = self.gen_model.model.elements[i].area * self.gen_model.model.elements[i].length

        all_d = np.zeros((len(self.gen_model.model.nodes)),dtype=np.float32)
        all_dt = np.zeros((len(self.gen_model.model.nodes)),dtype=np.float32)
        for i in range(len(self.gen_model.model.nodes)):
            if self.gen_model.model.nodes[i].top_node == 1:
                all_dt[i] = abs(self.gen_model.model.nodes[i].target - self.gen_model.model.nodes[i].coord[1])
            else:
                all_d[i]  = self.gen_model.model.nodes[i].global_d[1][0]/self.gen_model.max_deformation

        obj1 = np.sum(all_v) # Truss volume
        obj2 = np.sum(all_dt) # Difference btw target geometry and current truss geometry
        con1 = np.amax(np.abs(all_s)) # element stress constraint
        con2 = np.amax(np.abs(all_d)) # node deformation constraint

        point = [obj1/self.int_obj1, obj2/self.int_obj2, con1, con2]

        return point, St_S

    def reset(self):
        self.height_change = []
        self.topology_change = []

        self.reward_counter = [0,0]
        self.done_counter = 0

        self.current_hv = 0
        self.ref_point = [1,1]
        self.front_max_distance = 0
        self.front_dis_distance = 0

    def step(self):
        self.game_step += 1
