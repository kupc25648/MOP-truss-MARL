from FEM_2Dtruss import Load, Node, Element, Model
from set_seed_global import seedThis


import os
import pandas as pd
from pandas import DataFrame
#import matplotlib as mpl # use for mac bigsur
#mpl.use('tkagg') # use for mac bigsur
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import shutil
import csv
import ast
from os import listdir
from os.path import isfile, join

import random
import numpy as np

random.seed(seedThis)
np.random.seed(seedThis)

#--------------------------------
# function to read section from csv file
def read_section(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = np.array(data)
    section_data = data.astype(float)
    # return numpy array object
    return section_data

#--------------------------------
# Truss generate class / フレーム生成クラス
#--------------------------------

class gen_model:
    def __init__(self,
                num_x
                ,num_y
                ,span_x,span_y
                ,tar_y
                ,dmin
                ,loadx
                ,loady
                ,truss_type='roof',support_case=1,topo_code=None):
        # Truss dimension
        self.num_x = num_x # number of node in x direction (horizontal)
        self.num_y = num_y # number of node in y direction (veritcal)
        self.span_x = span_x # list of span x length (m.)
        self.span_y = span_y # list of span y height (m.)

        self.tar_y = tar_y # list of target y height (m.)

        self.YoungM = 2*1e11 # young modulus for all section (steel)
        self.truss_path   = './section_data/01_brace_rod2.csv'
        self.truss = read_section(self.truss_path) #use predetermine list truss_src

        self.max_truss_A = self.truss[-1][0]*1e-4
        self.max_truss_i = self.truss[-1][1]*1e-8

        # Load / 荷重
        self.loadx = loadx # load in x direction(N)
        self.loady = loady # load in y direction(N)-vertical

        self.truss_type = truss_type
        self.topo_code = topo_code
        self.support_case = support_case

        self.max_poss_brace_vol = 0

        self.max_short_stress = 235*1000000
        self.max_long_stress = 235*1000000/1.5
        self.max_deformation = 0.001*sum(self.span_x)

        self.y_max = span_y[0]#*2.5
        self.y_min = 0
        self.d_min = dmin
        print('------------------------')
        print(self.y_max)
        print(self.y_min)
        print(self.d_min)
        print('------------------------')

        # -------------------------------------------------
        # lists used to store structural data / 構造データを格納するために使用されるリスト
        self.n_u_x = [] # value of x coordinate / x座標の値
        self.n_u_y = [] # value of z coordinate / z座標の値
        self.n_u_coord = [] # value of [x,y,z] coordinate / [x、y、z]座標の値
        self.n_u_name_div =[] #node of the structure / 構造のノード


        # data for state
        self.node_bc =[] #boundary condition
        self.node_px =[] #x-position
        self.node_py =[] #y-position (vertical)

        self.node_topo =[] #connected element
        self.node_axial =[] #axial strain energy

        # type of element
        self.E_type1_name = [] # beam
        self.E_type2_name = [] # column
        # brace
        self.E_type3_name = [] # \ right
        self.E_type4_name = [] # / left

        self.model = None

        self.gennode()
        self.generate()


    def set_moveRange(self):
        # TOP NODE CAN BE MOVED
        # DOWN NODE CANNOT BE MOVED
        for i in range(len(self.model.nodes)):
            if self.model.nodes[i].top_node == 1:
                # TOP NODE
                self.model.nodes[i].max_up   = abs(self.y_max - self.model.nodes[i].coord[1])
                self.model.nodes[i].max_down = abs(self.model.nodes[i].coord[1] - self.model.nodes[i].vertical_pair[0].coord[1] - self.d_min)
            else:
                # DOWN NODE
                if self.truss_type=='bridge':
                    self.model.nodes[i].max_up = 0
                    self.model.nodes[i].max_down = 0
                elif self.truss_type=='roof':
                    self.model.nodes[i].max_up   = abs(self.model.nodes[i].vertical_pair[0].coord[1] - self.model.nodes[i].coord[1] - self.d_min)
                    self.model.nodes[i].max_down = abs(self.model.nodes[i].coord[1] - self.y_min)

    def read_src(self,src):
        #print(self.truss)

        #self.truss[self.E_type4_name[counter_E].section_no][0]*1e-4



        lineList = [[line.rstrip('\n')] for line in open(src)]
        for i in range(len(lineList)):
            val = ast.literal_eval(lineList[i][0].replace(" ", ""))
            if len(val)==2:
                # LOAD
                for j in range(len(self.model.loads)):
                    if self.model.loads[j].name == val[0]:
                        self.model.loads[j].size[0] = val[1][0]
                        self.model.loads[j].size[1] = val[1][1]
            elif len(val)==4:
                # NODE
                for j in range(len(self.model.nodes)):
                    if self.model.nodes[j].name == val[0]:
                        self.model.nodes[j].coord[0] = val[1][0]
                        self.model.nodes[j].coord[1] = val[1][1]
                        self.model.nodes[j].res[0]   = val[2][0]
                        self.model.nodes[j].res[1]   = val[2][1]
            elif len(val)==6:
                # ELEMENT
                for j in range(len(self.model.elements)):
                    if self.model.elements[j].name == val[0]:
                        self.model.elements[j].em      = val[3]
                        self.model.elements[j].area    = val[4]
                        self.model.elements[j].i[0][0] = val[5][0][0]
                        for k in range(len(self.truss)):
                            if self.model.elements[j].area == self.truss[k][0]*1e-4:
                                self.model.elements[j].section_no = k
            else:
                print('ERROR')
                break

    def re_value(self
                ,num_x
                ,num_y
                ,span_x,span_y
                ,tar_y
                ,dmin
                ,loadx
                ,loady
                ,truss_type='roof',support_case=1,topo_code=None):

        self.num_x = num_x # number of node in x direction (horizontal)
        self.num_y = num_y # number of node in y direction (veritcal)
        self.span_x = span_x # list of span x length (m.)
        self.span_y = span_y # list of span y height (m.)

        self.tar_y = tar_y # list of target y height (m.)

        self.YoungM = 2*1e11 # young modulus for all section (steel)
        self.truss_path   = './section_data/01_brace_rod2.csv'
        self.truss = read_section(self.truss_path) #use predetermine list truss_src

        self.max_truss_A = self.truss[-1][0]*1e-4
        self.max_truss_i = self.truss[-1][1]*1e-8

        # Load / 荷重
        self.loadx = loadx # load in x direction(N)
        self.loady = loady # load in y direction(N)-vertical

        self.truss_type = truss_type
        self.topo_code = topo_code
        self.support_case = support_case

        self.max_poss_brace_vol = 0

        self.max_short_stress = 235*1000000
        self.max_long_stress = 235*1000000/1.5
        self.max_deformation = 0.001*sum(self.span_x)

        self.y_max = span_y[0]#*2.5
        self.y_min = 0
        self.d_min = dmin

        self._reset_model()
        self.gennode()
        self.generate()

    def gennode(self):
        self.n_u_x = [sum(self.span_x[:i]) for i in range(self.num_x)]
        self.n_u_y = [sum(self.span_y[:i]) for i in range(self.num_y)]

        self.n_u_coord = [[None,None] for i in range(self.num_x*self.num_y)]
        counter = 0
        for i in range(len(self.n_u_y)):
            for j in range(len(self.n_u_x)):
                self.n_u_coord[counter] = [self.n_u_x[j],self.n_u_y[i]]
                counter += 1

    # Function to export structure data into .txt data / 構造データを.txtデータにエクスポートする関数
    def savetxt(self,name):
        # ------------------------------
        # Write and save output model  / 出力モデルファイルの書き込みと保存
        # ------------------------------
        new_file = open(name, "w+")
        for num1 in range(len(self.model.loads)):
            new_file.write(" {}\r\n".format(self.model.loads[num1]))
        for num1 in range(len(self.model.nodes)):
            new_file.write(" {}\r\n".format(self.model.nodes[num1]))
        for num1 in range(len(self.model.elements)):
            new_file.write(" {},{},{},{},{},{}\r\n".format(
                self.model.elements[num1].name,
                self.model.elements[num1].nodes[0].name,
                self.model.elements[num1].nodes[1].name,
                self.model.elements[num1].em,
                self.model.elements[num1].area,
                self.model.elements[num1].i
                ))
        new_file.close()

    def _reset_model(self):
        # reset self.model and related properties
        self.n_u_x = [] # value of x coordinate / x座標の値
        self.n_u_y = [] # value of z coordinate / z座標の値
        self.n_u_coord = [] # value of [x,y,z] coordinate / [x、y、z]座標の値
        self.n_u_name_div =[] #node of the structure / 構造のノード


        # data for state
        self.node_bc =[] #boundary condition
        self.node_px =[] #x-position
        self.node_py =[] #y-position (vertical)

        self.node_topo =[] #connected element
        self.node_axial =[] #axial strain energy

        # type of element
        self.E_type1_name = [] # beam
        self.E_type2_name = [] # column
        # brace
        self.E_type3_name = [] # \ right
        self.E_type4_name = [] # / left

        self.model = None



    # Function to generate truss structure
    def generate(self):

        '''
        ----------------------------------
        Generate Nodes / ノードを生成
        ----------------------------------
        '''
        n = 'n'
        n_u_name=[None for i in range(len(self.n_u_coord))]
        counter = 1
        for i in range(len(self.n_u_coord)):
            n_u_name[i] = n+str(counter)
            n_u_name[i] = Node()
            n_u_name[i].set_name(counter)
            n_u_name[i].set_coord(self.n_u_coord[i][0],self.n_u_coord[i][1])
            counter+=1

        self.n_u_name_div = [ [None for i in range(len(self.n_u_x))] for j in range(len(self.n_u_y))]
        floor = 0
        span = 0
        name_counter = 1
        for i in range(len(n_u_name)):
            n_u_name[i].set_name(name_counter)
            if (i != 0):
                if n_u_name[i].coord[1] > n_u_name[i-1].coord[1]:
                    floor += 1
                    span = 0
            self.n_u_name_div[floor][span] = n_u_name[i]
            span += 1
            name_counter += 1

        # ------------------------------------------------------------------------------------
        '''
        ----------------------------------
        Generate elements / 要素を生成する
        ----------------------------------
        '''

        e = 'e'
        self.E_type1_name =[ None for i in range(len(self.n_u_name_div) * (len(self.n_u_name_div[0])-1)) ] # beam
        counter = 1
        counter_E = 0
        for num in range(len(self.n_u_name_div)):
            for i in range((len(self.n_u_name_div[0])-1)):
                self.E_type1_name[counter_E] = e+str(counter)
                self.E_type1_name[counter_E] = Element()
                self.E_type1_name[counter_E].set_name(counter)
                self.E_type1_name[counter_E].set_nodes(self.n_u_name_div[num][i],self.n_u_name_div[num][i+1])
                self.E_type1_name[counter_E].section_no = len(self.truss)-1
                self.E_type1_name[counter_E].set_em(self.YoungM)
                self.E_type1_name[counter_E].set_area(self.truss[self.E_type1_name[counter_E].section_no][0]*1e-4)
                self.E_type1_name[counter_E].set_i(self.truss[self.E_type1_name[counter_E].section_no][1]*1e-8)

                counter+=1
                counter_E += 1



        self.E_type2_name =[ None for i in range((len(self.n_u_name_div)-1) * (len(self.n_u_name_div[0]))) ] # column
        counter_E = 0
        for num in range(len(self.n_u_name_div)-1):
            for i in range(len(self.n_u_name_div[0])):
                self.E_type2_name[counter_E] = e+str(counter)
                self.E_type2_name[counter_E] = Element()
                self.E_type2_name[counter_E].set_name(counter)
                self.E_type2_name[counter_E].set_nodes(self.n_u_name_div[num][i],self.n_u_name_div[num+1][i])
                if self.n_u_name_div[num][i].coord[1] > self.n_u_name_div[num+1][i].coord[1]:
                    self.n_u_name_div[num][i].top_node = 1
                else:
                    self.n_u_name_div[num+1][i].top_node = 1

                self.n_u_name_div[num][i].vertical_pair.append(self.n_u_name_div[num+1][i])
                self.n_u_name_div[num+1][i].vertical_pair.append(self.n_u_name_div[num][i])

                self.E_type2_name[counter_E].section_no = len(self.truss)-1
                self.E_type2_name[counter_E].set_em(self.YoungM)
                self.E_type2_name[counter_E].set_area(self.truss[self.E_type2_name[counter_E].section_no][0]*1e-4)
                self.E_type2_name[counter_E].set_i(self.truss[self.E_type2_name[counter_E].section_no][1]*1e-8)

                counter+=1
                counter_E += 1

        self.E_type3_name =[ None for i in range((len(self.n_u_name_div)-1) * (len(self.n_u_name_div[0])-1))  ] # \ right
        counter_E = 0
        for num in range(len(self.n_u_name_div)-1):
            for i in range(len(self.n_u_name_div[0])-1):
                self.E_type3_name[counter_E] = e+str(counter)
                self.E_type3_name[counter_E] = Element()
                self.E_type3_name[counter_E].set_name(counter)
                self.E_type3_name[counter_E].set_nodes(self.n_u_name_div[num+1][i],self.n_u_name_div[num][i+1])
                self.E_type3_name[counter_E].section_no = len(self.truss)-1
                self.E_type3_name[counter_E].set_em(self.YoungM)
                self.E_type3_name[counter_E].set_area(self.truss[self.E_type3_name[counter_E].section_no][0]*1e-4)
                self.E_type3_name[counter_E].set_i(self.truss[self.E_type3_name[counter_E].section_no][1]*1e-8)

                counter+=1
                counter_E += 1

        self.E_type4_name =[ None for i in range((len(self.n_u_name_div)-1) * (len(self.n_u_name_div[0])-1))  ] # / left
        counter_E = 0
        for num in range(len(self.n_u_name_div)-1):
            for i in range(len(self.n_u_name_div[0])-1):
                self.E_type4_name[counter_E] = e+str(counter)
                self.E_type4_name[counter_E] = Element()
                self.E_type4_name[counter_E].set_name(counter)
                self.E_type4_name[counter_E].set_nodes(self.n_u_name_div[num][i],self.n_u_name_div[num+1][i+1])
                self.E_type4_name[counter_E].section_no = len(self.truss)-1
                self.E_type4_name[counter_E].set_em(self.YoungM)
                self.E_type4_name[counter_E].set_area(self.truss[self.E_type4_name[counter_E].section_no][0]*1e-4)
                self.E_type4_name[counter_E].set_i(self.truss[self.E_type4_name[counter_E].section_no][1]*1e-8)

                counter+=1
                counter_E += 1

        self.model = Model()
        '''
        ----------------------------------
        Generate Loads / Loadを生成する
        ----------------------------------
        '''
        # load downward
        l1 = Load()
        l1.set_name(1)
        l1.set_size(0,self.loady)

        self.model.add_load(l1)

        # Add Node
        count_tar = 0
        for i in range(len(n_u_name)):
            # if top not set target
            if n_u_name[i].top_node == 1:
                n_u_name[i].target = self.tar_y[count_tar]
                count_tar += 1
            '''
            if n_u_name[i].top_node == 1:
                n_u_name[i].set_load(l1)
                n_u_name[i].has_loady = 1
                n_u_name[i].target = self.tar_y[count_tar]
                count_tar += 1
                #n_u_name[i].coord[1] += 1.5*np.random.random()*n_u_name[i].coord[1]
            '''
            self.model.add_node(n_u_name[i])

        # Beam
        for i in range(len(self.E_type1_name)):
            self.model.add_element(self.E_type1_name[i])
        # Column
        for i in range(len(self.E_type2_name)):
            self.model.add_element(self.E_type2_name[i])

       # Brace \
        for i in range(len(self.E_type3_name)):
            self.model.add_element(self.E_type3_name[i])

        # Brace /
        for i in range(len(self.E_type4_name)):
            self.model.add_element(self.E_type4_name[i])

        # set supports
        x_coord = [self.model.nodes[i].coord[0] for i in range(len(self.model.nodes))]
        y_coord = [self.model.nodes[i].coord[1] for i in range(len(self.model.nodes))]

        x_coord = list(set(x_coord))
        if self.support_case == 1:
            pass
        elif self.support_case == 2:
            x_coord.remove(max(x_coord))
        elif self.support_case == 3:
            x_coord.remove(min(x_coord))
        elif self.support_case == 4:
            x_coord.remove(min(x_coord))
            x_coord.remove(max(x_coord))

        for i in range(len(self.model.nodes)):
            if self.model.nodes[i].coord[1] == min(y_coord):
                if (self.model.nodes[i].coord[0] == max(x_coord)) or (self.model.nodes[i].coord[0] == min(x_coord)):
                    self.model.nodes[i].set_res(1,1)

        # SET LOADS
        if self.truss_type=='bridge':
            for i in range(len(self.model.nodes)):
                if (self.model.nodes[i].coord[1] == min(y_coord)) and (self.model.nodes[i].res[1] == 0):
                    self.model.nodes[i].set_load(l1)
                    self.model.nodes[i].has_loady = 1
        elif self.truss_type=='roof':
            for i in range(len(self.model.nodes)):
                if self.model.nodes[i].top_node == 1:
                    self.model.nodes[i].set_load(l1)
                    self.model.nodes[i].has_loady = 1


        self.model.restore()
        self.model.gen_all()
    def render_load(self,load='Y',factor=0.01,name='./name.png',target=None):
        plt.rcParams['figure.figsize'] = (20, 5)
        #ORIGIN and target
        Xcoord = []
        Ycoord = []
        for i in range(len(self.model.nodes)):
            Xcoord.append(self.model.nodes[i].coord[0])
            Ycoord.append(self.model.nodes[i].coord[1])

        Nodes = {'X':Xcoord, 'Y':Ycoord}
        nodeplot = pd.DataFrame(Nodes, columns = ['X', 'Y'])
        n = nodeplot.plot(x='X',y='Y',kind='scatter',color ='black',grid=True,legend=False)

        for i in range(len(self.model.elements)):

            xstart = self.model.elements[i].nodes[0].coord[0]
            xend = self.model.elements[i].nodes[1].coord[0]
            ystart = self.model.elements[i].nodes[0].coord[1]
            yend = self.model.elements[i].nodes[1].coord[1]
            Xcoord = [xstart,xend]
            Ycoord = [ystart,yend]
            Elements = {'X':Xcoord, 'Y':Ycoord}
            elementplot = pd.DataFrame(Elements, columns = ['X', 'Y'])

            thick = self.model.elements[i].area / self.max_truss_A
            #print(thick)

            elementplot.plot(x='X',y='Y',kind='line',color ='red',alpha=0.5,grid=True, ax=n,legend=False,  linewidth= thick*10)

        #TARGET
        if target != None:
            Tar_Xcoord = []
            Tar_Ycoord = []
            for i in range(len(target)):
                Tar_Xcoord.append(target[i][0])
                Tar_Ycoord.append(target[i][1])
            Tar_Nodes = {'X':Tar_Xcoord, 'Y':Tar_Ycoord}
            Tar_nodeplot = pd.DataFrame(Tar_Nodes, columns = ['X', 'Y'])
            Tar_nodeplot.plot(x='X',y='Y',kind='scatter',color ='magenta',grid=True, ax=n,legend=False)


        #SUPPORT
        Xcoord_Ysupport =[]
        Ycoord_Ysupport =[]
        Xcoord_XYsupport =[]
        Ycoord_XYsupport =[]
        for i in range(len(self.model.nodes)):
            if (self.model.nodes[i].res[0] == 0) and (self.model.nodes[i].res[1] == 1):
                Xcoord_Ysupport.append(self.model.nodes[i].coord[0])
                Ycoord_Ysupport.append(self.model.nodes[i].coord[1])
            if (self.model.nodes[i].res[0] == 1) and (self.model.nodes[i].res[1] == 1):
                Xcoord_XYsupport.append(self.model.nodes[i].coord[0])
                Ycoord_XYsupport.append(self.model.nodes[i].coord[1])

        if Xcoord_Ysupport != 0:
            Ysupport = {'X':Xcoord_Ysupport, 'Y':Ycoord_Ysupport}
            Ysupportplot = pd.DataFrame(Ysupport, columns = ['X', 'Y'])
            Ysupportplot.plot(x='X',y='Y',kind='scatter',color ='grey',marker="^",s=100,grid=True, ax=n,legend=False)
        if Xcoord_XYsupport != 0:
            XYsupport = {'X':Xcoord_XYsupport, 'Y':Ycoord_XYsupport}
            XYsupportplot = pd.DataFrame(XYsupport, columns = ['X', 'Y'])
            XYsupportplot.plot(x='X',y='Y',kind='scatter',color ='black',marker="^",s=100,grid=True, ax=n,legend=False)
        '''
        # LOAD
        if load == 'Y':
            Xcoord_Yload = []
            Ycoord_Yload = []
            Ysize = []
            for i in range(len(self.model.nodes)):
                if len(self.model.nodes[i].loads) != 0:
                    for l in range(len(self.model.nodes[i].loads)):
                        if self.model.nodes[i].loads[l][0].size[1] != 0:
                            Xcoord_Yload.append(self.model.nodes[i].coord[0])
                            Ycoord_Yload.append(self.model.nodes[i].coord[1])
                            Ysize.append(int(abs(self.model.nodes[i].loads[l][0].size[1])*factor))

            if len(Xcoord_Yload) != 0:
                Yload = {'X':Xcoord_Yload, 'Y':Ycoord_Yload}
                Yloadplot = pd.DataFrame(Yload, columns = ['X', 'Y'])
                Yloadplot.plot(x='X',y='Y',kind='scatter',color ='blue',s=Ysize, alpha=0.5,grid=True, ax=n,legend=False)

        elif load == 'X':
            Xcoord_Xload = []
            Ycoord_Xload = []
            Xsize = []
            for i in range(len(self.model.nodes)):
                if len(self.model.nodes[i].loads) != 0:
                    for l in range(len(self.model.nodes[i].loads)):
                        if self.model.nodes[i].loads[l][0].size[0] != 0:
                            Xcoord_Xload.append(self.model.nodes[i].coord[0])
                            Ycoord_Xload.append(self.model.nodes[i].coord[1])
                            #print(self.model.nodes[i].loads[l][0].size[0][0])
                            Xsize.append(int(abs(self.model.nodes[i].loads[l][0].size[0]*factor)))
            #print(Xsize)
            if len(Xcoord_Xload) != 0:
                Xload = {'X':Xcoord_Xload, 'Y':Ycoord_Xload}
                Xloadplot = pd.DataFrame(Xload, columns = ['X', 'Y'])
                Xloadplot.plot(x='X',y='Y',kind='scatter',color ='red',s=Xsize, alpha=0.5,grid=True, ax=n,legend=False)
        '''
        '''
        #DEFROM
        NewXcoord = []
        NewYcoord = []
        for i in range(len(self.model.nodes)):
            NewXcoord.append(self.model.nodes[i].coord[0])
            NewYcoord.append(self.model.nodes[i].coord[1])
        for i in range(len(self.model.d)):
            for j in range(len(self.model.tnsc)):
                if self.model.tnsc[j][0] == i + 1:
                    NewXcoord[j] += self.model.d[i][0]
                if self.model.tnsc[j][1] == i + 1:
                    NewYcoord[j] += self.model.d[i][0]
        NewNodes = {'X':NewXcoord, 'Y':NewYcoord}
        Newnodeplot = pd.DataFrame(NewNodes, columns = ['X', 'Y'])
        Newn = Newnodeplot.plot(x='X',y='Y',kind='scatter',color ='red',alpha=0.7,grid=True, ax=n,legend=False)

        for i in range(len(self.model.elements)):
            Newxstart = self.model.elements[i].nodes[0].coord[0]
            Newxend = self.model.elements[i].nodes[1].coord[0]
            Newystart = self.model.elements[i].nodes[0].coord[1]
            Newyend = self.model.elements[i].nodes[1].coord[1]

            for k in range(len(self.model.d)):
                if self.model.tnsc[self.model.elements[i].nodes[0].name-1][0] == k + 1:
                    Newxstart += self.model.d[k][0]
                if self.model.tnsc[self.model.elements[i].nodes[1].name-1][0] == k + 1:
                    Newxend += self.model.d[k][0]
                if self.model.tnsc[self.model.elements[i].nodes[0].name-1][1] == k + 1:
                    Newystart += self.model.d[k][0]
                if self.model.tnsc[self.model.elements[i].nodes[1].name-1][1] == k + 1:
                    Newyend += self.model.d[k][0]
                else:
                    pass
            NewXcoord = [Newxstart,Newxend]
            NewYcoord = [Newystart,Newyend]
            NewElements = {'X':NewXcoord, 'Y':NewYcoord}
            Newelementplot = pd.DataFrame(NewElements, columns = ['X', 'Y'])
            Newelementplot.plot(x='X',y='Y',kind='line',color ='red',alpha=0.7,grid=True, ax=n,legend=False)
        '''
        plt.axis('off')
        plt.axis('equal')
        #plt.show()
        plt.savefig(name)
        plt.close('all')


'''
span_x = [5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0]

span_y = [14]
tar_y  = [7.0,5.0,4.0,3.5,3.0,2.5,2,2.5,3.0,3.5,4.0,5.0,7.0]
truss_src = None
loadx = 0
loady = -10000
game = 1
num_sol = 10
'''

gen_load_x           = 0
gen_load_y           = -7500
gen_topo_code        = None
testChoice = [[[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
                [6.0],
                [3.00, 2.75, 2.50, 2.25, 2.25, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.25, 2.25, 2.50, 2.75, 3.00],
                6.0,0]]
truss_type = 'bridge'
support_case = 1
num_x = len(testChoice[0][0])+1
num_y = len(testChoice[0][1])+1
dmin = 0.3#sum(testChoice[0][0])*0.01

#tar_y = [ max([tar_y[i]*((1+random.random())/2),dmin]) for i in range(len(tar_y)) ] # add some randomness + guarntee dmin

#tar_xy = [11.77,9.31,9.92,11.31,11.57,11.95,9.31,7.51,7.51]

sum_x = [ sum(testChoice[0][0][:i]) for i in range(len(testChoice[0][0])) ]
#print(sum_x)

tar_xy = [ [sum_x[i],testChoice[0][2][i]] for i in range(len(testChoice[0][0]))]
tar_xy.append([sum(testChoice[0][0]), testChoice[0][2][-1]])

test = gen_model(num_x,num_y,testChoice[0][0],testChoice[0][1],testChoice[0][2],dmin,gen_load_x,gen_load_y,truss_type,support_case,gen_topo_code)


num_sol = 100
for g in range(1,11):
    game = g

    for i in range(num_sol):
        try:
            str_src = './MADDPG_Model_data_txt_Game{}/Game{}_Step501_Sol_{}.txt'.format(game,game,i)
            test.read_src(str_src)
            save_name = './MADDPG_Model_data_txt_Game{}/Sol{}.png'.format(game,i)
            target = tar_xy


            test.render_load(load='Y',factor=0.01,name=save_name,target=target)
            all_v = np.zeros((len(test.model.elements)),dtype=np.float32)
            for num in range(len(test.model.elements)):
                all_v[num] = test.model.elements[num].area * test.model.elements[num].length

            all_dt = np.zeros((len(test.model.nodes)),dtype=np.float32)
            for num in range(len(test.model.nodes)):
                if test.model.nodes[num].top_node == 1:
                    all_dt[num] = abs(test.model.nodes[num].target - test.model.nodes[num].coord[1])

            obj1 = np.sum(all_v) # Truss volume
            obj2 = np.sum(all_dt) # Difference btw target geometry and current truss geometry

            print('NAME {} V {} dt {}'.format(save_name,round(obj1/int_obj1,3),round(obj2/int_obj2,3)))
        except:
            pass
    print(game)
