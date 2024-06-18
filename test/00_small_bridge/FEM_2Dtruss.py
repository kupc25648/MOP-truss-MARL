import numpy as np

'''
Planar Truss is subjected to  Axial Load

1. Create Loads
2. Create Nodes
3. Set Loads for Nodes
4. Create Elements from Nodes
5. Create Model
'''
class Load():
    def __init__(self):
        self.name = 1
        self.size = [0, 0]  # size[0] = size in x axis, size[1] = size in y axis

    def set_name(self, name):
        self.name = name

    def set_size(self, x, y):
        self.size[0] = x
        self.size[1] = y

    def __repr__(self):
        return "{0}, {1}".format(self.name, self.size)

class Node:
    def __init__(self):
        self.name = 1
        self.coord = [0, 0]  # coord[0]=xcoord,coord[1]=ycoord
        self.res = [0, 0]  # res[0]=x-restrain,res[1]=y-restrain
        self.loads = [] #[Load]s in this node
        self.global_d = [] #Deformation

        self.adj_ele = []
        self.connected = 0

        self.top_node = 0

        self.vertical_pair = []

        self.int_y    = 0
        self.max_up   = 0 # max value to move this node upward
        self.max_down = 0 # max value to move this node downward

        self.target = 0 # difference fro target coordinate (for top_node only)

        self.has_loady = 0

        self.adj_ele = []

    def set_target(self):
        if self.top_node == 0:
            self.target = self.coord[1]
        else:
            pass


    def set_name(self, name):
        self.name = name

    def set_coord(self, xval, yval):
        self.coord[0] = xval
        self.coord[1] = yval
        self.int_y = yval

    def set_res(self, xres, yres):
        self.res[0] = xres
        self.res[1] = yres

    def set_load(self,load):
        self.loads.append([load])
        self.has_loady = load.size[1]

    def __repr__(self):
        return "{0}, {1}, {2}, {3}".format(self.name, self.coord, self.res, self.loads)


class Element(Node):
    def __init__(self):
        self.name = 1
        self.nodes = []  # nodes[0]=start node,nodes[1]=end node
        self.em = 0  # elastic modulus
        self.area = 0  # Sectional area
        self.dia = 0
        self.length = None
        self.e_q = []
        self.i = [[0]] # moment of inertia on x

        self.section_no = 0

        self.has_changed  = 0
        self.yield_stress = 235*1e6 # long term N/sqm (235N/sqmm)
        self.long_stress  = self.yield_stress / 1.5  # allowable
        self.iscompress   = None

        self.prop_yeield = 0

    def gen_length(self): # Lenght
        self.length = (
            ((self.nodes[1].coord[0]-self.nodes[0].coord[0])**2)+
            ((self.nodes[1].coord[1]-self.nodes[0].coord[1])**2)
            )**0.5

        return self.length

    def set_name(self, name):
        self.name = name

    def set_nodes(self, startnode, endnode):
        self.nodes.append(startnode)
        self.nodes.append(endnode)

        startnode.adj_ele.append(self.name)
        endnode.adj_ele.append(self.name)

        startnode.connected += 1
        endnode.connected += 1

    def set_em(self, emval):
        self.em = emval

    def set_area(self, area):
        self.area = area

    def set_i(self, xval):
        self.i[0][0] = xval

    def __repr__(self):
        return "{0}, {1}, {2}".format(self.nodes, self.em, self.area)



class Model():
    def __init__(self):
        self.nodes = []
        self.elements = []
        self.loads = []

        self.jp = []
        self.pj = []

        self.nsc   = []
        self.tnsc  = []
        self.ttnsc = []
        self.ndof  = 0
        self.jlv   = []

        self.local_k   = []
        self.global_k  = []
        self.T_matrix  = []
        self.Tt_matrix = []


        self.ssm = []
        self.d   = []
        self.v   = []
        self.u   = []
        self.q   = []
        self.f   =[]
        self.r   =[]
        self.U_full = 0

    def restore(self):

        self.jp = []
        self.pj = []

        self.nsc   = []
        self.tnsc  = []
        self.ttnsc = []
        self.ndof  = 0
        self.jlv   = []
        self.local_k   = []
        self.global_k  = []
        self.T_matrix  = []
        self.Tt_matrix = []

        self.ssm = []
        self.d   = []
        self.v   = []
        self.u   = []
        self.q   = []
        self.f   =[]
        self.r   =[]
        self.U_full = 0

    # add load to model
    def add_load(self, load):
        self.loads.append(load)

    # add a node to model
    def add_node(self, node):
        self.nodes.append(node)

    # add an element to model
    def add_element(self, element):
        self.elements.append(element)

    # remove all node, element and load from model
    def reset(self):
        self.nodes = []
        self.elements = []
        #self.loads = []

    # generate support joint matrix-will be called my gen_all method
    def gen_jp(self):
        for i in range(len(self.nodes)):
            if len(self.nodes[i].loads) != 0:
                self.jp.append(self.nodes[i].name)


    # generate force joint matrix-will be called my gen_all method
    def gen_pj(self):
        for i in range(len(self.nodes)):
            sumX = 0
            sumY = 0
            # If Node has Load
            if (len(self.nodes[i].loads) != 0):
                for j in range(len(self.nodes[i].loads)):
                    sumX += self.nodes[i].loads[j][0].size[0]
                    sumY += self.nodes[i].loads[j][0].size[1]
                self.pj.append([sumX,sumY])


    # generate structure coordinate number vector
    def gen_nsc(self):
        self.nsc=[None for i in range(len(self.nodes)*2)]
        count = 0
        for i in range(len(self.nodes)):
            for j in range(2):
                if self.nodes[i].res[j] == 0:
                    self.nsc[count] = 'R'
                elif self.nodes[i].res[j] == 1:
                    self.nsc[count] = 'UR'
                count += 1
        coord_num = 1
        for i in range(len(self.nsc)):
            if self.nsc[i] == 'R':
                self.nsc[i] = coord_num
                coord_num+=1
        for i in range(len(self.nsc)):
            if self.nsc[i] == 'UR':
                self.nsc[i] = coord_num
                coord_num+=1

    # transform nsc so that element in tnsc = joint
    def gen_tnsc(self):
        i = 0
        while i < len(self.nsc):
            self.tnsc.append([self.nsc[i], self.nsc[i + 1]])
            i += 2

    # generate numbers of DOF
    def gen_ndof(self):
        nr = 0
        for i in range(len(self.nodes)):
            for j in range(2):
                if self.nodes[i].res[j] == 1:
                    nr += 1
        self.ndof = 2 * (len(self.nodes)) - nr

    # generate joint loading vector matrix
    def gen_jlv(self):
        x = [[0,0] for i in range(len(self.tnsc))]
        for i in range(len(self.jp)):
            x[self.jp[i]-1] = self.pj[i]
        '''
        for i in range(len(self.tnsc)):
            for j in range(2):
                print('self.tnsc[i][j] {}'.format(self.tnsc[i][j]))
        print('-------------')
        '''
        for i in range(len(self.tnsc)):
            for j in range(2):
                #print('self.tnsc[i][j] {}'.format(self.tnsc[i][j]))
                #print('self.ndof       {}'.format(self.ndof))

                if self.tnsc[i][j] <= self.ndof:
                    self.jlv.append([x[i][j]])
        #print('-------------')

    #generate global member stiffness matrix
    def gen_global_k(self):
        self.local_k = [None for i in range(len(self.elements))]
        self.global_k = [None for i in range(len(self.elements))]
        self.T_matrix = [None for i in range(len(self.elements))]
        self.Tt_matrix = [None for i in range(len(self.elements))]

        for i in range(len(self.elements)):
            #Value
            E = self.elements[i].em
            A = self.elements[i].area
            XEndMinStart = self.elements[i].nodes[1].coord[0]-self.elements[i].nodes[0].coord[0]
            YEndMinStart = self.elements[i].nodes[1].coord[1]-self.elements[i].nodes[0].coord[1]
            L = ((XEndMinStart**2)+(YEndMinStart**2))**0.5
            cos = XEndMinStart/L
            sin = YEndMinStart/L
            EApL = E*A/L
            #local K
            self.local_k[i] = np.array([[EApL,0,-EApL,0],[0,0,0,0],[-EApL,0,EApL,0],[0,0,0,0]])
            #TransformationMatrix
            self.T_matrix[i] = np.array([[cos,sin,0,0],[-sin,cos,0,0],[0,0,cos,sin],[0,0,-sin,cos]])
            #TransformationMatrix.transpose
            self.Tt_matrix[i] = np.array(self.T_matrix[i]).transpose()
            #change Local K to Global K
            self.global_k[i] = (self.Tt_matrix[i].dot(self.local_k[i])).dot(self.T_matrix[i])

    #generate structure stiffness matrix
    def gen_ssm(self):
        self.ttnsc = [None for i in range(len(self.elements))]
        for i in range(len(self.elements)):
            RowCol1 = self.tnsc[self.elements[i].nodes[0].name-1][0]
            RowCol2 = self.tnsc[self.elements[i].nodes[0].name-1][1]
            RowCol3 = self.tnsc[self.elements[i].nodes[1].name-1][0]
            RowCol4 = self.tnsc[self.elements[i].nodes[1].name-1][1]
            self.ttnsc[i] = [RowCol1,RowCol2,RowCol3,RowCol4]

        self.ssm = np.zeros((self.ndof,self.ndof))
        for i in range(len(self.elements)):
            for j in range(4):
                for k in range(4):
                    if (self.ttnsc[i][j] <= self.ndof) and (self.ttnsc[i][k] <= self.ndof):
                        self.ssm[self.ttnsc[i][j]-1][self.ttnsc[i][k]-1] += self.global_k[i][j][k]

    #Joint Displacement Matrix
    def gen_d(self):
        P = np.array(self.jlv)
        S = np.array(self.ssm)
        '''
        # Using LU Decomposition here
        self.ssm_L, self.ssm_U = decom_lu(S,True)
        self.inv_ssm_L = np.linalg.inv(self.ssm_L)
        self.inv_ssm_U = np.linalg.inv(self.ssm_U)
        self.d = np.dot(self.inv_ssm_U,np.dot(self.inv_ssm_L,P))
        '''
        self.d = np.linalg.solve(S, P)


    #member end displacements in the global coordinate system
    def gen_v(self):
        self.v = [None for i in range(len(self.elements))]
        for i in range(len(self.ttnsc)):
            zerov = [[0],[0],[0],[0]]
            for j in range(4):
                if self.ttnsc[i][j]<=self.ndof:
                    zerov[j][0] += float(self.d[self.ttnsc[i][j]-1])
                else:
                    pass
            self.elements[i].nodes[0].global_d = [zerov[0],zerov[1]]
            self.elements[i].nodes[1].global_d = [zerov[2],zerov[3]]
            self.v[i] = zerov


    # member end displacements in the local coordinate
    def gen_u(self):
        self.u = [self.T_matrix[i].dot(self.v[i]) for i in range(len(self.elements))]
        '''
        for i in range(len(self.elements)):
            #Value
            XEndMinStart = self.elements[i].nodes[1].coord[0]-self.elements[i].nodes[0].coord[0]
            YEndMinStart = self.elements[i].nodes[1].coord[1]-self.elements[i].nodes[0].coord[1]
            L = ((XEndMinStart**2)+(YEndMinStart**2))**0.5
            cos = XEndMinStart/L
            sin = YEndMinStart/L
            #TransformationMatrix
            T = np.array([[cos,sin,0,0],[-sin,cos,0,0],[0,0,cos,sin],[0,0,-sin,cos]])
            #change member end displacements in the global coordinate(v) to member end displacements in the local coordinate(u)
            u = T.dot(self.v[i])
            u = u.tolist()
            #Append to self.global_k
            self.u.append(u)
        '''
    def gen_U_full(self):
        #arrayd = np.array(self.d)
        #arrayssm = np.array(self.ssm)
        #energy  = np.dot((np.dot(arrayd.transpose(),self.ssm)),arrayd) * 0.5
        #self.U_full = energy[0]
        self.U_full = np.dot((np.dot(self.d.transpose(),self.ssm)),self.d)[0] * 0.5


    # member end forces in the local coordinate system
    def gen_q(self):
        self.q = [self.local_k[i].dot(self.u[i]) for i in range(len(self.elements))]
        for i in range(len(self.q)):
            self.elements[i].e_q = self.q[i]

    #member global end forces
    def gen_f(self):
        self.f = [self.Tt_matrix[i].dot(self.q[i]) for i in range(len(self.elements))]

    #reaction
    def gen_r(self):
        nall = 2 * (len(self.nodes))
        x = np.zeros((1,nall))
        for i in range(len(self.elements)):
            for j in range(4):
                if self.ttnsc[i][j] > self.ndof:
                    x[0][self.ttnsc[i][j]-1] += self.f[i][j]

        self.r = [None for i in range(nall)]
        for i in range(len(x[0])):
            if i+1 > self.ndof:
                self.r[i] = x[0][i]

        '''
        for i in range(len(x[0])):
            if i+1 > self.ndof:
                self.r.append([round(x[0][i],5)])
        '''
        return

# call every generate methods
    def gen_yield(self):
        for ele in range(len(self.elements)):
            # truss element case
                if self.elements[ele].e_q[0][0] <= 0:
                    # tension
                    self.elements[ele].prop_yeield = abs(self.elements[ele].e_q[0][0]/self.elements[ele].area)/self.elements[ele].long_stress
                    self.elements[ele].iscompress = 0
                elif self.elements[ele].e_q[0][0] > 0:
                    # compression
                    # consider 1st bucking mode of pin-pin



                    buckling = (np.pi**2)*self.elements[ele].em*self.elements[ele].i[0][0]/(self.elements[ele].area*self.elements[ele].length**2)
                    yield_buckling = 0#abs(self.elements[ele].e_q[0][0])/buckling
                    yield_non_buckling = abs(self.elements[ele].e_q[0][0]/self.elements[ele].area)/self.elements[ele].long_stress
                    self.elements[ele].prop_yeield = max([yield_buckling,yield_non_buckling]) # compare 2 yields
                    self.elements[ele].iscompress = 1


    def gen_all(self):
        for i in range(len(self.elements)):
            self.elements[i].gen_length()
        self.gen_jp()
        self.gen_pj()

        self.gen_nsc()
        self.gen_tnsc()
        self.gen_ndof()
        self.gen_jlv()

        self.gen_global_k()
        self.gen_ssm()

        self.gen_d()
        self.gen_v()
        self.gen_u()
        self.gen_U_full()
        self.gen_q()
        self.gen_f()
        self.gen_r()
        self.gen_yield()

        # set target
        for i in range(len(self.nodes)):
            self.nodes[i].set_target()


#-------------------------------------

'''
Planar Truss is subjected to  Axial Load

1. Create Loads
2. Create Nodes
3. Set Loads for Nodes
4. Create Elements from Nodes
5. Create Model
'''

'''
# TEST MODEL Example3.8 Pg107
# Loads
# l1
l1 = Load()
l1.set_name(1)
l1.set_size(0, -300)
# l2
l2 = Load()
l2.set_name(2)
l2.set_size(150, 0)

# Nodes

# n1
n1 = Node()
n1.set_name(1)
n1.set_coord(144, 192)
n1.set_res(0,0)
n1.set_load(l1)
n1.set_load(l2)

# n2
n2 = Node()
n2.set_name(2)
n2.set_coord(0, 0)
n2.set_res(1, 1)

# n3
n3 = Node()
n3.set_name(3)
n3.set_coord(144, 0)
n3.set_res(1, 1)

# n4
n4 = Node()
n4.set_name(4)
n4.set_coord(288, 0)
n4.set_res(1, 1)

# Elements
# e1
e1 = Element()
e1.set_name(1)
e1.set_nodes(n2, n1)
e1.set_em(29000)
e1.set_i(1000)
e1.set_area(8)
# e2
e2 = Element()
e2.set_name(2)
e2.set_nodes(n3, n1)
e2.set_em(29000)
e2.set_i(1000)
e2.set_area(6)
# e3
e3 = Element()
e3.set_name(3)
e3.set_nodes(n4, n1)
e3.set_em(29000)
e3.set_i(1000)
e3.set_area(8)

# Model
test_model = Model()

test_model.add_load(l1)
test_model.add_load(l2)


test_model.add_node(n1)
test_model.add_node(n2)
test_model.add_node(n3)
test_model.add_node(n4)


test_model.add_element(e1)
test_model.add_element(e2)
test_model.add_element(e3)


test_model.gen_all()


print('d = {0}'.format(test_model.d))


def render(m1):
    #ORIGIN
    Xcoord = []
    Ycoord = []
    for i in range(len(m1.nodes)):
        Xcoord.append(m1.nodes[i].coord[0])
        Ycoord.append(m1.nodes[i].coord[1])

    Nodes = {'X':Xcoord, 'Y':Ycoord}
    nodeplot = pd.DataFrame(Nodes, columns = ['X', 'Y'])
    n = nodeplot.plot(x='X',y='Y',kind='scatter',color ='black',grid=True,legend=False)

    for i in range(len(m1.elements)):
        xstart = m1.elements[i].nodes[0].coord[0]
        xend = m1.elements[i].nodes[1].coord[0]
        ystart = m1.elements[i].nodes[0].coord[1]
        yend = m1.elements[i].nodes[1].coord[1]
        Xcoord = [xstart,xend]
        Ycoord = [ystart,yend]
        Elements = {'X':Xcoord, 'Y':Ycoord}
        elementplot = pd.DataFrame(Elements, columns = ['X', 'Y'])
        elementplot.plot(x='X',y='Y',kind='line',color ='black',grid=True, ax=n,legend=False)

    #SUPPORT
    Xcoord_Ysupport =[]
    Ycoord_Ysupport =[]
    Xcoord_XYsupport =[]
    Ycoord_XYsupport =[]
    for i in range(len(m1.nodes)):
        if (m1.nodes[i].res[0] == 0) and (m1.nodes[i].res[1] == 1):
            Xcoord_Ysupport.append(m1.nodes[i].coord[0])
            Ycoord_Ysupport.append(m1.nodes[i].coord[1])
        if (m1.nodes[i].res[0] == 1) and (m1.nodes[i].res[1] == 1):
            Xcoord_XYsupport.append(m1.nodes[i].coord[0])
            Ycoord_XYsupport.append(m1.nodes[i].coord[1])

    if Xcoord_Ysupport != 0:
        Ysupport = {'X':Xcoord_Ysupport, 'Y':Ycoord_Ysupport}
        Ysupportplot = pd.DataFrame(Ysupport, columns = ['X', 'Y'])
        Ysupportplot.plot(x='X',y='Y',kind='scatter',color ='grey',marker="^",s=100,grid=True, ax=n,legend=False)
    if Xcoord_XYsupport != 0:
        XYsupport = {'X':Xcoord_XYsupport, 'Y':Ycoord_XYsupport}
        XYsupportplot = pd.DataFrame(XYsupport, columns = ['X', 'Y'])
        XYsupportplot.plot(x='X',y='Y',kind='scatter',color ='black',marker="^",s=100,grid=True, ax=n,legend=False)

    #DEFROM
    NewXcoord = []
    NewYcoord = []
    for i in range(len(m1.nodes)):
        NewXcoord.append(m1.nodes[i].coord[0])
        NewYcoord.append(m1.nodes[i].coord[1])
    for i in range(len(m1.d)):
        for j in range(len(m1.tnsc)):
            if m1.tnsc[j][0] == i + 1:
                NewXcoord[j] += m1.d[i][0]
            if m1.tnsc[j][1] == i + 1:
                NewYcoord[j] += m1.d[i][0]
    NewNodes = {'X':NewXcoord, 'Y':NewYcoord}
    Newnodeplot = pd.DataFrame(NewNodes, columns = ['X', 'Y'])
    Newn = Newnodeplot.plot(x='X',y='Y',kind='scatter',color ='red',alpha=0.7,grid=True, ax=n,legend=False)

    for i in range(len(m1.elements)):
        Newxstart = m1.elements[i].nodes[0].coord[0]
        Newxend = m1.elements[i].nodes[1].coord[0]
        Newystart = m1.elements[i].nodes[0].coord[1]
        Newyend = m1.elements[i].nodes[1].coord[1]

        for k in range(len(m1.d)):
            if m1.tnsc[m1.elements[i].nodes[0].name-1][0] == k + 1:
                Newxstart += m1.d[k][0]
            if m1.tnsc[m1.elements[i].nodes[1].name-1][0] == k + 1:
                Newxend += m1.d[k][0]
            if m1.tnsc[m1.elements[i].nodes[0].name-1][1] == k + 1:
                Newystart += m1.d[k][0]
            if m1.tnsc[m1.elements[i].nodes[1].name-1][1] == k + 1:
                Newyend += m1.d[k][0]
            else:
                pass
        NewXcoord = [Newxstart,Newxend]
        NewYcoord = [Newystart,Newyend]
        NewElements = {'X':NewXcoord, 'Y':NewYcoord}
        Newelementplot = pd.DataFrame(NewElements, columns = ['X', 'Y'])
        Newelementplot.plot(x='X',y='Y',kind='line',color ='red',alpha=0.7,grid=True, ax=n,legend=False)



    plt.axis('equal')
    plt.show()

#render(test_model)
'''
