from set_seed_global import seedThis
import numpy as np
from collections import Counter
import random
import numpy as np
random.seed(seedThis)
np.random.seed(seedThis)
def dominates(row, candidateRow):
    return sum([row[x] < candidateRow[x] for x in range(2)]) == 2

def simple_cull(Allpoint,optional=False):
    inputPoints = []#Allpoint.copy()
    all_points = []
    '''
    '''
    Ng = []
    for i in range(len(Allpoint)):
        if Allpoint[i][2] > 1 or Allpoint[i][3] > 1:
            Ng.append(i)
    for i in range(len(Allpoint)):
        if i not in Ng:
            inputPoints.append(Allpoint[i])
            all_points.append(Allpoint[i])

    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break

    is_front = [list(i) for i in paretoPoints]
    is_front = sorted(is_front, key=lambda x: x[0])
    '''
    if optional==True:
        print('________________________________________________')
        print(all_points)
        print('________________________')
    '''
    all_pass_maxx = sorted(all_points, key=lambda x: x[0])
    '''
    if optional==True:
        print(all_pass_maxx)
        print('________________________')
    '''
    all_pass_maxx = sorted(all_pass_maxx[-max([1,min(3,len(all_pass_maxx))]):], key=lambda x: x[1])
    '''
    if optional==True:
        print(all_pass_maxx)
        print('________________________________________________')
    '''

    all_pass_maxy = sorted(all_points, key=lambda x: x[1])

    all_pass_maxy = sorted(all_pass_maxy[-max([1,min(3,len(all_pass_maxy))]):], key=lambda x: x[0])

    '''
    if len(all_pass_maxx) != 0:
        all_pass_maxx = all_pass_maxx[-1]

    if len(all_pass_maxy) != 0:
        all_pass_maxy = all_pass_maxy[-1]
    '''

    is_front_and_edges = [list(i) for i in paretoPoints]

    for i in range(min([len(all_pass_maxx),1])):
        try:
            is_front_and_edges.append(all_pass_maxx[-i-1])
        except:
            pass
    for i in range(min([len(all_pass_maxy),1])):
        try:
            is_front_and_edges.append(all_pass_maxy[-i-1])
        except:
            pass


    is_front_and_edges = sorted(is_front_and_edges, key=lambda x: x[0])
    # add distance
    if len(is_front) > 1:
        distances = [None for i in range(len(is_front)-1)]
        #is_front = is_front.tolist()
        for i in range(len(is_front)-1):
            distances[i] = ((is_front[i][0]-is_front[i+1][0])**2 + (is_front[i][1]-is_front[i+1][1])**2)**0.5

        for i in range(len(is_front)):
            if i == 0:
                is_front[i].append(distances[i])
            elif i == len(is_front)-1:
                is_front[i].append(distances[-1])
            else:
                is_front[i].append(distances[i-1]+distances[i])
                #distances[i] = ((is_front[i][0]-is_front[i+1][0])**2 + (is_front[i][1]-is_front[i+1][1])**2)**0.5
    else:
        is_front[0].append(0)


    MAX_FRONT = 20
    if len(is_front) > MAX_FRONT:
        distributed_font = is_front[1:-1]
        distributed_font = sorted(distributed_font, key=lambda x: x[-1], reverse=True)
        #print(distributed_font)
        #print('-----')
        distributed_font = random.sample(distributed_font, MAX_FRONT-2)
        is_front = [is_front[0]] + distributed_font + [is_front[-1]]
        #is_front = random.sample(is_front,MAX_FRONT)
        #is_front = [is_front[0]]+random.sample(is_front[1:-1],MAX_FRONT-2)+[is_front[-1]]
    else:
        is_front = is_front
    if len(is_front_and_edges) > MAX_FRONT:
        distributed_font = is_front_and_edges[1:-1]
        distributed_font = sorted(distributed_font, key=lambda x: x[-1], reverse=True)
        #print(distributed_font)
        #print('-----')
        distributed_font = random.sample(distributed_font, MAX_FRONT-2)
        is_front_and_edges = [is_front_and_edges[0]] + distributed_font + [is_front_and_edges[-1]]
        #is_front = random.sample(is_front,MAX_FRONT)
        #is_front = [is_front[0]]+random.sample(is_front[1:-1],MAX_FRONT-2)+[is_front[-1]]
    else:
        is_front_and_edges = is_front_and_edges
    #print(is_front)
    #print(pareto_point)
    #array_front = np.array(pareto_point)
    #print(array_front)
    #is_front = array_front[(array_front)[:, 1].argsort()]
    #print('----------------------')
    #print(is_front)
    #print('----------------------')
    max_distance = 0
    dis_distance = 0

    is_front = [i[:-1] for i in is_front] # remove distance
    distances = [None for i in range(len(is_front)-1)]
    #is_front = is_front.tolist()
    for i in range(len(is_front)-1):
        distances[i] = ((is_front[i][0]-is_front[i+1][0])**2 + (is_front[i][1]-is_front[i+1][1])**2)**0.5
    ###
    if len(is_front) >= 2:
        max_distance = max(distances)
        dis_distance = (sum([((x - max_distance/len(distances)) ** 2) for x in distances]) / len(distances))**0.5
        sum_distance = sum(distances)
    if len(is_front) <= 1:
        dis_distance = 1
        max_distance = 0
        sum_distance = 0

    # p-of distance
    p_norm_val = 10
    power_distances = [abs(i)**p_norm_val for i in distances]
    p_norm_distance = sum(power_distances)**(1/p_norm_val)

    # crowding distances
    c_distance = [None for i in range(len(is_front)-2)]

    if len(is_front) > 3:
        #print(c_distance)
        for i in range(1,len(is_front)-1):
            x_range = abs(is_front[i-1][0]-is_front[i+1][0])
            y_range = abs(is_front[i-1][1]-is_front[i+1][1])
            c_distance[i-1] = x_range+y_range

        if np.sum(np.array(c_distance)) == 0:
            std_cd = 1
            p_norm_inv_cd = 0
        else:
            #print(c_distance)
            c_distance = np.array(c_distance)/np.max(np.array(c_distance))
            #print(c_distance)
            std_cd = np.std(c_distance)
            #print(std_cd)
            power_inv_cd = [abs(i)**p_norm_val for i in c_distance]
            p_norm_inv_cd = sum(power_inv_cd)**(1/p_norm_val)
    else:
        std_cd = 1

        p_norm_inv_cd = 0

    #print(c_distance)
    '''
    if optional==True:
        print('====================================================')
        print(is_front)
        print('-----------------------------------')
        print(is_front_and_edges)
        print('====================================================')
    '''

    ###
    #print('----------------------')
    #print(sum_distance)
    #print('----------------------')
    if optional==True:
        return is_front, max_distance, dis_distance, p_norm_inv_cd, sum_distance, std_cd, is_front
    else:
        return is_front, max_distance, dis_distance, p_norm_inv_cd, sum_distance, std_cd


# SRC https://tryalgo.org/en/geometry/2016/06/25/union-of-rectangles/
# FOR computing hypervolume
class CoverQuery:
    """Segment tree to maintain a set of integer intervals
    and permitting to query the size of their union.
    """
    def __init__(self, L):
        """creates a structure, where all possible intervals
        will be included in [0, L - 1].
        """
        assert L != []              # L is assumed sorted
        self.N = 1
        while self.N < len(L):
            self.N *= 2
        self.c = [0] * (2 * self.N)         # --- covered
        self.s = [0] * (2 * self.N)         # --- score
        self.w = [0] * (2 * self.N)         # --- length
        for i, _ in enumerate(L):
            self.w[self.N + i] = L[i]
        for p in range(self.N - 1, 0, -1):
            self.w[p] = self.w[2 * p] + self.w[2 * p + 1]

    def cover(self):
        """:returns: the size of the union of the stored intervals
        """
        return self.s[1]


    def change(self, i, k, offset):
        """when offset = +1, adds an interval [i, k],
        when offset = -1, removes it
        :complexity: O(log L)
        """
        self._change(1, 0, self.N, i, k, offset)


    def _change(self, p, start, span, i, k, offset):
        if start + span <= i or k <= start:   # --- disjoint
            return
        if i <= start and start + span <= k:  # --- included
            self.c[p] += offset
        else:
            self._change(2 * p, start, span // 2, i, k, offset)
            self._change(2 * p + 1, start + span // 2, span // 2,
                         i, k, offset)
        if self.c[p] == 0:
            if p >= self.N:                   # --- leaf
                self.s[p] = 0
            else:
                self.s[p] = self.s[2 * p] + self.s[2 * p + 1]
        else:
            self.s[p] = self.w[p]

#OPENING = +1  # constants for events
#CLOSING = -1  # -1 has higher priority
def union_rectangles_fastest(R,OPENING,CLOSING,ref_point=[1,1]):


    #print('-----------------------------------------------------')
    #print(R)
    #print('-----------------------------------------------------')
    """Area of union of rectangles

    :param R: list of rectangles defined by (x1, y1, x2, y2)
       where (x1, y1) is top left corner and (x2, y2) bottom right corner
    :returns: area
    :complexity: :math:`O(n \\log n)`
    """
    if R == []:               # segment tree would fail on an empty list
        return 0
    if len(R) == 1 and R[0][0] == 1 and R[0][1] == 1:        # segment tree would fail on an empty list
        return 0
    X = set()                 # set of all x coordinates in the input
    events = []               # events for the sweep line
    #print(R)

    all_x = [None for i in R]
    all_y = [None for i in R]
    for num in range(len(R)):
        #print(Rj)
        x = min([R[num][0],1])
        y = min([R[num][1],1])
        #[x,y] = Rj
        #print(x)
        #print(y)
        #x1, y1, x2, y2 = round(x,4), round(1-ref_point[1],4), round(ref_point[0],4), round(ref_point[1]-y,4)
        x1, y1, x2, y2 = x, 0, 1, 1-y
        #print(Rj)
        all_x[num] = R[num][0]
        all_y[num] = R[num][1]
        assert x1 <= x2 and y1 <= y2


        X.add(x1)
        X.add(x2)
        events.append((y1, OPENING, x1, x2))
        events.append((y2, CLOSING, x1, x2))

    i_to_x = list(sorted(X))
    # inverse dictionary
    x_to_i = {i_to_x[i]: i for i in range(len(i_to_x))}
    L = [i_to_x[i + 1] - i_to_x[i] for i in range(len(i_to_x) - 1)]
    #print(L)
    if L == []:
        L = [0]
    C = CoverQuery(L)
    area = 0
    previous_y = 0  # arbitrary initial value,
    #                 because C.cover() is 0 at first iteration
    for y, offset, x1, x2 in sorted(events):
        area += (y - previous_y) * C.cover()
        #print(area)
        i1 = x_to_i[x1]
        i2 = x_to_i[x2]
        C.change(i1, i2, offset)
        previous_y = y

    #print(min(all_x))
    #print(min(all_y))

    remove_area = (1-ref_point[0])*(1-min(all_x)) + (1-ref_point[1])*(1-min(all_y)) - (1-ref_point[0])*(1-ref_point[1])

    return area - remove_area

'''
point_1 = [ 0.5, 0.5, 0.8, 0.8]
point_2 = [ 1.0, 1.0, 0.8, 0.8]
point_3 = [ 0.25, 0.75, 0.8, 0.8]
point_4 = [ 0.75, 0.25, 0.8, 0.8]
points = [point_1,point_2,point_3,point_4]

OPENING = +1  # constants for events
CLOSING = -1  # -1 has higher priority
area =  union_rectangles_fastest(points,OPENING,CLOSING,ref_point=[1,1])
print(area)
area =  union_rectangles_fastest(points,OPENING,CLOSING,ref_point=[1,0.75])
print(area)
'''

'''
point_1 = [ 0.5, 0.7]
point = np.random.random((500,4))
point = point.tolist()
#print(point)
#print('------------------')
is_front, max_distance, dis_distance = simple_cull(point)


for i in range(len(is_front)):
    print(is_front[i])
    if i != len(is_front)-1:
        print(dis_distance[i])
print('------------------')

from matplotlib import pyplot as plt
x = [i[0] for i in is_front]
y = [i[1] for i in is_front]
plt.scatter(x,y)
plt.show()

'''
