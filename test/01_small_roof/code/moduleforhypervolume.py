from collections import Counter

# SRC https://tryalgo.org/en/geometry/2016/06/25/union-of-rectangles/
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

OPENING = +1  # constants for events
CLOSING = -1  # -1 has higher priority

def union_rectangles_fastest(R):
    """Area of union of rectangles

    :param R: list of rectangles defined by (x1, y1, x2, y2)
       where (x1, y1) is top left corner and (x2, y2) bottom right corner
    :returns: area
    :complexity: :math:`O(n \\log n)`
    """
    if R == []:               # segment tree would fail on an empty list
        return 0
    X = set()                 # set of all x coordinates in the input
    events = []               # events for the sweep line
    for Rj in R:
        (x1, y1, x2, y2) = Rj
        assert x1 <= x2 and y1 <= y2
        X.add(x1)
        X.add(x2)
        events.append((y1, OPENING, x1, x2))
        events.append((y2, CLOSING, x1, x2))
    i_to_x = list(sorted(X))
    # inverse dictionary
    x_to_i = {i_to_x[i]: i for i in range(len(i_to_x))}
    L = [i_to_x[i + 1] - i_to_x[i] for i in range(len(i_to_x) - 1)]
    C = CoverQuery(L)
    area = 0
    previous_y = 0  # arbitrary initial value,
    #                 because C.cover() is 0 at first iteration
    for y, offset, x1, x2 in sorted(events):
        area += (y - previous_y) * C.cover()
        i1 = x_to_i[x1]
        i2 = x_to_i[x2]
        C.change(i1, i2, offset)
        previous_y = y
    return area


y_max = 1
BOX1 = (0.5,y_max-1,1,y_max-0.5)
BOX2 = (0.25,y_max-1,1,y_max-0.75)

print(union_rectangles_fastest([BOX1,BOX2]))
