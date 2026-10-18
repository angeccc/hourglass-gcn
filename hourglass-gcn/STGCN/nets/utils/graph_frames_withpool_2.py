import numpy as np

class Graph_pool():

    def __init__(self,
                 layout,
                 strategy,
                 pad=0,
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.seqlen = 2*pad+1
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)


        self.dist_center = self.get_distance_to_center(layout)
        self.get_adjacency(strategy)

    def get_distance_to_center(self,layout):
        dist_center = np.zeros(self.num_node)
        if layout == 'hm36_gt' or layout == 'hm36_sh':
            for i in range(self.seqlen):
                index_start = i*self.num_node_each
                dist_center[index_start+ 0:index_start+4] = [1,1,1,1]
                dist_center[index_start+4] = 0
        return dist_center

    def __str__(self):
        return self.A

    def graph_link_between_frames(self,base):
        return [((front ) + i*self.num_node_each, (back )+ i*self.num_node_each) for i in range(self.seqlen) for (front, back) in base]

    def get_edge(self, layout):
        if layout == 'hm36_gt' or layout == 'hm36_sh':
            self.num_node_each = 5

            self.num_node = self.num_node_each * self.seqlen
            view_link12_23 = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1) for j in range(self.num_node_each)]
            view_link_13 = [(j, 2 * self.num_node_each + j) for j in range(self.num_node_each)]
            view_link21_32 = [((i + 1) * self.num_node_each + j, i * self.num_node_each + j) for i in range(self.seqlen - 1) for j in range(self.num_node_each)]
            view_link_31 = [(2 * self.num_node_each + j, j) for j in range(self.num_node_each)]
            view_link = view_link12_23 + view_link_13
            self.view_wise = view_link_13 + view_link21_32
            self.view_counter = view_link12_23 + view_link_31

            self_link = [(i, i) for i in range(self.num_node)]

            neighbour_base = [(0,4), (1, 4), (2, 4), (3, 4)]
            self.neighbour_link_all = self.graph_link_between_frames(neighbour_base)

            sym_base = [(0, 1), (2, 3)]
            self.sym_link_all = self.graph_link_between_frames(sym_base)

            self.edge = self_link + self.neighbour_link_all + self.sym_link_all + view_link

            self.center = 5-1

        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                a_sym = np.zeros((self.num_node, self.num_node))
                a_wise = np.zeros((self.num_node, self.num_node))
                a_counter = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if (j,i) in self.sym_link_all or (i,j) in self.sym_link_all:
                                a_sym[j, i] = normalize_adjacency[j, i]
                            elif (j,i) in self.view_wise:
                                a_wise[j, i] = normalize_adjacency[j, i]
                            elif (j,i) in self.view_counter:
                                a_counter[j, i] = normalize_adjacency[j, i]
                            elif self.dist_center[j] == self.dist_center[i]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.dist_center[j] > self.dist_center[i]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_close)
                    A.append(a_further)
                    A.append(a_sym)
                    if self.seqlen > 1:
                        A.append(a_wise)
                        A.append(a_counter)
            A = np.stack(A)
            self.A = A

        else:
            raise ValueError("Do Not Exist This Strategy")

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD