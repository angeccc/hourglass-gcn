import numpy as np

class Graph(): #三个双引号开始和结束，注释。
    """
        strategy (string): must be one of the follow candidates
        - spatial: Clustered Configuration
        dilation (int): controls the spacing between the kernel points
    """
    def __init__(self,
                 layout,
                 strategy,
                 pad=0,  #1
                 max_hop=1,
                 dilation=1):

        self.max_hop = max_hop  #重新定义为self.xx,使其它子函数可调用
        self.dilation = dilation
        self.seqlen = 2*pad+1   # 3帧

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)  #get_hop_distance函数在下面定义时没缩进，与class同级，前不加self; self.edge是属性，前面定义了，此处才能用。
        self.dist_center = self.get_distance_to_center(layout)
        self.get_adjacency(strategy)

    def get_distance_to_center(self,layout):
        dist_center = np.zeros((self.num_node)) #51个节点到各自帧的中心节点（7，24，31）的距离
        if layout == 'hm36_gt':
            for i in range(self.seqlen):   # 0,1,2
                index_start = i*self.num_node_each #*17
                dist_center[index_start+0 : index_start+7] = [1, 2, 3, 4, 2, 3, 4] #一维数组，前七个数用一个list赋值
                dist_center[index_start+7 : index_start+11] = [0, 1, 2, 3]
                dist_center[index_start+11 : index_start+17] = [2, 3, 4, 2, 3, 4]
        return dist_center

    def __str__(self):   #？？？？
        return self.A #def __str__(self):用print输出对象，要定义__str__(self)方法，会打印return的数据

    def graph_link_between_frames(self,base):
        return [((front - 1) + i*self.num_node_each, (back - 1)+ i*self.num_node_each) for i in range(self.seqlen) for (front, back) in base] # 节点从1开始，所以先减1，再加上17的倍数。

    def basic_layout(self,neighbour_base, sym_base):
        self.num_node = self.num_node_each * self.seqlen #17x3=51   self.num_node_each是在get_edge中定义的，且按顺序在此句之前，所以此处可用。
        view_link12_23 = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1)#整个是列表[]，两个for循环是嵌套的，i取0，
                          # j取0到16，i取1，j取0到16，每一对（i,j），产生一个元组，所有元组自动累积。  虽换行但有[]     for xx in range(整数n)，取0到n-1
                     for j in range(self.num_node_each)]  #[(0, 17), (1, 18),..(33, 50)]  34个
        view_link_13 = [(j, 2 * self.num_node_each + j) for j in range(self.num_node_each)]  #[(0, 34), (1, 35),..(16, 50)] 17个
        view_link21_32 = [((i + 1) * self.num_node_each + j,i * self.num_node_each + j) for i in range(self.seqlen - 1)
                     for j in range(self.num_node_each)]  #[(17, 0), (18, 1)..(50, 33)]
        view_link_31 = [(2 * self.num_node_each + j, j) for j in range(self.num_node_each)]  #[(34, 0), (35, 1),..(50, 16)]
        view_link = view_link12_23 + view_link_13  #三视角图中所有51个视角节点对，不分前后顺序； list直接相加进行连接
        self.view_wise = view_link_13 + view_link21_32 #顺时针，51个
        self.view_counter = view_link12_23 + view_link_31  #逆时针，51个

        self_link = [(i, i) for i in range(self.num_node)] #自身连接[(0, 0), (1, 1), (2, 2), (3, 3)....(50,50),51个

        self.neighbour_link_all = self.graph_link_between_frames(neighbour_base)  #16x3=48 三帧的相邻节点对
        self.sym_link_all = self.graph_link_between_frames(sym_base)     #6x3=18,三帧的对称节点对

        return self_link, view_link

    def get_edge(self, layout):
        if layout == 'hm36_gt':  #True
            self.num_node_each = 17
            neighbour_base = [(1, 2), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 1), (9, 8), (10, 9), (11, 10), (12, 9),
                              (13, 12), (14, 13), (15, 9), (16, 15), (17, 16)]  #16个相邻节点对，每对下标不分前后顺序，从1开始
            sym_base = [(7, 4), (6, 3), (5, 2), (12, 15), (13, 16), (14, 17)]   #6个对称节点对
            self_link, view_link = self.basic_layout(neighbour_base, sym_base)  ##return两个值， 三帧的关节自连接，视角节点对

            self.la, self.ra = [11, 12, 13], [14, 15, 16]  #左/右臂  关节点分五类，骨架图如打印论文所示，等号左右都用逗号。
            self.ll, self.rl = [4, 5, 6], [1, 2, 3]       #左/右腿
            self.cb = [0, 7, 8, 9, 10]                    #中心骨干
            self.part = [self.la, self.ra, self.ll, self.rl, self.cb]

            self.edge = self_link + self.neighbour_link_all + self.sym_link_all + view_link #图中所有连接的节点对，不分前后顺序

            self.center = 8 - 1   #中心节点设为7  位于脖子与中心髋之间
        else:
            raise ValueError("Do Not Exist This Layout.") #手动报异常，见笔记raise

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)   #range(0,2,1) ; valid_hop不是0和1的列表或元组，是range类
        adjacency = np.zeros((self.num_node, self.num_node))   #51x51
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1       #广播，每个元素都判断==，节点间距离为0和1的，元素为1，得到未标准化的毗邻矩阵A
        normalize_adjacency = normalize_digraph(adjacency)  #标准化

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:      #0和1  注意嵌套运行顺序。
                a_root = np.zeros((self.num_node, self.num_node))  #将连接的节点对分为六种，每种对应一个毗邻矩阵，六个合起来是此前的AD
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                a_sym = np.zeros((self.num_node, self.num_node))
                a_wise = np.zeros((self.num_node, self.num_node))
                a_counter = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node): #51
                    for j in range(self.num_node): #51
                        if self.hop_dis[j, i] == hop: #hop=0,只有对角线元素符合，表示节点自身,第三个elif符合;  hop=1,所有连接的节点对
                            if (j,i) in self.sym_link_all or (i,j) in self.sym_link_all: #or和and，满足其一和都满足
                                a_sym[j, i] = normalize_adjacency[j, i]
                            elif (j,i) in self.view_wise:   #if和else之间可有很多elif
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
                    A.append(a_root) #list.append()
                else:
                    A.append(a_close)
                    A.append(a_further)
                    A.append(a_sym)
                    if self.seqlen > 1:
                        A.append(a_wise)
                        A.append(a_counter)
            A = np.stack(A)         # 6x51x51
            self.A = A

        else:
            raise ValueError("Do Not Exist This Strategy")

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node)) #np定义ndarray(数组)，参数为元组，51x51，表示每个节点与51个节点的连接关系。
    for (i, j) in edge: #所有连接的节点对列表。每次取一个元组 ,   (i,j) bu加括号也行
        A[j, i] = 1 #数组索引[a ,b ]:第a+1行b+1列的元素     [ : , 1]:所有行的第2列元素
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf   #全为inf,正无穷大，float型，python中数组的广播，每个元素加这个数。
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]#[I,A]  matrix_power，数组的幂，即d个A相乘
    arrive_mat = (np.stack(transfer_mat) > 0) #优先级，stack高于是否大于，除非加（）强行改变优先级;np.stack堆叠操作，把列表中两个元素堆叠成数组; 数组中属性：min(最小值，float)
    # max， shape(tuple，本例是（2，51，51）)，  dtype(dtype类，取值为float)，  size(int)，  array(是NdArrayItemsContainer类，id 0：ndarray(51,51)， id 1：ndarray(51,51),
    # 每个array像上述循环直到最后一维度，一行51个元素时，shape是（1，51），但定义为（51，）这是一个元素的元组，加逗号是避免歧义，  （51）可被理解是数字51加了括号);
    # 是否大于也可用传播; 大于号优先级大于=，所以这里不加（）也可以.
    for d in range(max_hop, -1, -1): # 1到-1，步长-1(第三个数),更完整的range用法     1，0
        hop_dis[arrive_mat[d]] = d #d=1时，array[1]相当于array[1, : , ： ]，    array[array],两个都是（51，51），第二个的元素是True/False，如果是True,第一个array对应位置的元素赋值1，
                                   #所有连接的位置变为1； d=0时，自关联节点对变为0；    设置max_hop=1，表示只关注一步相连
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)     #np.sum，axis=0,按最后一维求和，Dl.shape=tuple(51, )
    num_node = A.shape[0] #array.shape=tuple(51,51)   tuple[0]=51
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:   #说明这一列有至少一个1元素
            Dn[i, i] = Dl[i]**(-1) #Dn的对角元素从0变为每列1个数的倒数  两个乘号表示次方
    AD = np.dot(A, Dn)   #数组作矩阵乘法，使A每列的1标准化为每列1个数的倒数      对tensor z = x.mm(y) , (m,n) x (n,p)
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