import torch #定义网络层都import这两个
import torch.nn as nn

from nets.utils.tgcn import ConvTemporalGraphical #这里以主函数py文件为标准，nets与main_graph同级别
from nets.utils.dotmul import per_node_dot
from nets.utils.graph_frames import Graph
from nets.utils.graph_frames_withpool_2 import Graph_pool
from nets.non_local_embedded_gaussian import NONLocalBlock2D


inter_channels = [128, 128, 256] #类外定义，可被此文档中所有函数调用
fc_out = inter_channels[-1] #list[-1],索引，表示最后一个
fc_unit = 512
class Model(nn.Module): #网络层，都要加nn.Module父类     定义图结构、网络中各种模块和层

    def __init__(self, opt):
        super().__init__()   #super()等同于Model的父类，执行父类的__init__()函数，此时self仍是Model，定义的self._backend,_parameters等继承到Model类,表面上只增加这些属性，
                             #实际上父类所有子函数都可调用。有此句才能继承父类
        self.momentum = 0.1
        self.in_channels = opt.in_channels #2,
        self.out_channels = opt.out_channels #3 输出通道数
        self.layout = opt.layout
        self.strategy = opt.strategy
        self.cat = True
        self.inplace = True
        self.pad = opt.pad
        #定义图结构
        self.graph = Graph(self.layout, self.strategy, pad=opt.pad)
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).cuda() #torch.tensor将数组转成tensor， 原dtype是float64，现改为torch中的float32，
            #Tensor类:data(Tensor6,51,51,再下层的内容和A下层的内容一样),device(device类：cpu),dtype(dtype类：torch.float32)，requires_grad(bool:false，毗邻矩阵是固定的，
            # 不加入梯度计算图)）,    shape（Size类：torch.Size([6,51,51]))。   .cuda()，加载到GPU上运算,因为要参与网络计算.
        self.graph_pool = Graph_pool(self.layout, self.strategy, pad=opt.pad) #每帧五节点的图
        self.A_pool = torch.tensor(self.graph_pool.A, dtype=torch.float32, requires_grad=False).cuda()

        kernel_size = self.A.size(0) #6种关联  tensor.shape[0]与tensor.size(0)一样; tensor.size()=torch.Size([6,51,51]),tensor.size()[0]=6
                                     # 无tensor.shape[:]=tensor.shape=torch.Size([6,51,51])????????????
        kernel_size_pool = self.A_pool.size(0) #6

        self.data_bn = nn.BatchNorm1d(self.in_channels * self.graph.num_node_each, self.momentum)  #网络输入通道数2, x17=34,0.1

        self.st_gcn_networks = nn.ModuleList(  #几个块组成的list，  参数是tuple,两对括号
            (st_gcn(self.in_channels, inter_channels[0], kernel_size, self.A.shape[1], residual=False),  #2,128,6，51
            st_gcn(inter_channels[0], inter_channels[1], kernel_size, self.A.shape[1]),                  #128,128,6，51
            st_gcn(inter_channels[1], inter_channels[2], kernel_size, self.A.shape[1]), )                #128,256,6，51
        )

        self.st_gcn_pool = nn.ModuleList(
            (st_gcn(inter_channels[-1], fc_unit, kernel_size_pool, self.A_pool.shape[1]),   #256,512,6，15
            st_gcn(fc_unit, fc_unit, kernel_size_pool, self.A_pool.shape[1]),)               #512,512,6，15
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(fc_unit, fc_unit, kernel_size=(3, 1), padding=(1, 0)),  #512,512   (256,512,3,1),第三维pad=1,变为5，第四维pad=0
            nn.BatchNorm2d(fc_unit, momentum=self.momentum),#512  momentum：移动平均的动量值
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.25))

        self.conv2 = nn.Sequential(
            nn.Conv2d(fc_unit*2, fc_out, kernel_size=(1, 1), padding=(0, 0)), #1024,256
            nn.BatchNorm2d(fc_out, momentum=self.momentum), #256
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.1)) #通道数更多，dropout稍大 0.05 -> 0.1

        self.non_local = NONLocalBlock2D(in_channels=fc_out*2, sub_sample=False)  #256x2=512

        #最后的全连接层
        fc_in = inter_channels[-1]+fc_out if self.cat else inter_channels[-1]     #256+256=512   True: concatinate coarse and fine features,   False: add coarse and fine features
        self.fcn = nn.Sequential(
            nn.Dropout(0.1, inplace=True),
            nn.Conv2d(fc_in, self.out_channels, kernel_size=1))  #512,3


    def graph_max_pool(self, x, p,stride=None):  # p:（1，3）
        if max(p) > 1: #3>1，true
            if stride is None:#true
                x = nn.MaxPool2d(p)(x)  #从[256,256,3,3]到[256,256,3,1]
            else:
                x = nn.MaxPool2d(kernel_size=p, stride=stride)(x)  # B x F x V/p
            return x
        else:
            return x


    def forward(self, x, out_all_frame=False):#out_all_frame=true    x:tensor[256,2,3,17,1]
        N, C, T, V, M = x.size() #和列表差不多，逗号赋值
        x = x.permute(0, 4, 3, 1, 2).contiguous() #256,1,17,2,3
        x = x.view(N * M, V * C, T) #256,34,3;
       #17个节点的x、y坐标，在三帧上的3个数的分布相近，通过BN将它们的分布设置在(-1,1)?所以1dBN的1d指的是维度中的3？？？？？
        x = self.data_bn(x) #结果：tensor[256，34，3];  self.data_bn(x)等价于nn.BatchNorm1d( , )(x),运行它的forward，输入数据require_grad为false，经data_bn后，x变为链式的一环，变为true。
        x = x.view(N, M, V, C, T) #[256, 1, 17, 2, 3]
        x = x.permute(0, 1, 3, 4, 2).contiguous() #[256, 1, 2, 3, 17]
        x = x.view(N * M, C, 1, -1) #view(256,2,1,-1)=[256, 2, 1, 51]

        gcn_list = list(self.st_gcn_networks)  #st_gcn_networks:{Modulelist:3};  list的结果为[ st_gcn,st_gcn,st_gcn ];
        for i_gcn, gcn in enumerate(gcn_list): #tuple不可改元素的值;
            x, _ = gcn(x, self.A) #st_gcn类也有Module父类，  ()跳到st_gcn的forward     x结果:256，128，1，51

        x = x.view(N, -1, T, V) #[256,256,1,51].view(256,-1,3,17)=[256,256,3,17]
        # 池化
        for i in range(len(self.graph.part)): #len( [[11, 12, 13], [14, 15, 16], [4, 5, 6], [1, 2, 3], [0, 7, 8, 9, 10]] )=5
            num_node = len(self.graph.part[i]) #第i个部分的节点数，分别为3，3，3，3，5
            x_i = x[:, :, :, self.graph.part[i]] #[11,12,13];   256,256,3,3
            x_i = self.graph_max_pool(x_i, (1, num_node)) #5部分分别最大池化 256,256,3,1
            x_sub1 = torch.cat((x_sub1, x_i), -1) if i > 0 else x_i #i=0时，x_sub1等于x_i,之后叠加，-1表示最后一维，最终得到256，256，3，5

        x_sub1, _ = self.st_gcn_pool[0](x_sub1.view(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone())#.view(256,-1,1,3x5)=(256,256,1,15); 6x15x15,为何要clone  x_sub1:256,512,1,15
        x_sub1, _ = self.st_gcn_pool[1](x_sub1, self.A_pool.clone())  #Modulelist类，[ ]调用其__getitem__     返回的A下面用不到，用短横线表示     x_sub1:256,512,1,15
        x_sub1 = x_sub1.view(N, -1, T, len(self.graph.part)) #.view(256,-1,3,5)=[256,512,3,5]

        x_pool_1 = self.graph_max_pool(x_sub1, (1, len(self.graph.part)))  # 再次池化,结果256,512,3,1
        x_pool_1 = self.conv4(x_pool_1)# 256,512,3,1

        x_up_sub = torch.cat((x_pool_1.repeat(1, 1, 1, len(self.graph.part)), x_sub1), 1) #上采样   .repeat=256,512,3,5,和[256,512,3,5]在第二维cat, 结果是256，1024，3，5
        x_up_sub = self.conv2(x_up_sub) #256,256,3,5
        #上采样，5到17
        x_up = torch.zeros((N * M, fc_out, T, V)).cuda() #256,256,3,17,维度要括号括起来tuple，传到GPU
        for i in range(len(self.graph.part)): #5
            num_node = len(self.graph.part[i]) #分别取3,3,3,3,5
            x_up[:, :, :, self.graph.part[i]] = x_up_sub[:, :, :, i].unsqueeze(-1).repeat(1, 1, 1, num_node)#[256,256,3].unsq[-1]=[256,256,3,1].repeat=[256,256,3,5]
        #non-local and fcn
        x = torch.cat((x, x_up), 1)#x是池化之前的,   256，256，3，17和256，256，3，17。结果是256，512，3，17
        x = self.non_local(x)  #256，512，3，17  跳到_NonLocalBlockND的forward
        x = self.fcn(x) # 256, 3, 3, 17 ,第1个3表示输出通道数，三维坐标
        # output
        x = x.view(N, M, -1, T, V).permute(0, 2, 3, 4, 1).contiguous() # .view(256,1,-1,3,17)=256,1,3,3,17.permute=256,3,3,17,1
        if out_all_frame: #true
            x_out = x  #256,3,3,17,1
        else:
            x_out = x[:, :, self.pad].unsqueeze(2) #[256,3,17,1].unsq=256,3,1,17,1
        return x_out

class st_gcn(nn.Module): #stgcn模块定义

    def __init__(self, in_channels, out_channels, kernel_size, node_num, stride=1, dropout=0.05, residual=True):
        super().__init__()
        self.inplace = True
        self.momentum = 0.1

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size) #2,128,6;  128,128,6;  128,256,6;  256,512,6;  512,512,6  对应5个stgcn模块
        self.dotmul = per_node_dot(out_channels, node_num)    #128,51;  128,51;  256,51;  512,15;   512,15
        self.tcn = nn.Sequential(  #只有一对括号  必须按次序运行，这和modulelist不同。
            nn.BatchNorm2d(out_channels, momentum=self.momentum),    #128;  128;  256;  512;  512
            nn.Dropout(0.05), #0.05是丢弃的比例
        )

        if not residual:   #residual=false;  true;  true;  True;  True
            self.residual = lambda x: 0    #lambda表示一个函数，x是参数，0与下面的x是返回值。  不进行残差，返回0    第1个st_gcn运行
        elif (in_channels == out_channels) and (stride == 1): #两个条件分别加括号 2!=128;  128=128；  128!=256;  256!=512;   512=512
            self.residual = lambda x: x    #进行残差，返回x,相加的支路为x    第2个st_gcn运行
        else:                              #第3,4个st_gcn运行
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),#128; 256  #256; 512  #会有_pair函数将kernelsize从1变成（1，1）   #1
                nn.BatchNorm2d(out_channels, momentum=self.momentum),  #256; 512
            )

        self.relu = nn.ReLU(inplace=self.inplace) #True     inplace = False时,不会修改输入对象的值,返回一个新创建的对象,即不覆盖。

    def forward(self, x, A):  # A:tensor[6,51,51]   x:(256, 2, 1, 51)
        res = self.residual(x)   #lamda x:0(x) ==> 0,没有残差分支;   256,128,1,51;   256，256，1，51;   256,512,1,15;    256,512,1,15
        x = self.gcn(x) #结果：256，6，128，1，51;   256，6，128，1，51;   256，6，256，1，51;   256，6，512，1，15;    256，6，512，1，15
        x, A = self.dotmul(x, A)  #结果：256，128，1，51；   256，128，1，51；   256，256，1，51;    256，512，1，15;    256，512，1，15
        x = self.tcn(x) + res  #sequential也有forward函数   256，128，1，51;   256，128，1，51;  256，256，1，51;  256，512，1，15;   256，512，1，15

        return self.relu(x), A
