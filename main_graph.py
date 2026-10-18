from __future__ import absolute_import
from __future__ import division #一般主函数都加这三行，其它py文件不加,调用其他py中函数时也处于这三条的作用下。单独调试其它py文件，也要加.
from __future__ import print_function #python3.6/__future__.py   from py名 import class名/函数名，import多个时，可逗号隔开；  from __future__ import *则import了所有   调用xx函数时：xx() ;
#import 大部分是python3.6文件夹或python3.6/site-packages中的
import math #/home/chuanjiang/anaconda3/envs/STGAN/lib/python3.6/lib-dynload/math.cpython-36m-x86_64-linux-gnu.so'>  数学公式定义、常量如pi,nan的数值，pycharm自带的，不用写具体路径。不知何时会用到，一般加上此句。
import os #/python3.6/os.py  把py里所有函数导入，  调用xx函数时：  os.xx（）
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #指定GPU工作; =是赋值，==是判断是否等于   或终端中CUDA_VISIBLE_DEVICES=3，4，5 python train.py ...
import random  #/python3.6/random.py
import time #/home/chuanjiang/.cache/JetBrains/PyCharm2021.2/python_stubs/-1468751694/time.py 内置
import logging #/python3.6/logging/__init__.py   import文件夹,则运行里面的__init__.py
import pickle #/python3.6/pickle.py    没看到程序中引用，能否去除？或其它子文档中要调用？
import torch #/python3.6/site-packages/torch/__init__.py
import torch.utils.data  #文件.文件.文件;    如果as xx，则调用时可简化为xx,   能否去除
import torch.nn as nn
import sys #/-1468751694/sys.py 能否去除
import h5py #/site-packages/h5py/__init__.py 能否去除
import copy #/python3.6/copy.py   能否去除
import re #/python3.6/re.py  能否去除
import numpy as np #/site-packages/numpy/__init__.py
from six.moves import xrange  #内置 range相关
import socket #/python3.6/socket.py  能否去除
import torch.optim as optim #torch/optim/__init__.py

from utils.data_utils import define_actions  #与main_graph同级目录的utils文件下的data_utils.py import函数
from utils.utils1 import save_model
from nets.post_refine import post_refine
from train_graph_time import train, val
from opt1 import opts
from data.load_data_hm36 import Fusion


opt = opts().parse() #运行opts类中的__init__函数，然后运行该类下的parse子函数(只有self参数)，并返回该子函数的return值。   opt: Namespace类(actions='*', batchSize=256,..., workers=6)
opt.manualSeed = 1
print("Random Seed: ", opt.manualSeed) #一个str,一个数，都可打印，逗号隔开
random.seed(opt.manualSeed) #此句和下句都只设置一个条件。 后面普通数据和tensor初始化时将受影响，产生固定初始化，如random.random（），torch.random()等，目的是复现结果和便于比较不同参数的网络性能。
torch.manual_seed(opt.manualSeed) #torch的_init.py__中：import torch.random， random中有manual_seed，所以torch.manual_seed，种子数取不同数，初始化不同。
try:    #见笔记
    os.makedirs(opt.save_dir) #已经建立过这文件，则异常，执行下句。
except OSError:
    pass #pass是空语句，占位语句，保持程序结构的完整性。
#程序续行用\，此行后面不能加#注释。  创建log文件。文件名为./.../cpn/train_test.log。 format：log文件中输入时间，然后空格，然后输入信息，datefmt是日期格式：年月日 空格 时分秒
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(opt.save_dir, 'train_test.log'), level=logging.INFO)
logging.info('======================================================')    #log文件中写入括号里的内容

if opt.pad > 0: #导入单帧/多帧模型
    from nets.st_gcn_multi_frame import Model
else:
    from nets.st_gcn_single_frame import Model
model = {} #空字典
model['st_gcn'] = Model(opt).cuda()  #opt是__init__的参数; Module定义的每一网络层的parameter默认在cpu上;  .cuda()是Module中的子函数，
                                     #这里被继承给了Model， tensor, self.A则需单独cuda化。
model['post_refine'] = post_refine(opt).cuda()  #待看

#加载3d-gt数据
root_path = opt.root_path  #'/home/chuanjiang/Projects/STGCN/STGCN/data/dataset/
if opt.dataset == 'h36m':  #true
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz' #'.../dataset/data_3d_h36m.npz'
    from data.common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path, opt)
else:
    raise KeyError('Invalid dataset')

if opt.pro_train: #1
    train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)  #train区分训练集和测试集
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), pin_memory=False)#main.py
                                                                               #L66-69 shuffle取值-->dataloader.py L483 -->sampler.py L139 --> L50-->数据集乱序
if opt.pro_test:#1
    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path=root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), pin_memory=False) #

#对模型的所有可学习的参数设置优化器和学习率变化
lr = opt.learning_rate #1e-04
all_param = []
for i_model in model:  # model:{ 'stgcn':Model类,  'postrefine': postrefine类 }
    all_param += list(model[i_model].parameters()) #parameters()是Module类的子函数，结果是generator类,list后变[ Parameter类,...,Parameter]，Parameter类和tensor类完全一样，只是Parameter的requires_grad=true。
                                                   #Parameter内的值是定义网络层时的预初始化，接近0
if opt.optimizer == 'SGD': #false
    optimizer_all = optim.SGD(all_param, lr=opt.learning_rate, momentum=0.9, nesterov=True, weight_decay=opt.weight_decay)
elif opt.optimizer == 'Adam': #true
    optimizer_all = optim.Adam(all_param, lr=lr, amsgrad=True) #定义Adam类的优化器，没有真正执行
optimizer_all_scheduler = optim.lr_scheduler.StepLR(optimizer_all, step_size=5, gamma=0.1) #每隔5 epoch，lr变为之前的0.1,这句没用，用另一种方式达成此目的

#重新加载训练过的模型
stgcn_dict = model['st_gcn'].state_dict()  #state_dict是Module的子函数，   stgcn_dict:{OrderedDict:88},按顺序存放的dict;   如：'data_bn.weight'=tensor(34,)，各层的权重和bias
if opt.stgcn_reload == 1:   #1  是否重新加载
    pre_dict_stgcn = torch.load(os.path.join(opt.previous_dir, opt.stgcn_model)) #join('../results/3_frame/st_gcn/no_pose_refine/cpn/','model_st_gcn_3_eva_xyz_5089.pth'); torch.load加载
                                                                                 # pre_dict_stgcn:和stgcn_dict数据结构一样
    for name, key in pre_dict_stgcn.items():  #类似dict.items()
        if name.startswith('A') == False:  #以A开头，为何作此判断
            stgcn_dict[name] = pre_dict_stgcn[name] #类似dict[ ]赋值，训练过的参数赋当前模型
        if name == 'st_gcn_networks.0.dotmul.per_node_weight' or name == 'st_gcn_networks.1.dotmul.per_node_weight' or name == 'st_gcn_networks.2.dotmul.per_node_weight':#可用in [ , , ]简化
            key_abs = torch.abs(key)  #torch.abs(tensor),绝对值
            key = torch.mean(key, 1, keepdim=False, out=None)#取均值，  6,128,1,51变为6,1,51，  1表示针对第二维，keepdim为false,不保持维度，如果为true，则为6，1，1，51
            key_abs = torch.mean(key_abs, 1, keepdim=False, out=None) #绝对值后求均值
            key = key.view(6,51,1) #6,51,1
            key_abs = key_abs.view(6, 51, 1)
            key = key.repeat(1,1,51)  #6，51，51
            key_abs = key_abs.repeat(1,1,51)
            A = model['st_gcn'].A   #tensor 6，51，51
            weigh = key.mul(A)   #a.mul(b)，两个6,51,51的tensor,对应元素的点乘，得到6，51，51，表示六种相邻情况下，51个节点间的最终关联权值
            weigh_abs = key_abs.mul(A)
            weigh2 = weigh.reshape(-1,weigh.shape[2]) #tensor和array都有reshape;(-1,51)      306,51
            weigh2_abs = weigh_abs.reshape(-1, weigh.shape[2])
            a = weigh2.cpu().numpy() #cuda到cpu,从tensor变数组
            a_abs = weigh2_abs.cpu().numpy()
            np.savetxt('hn.csv', a, delimiter=',')#numpy.lib.npyio中的函数,当前目录下创建文件，分隔符为逗号，只能保存一维或二维numpy,把csv换成txt也可以
            np.savetxt('hn_abs.csv', a_abs, delimiter=',')
    model['st_gcn'].load_state_dict(stgcn_dict) #将参数load给模型

post_refine_dict = model['post_refine'].state_dict() #只有weight,bias,weight,bias四个tensor
if opt.post_refine_reload == 1: #0
    pre_dict_post_refine = torch.load(os.path.join(opt.previous_dir, opt.post_refine_model)) #'model_post_refine_10_eva_post_4870.pth'
    for name, key in post_refine_dict.items():
        post_refine_dict[name] = pre_dict_post_refine[name]
    model['post_refine'].load_state_dict(post_refine_dict)


criterion = {}
criterion['MSE'] = nn.MSELoss(size_average=True).cuda() #均方误差损失  MSELoss父类的父类为Module,所以可用cuda
criterion['L1'] = nn.L1Loss(size_average=True).cuda()  #size_average=True表示要除以n


actions = define_actions(opt.actions)  #opt.actions:'*'    actions:[act1,act2,...,act15]
#开始训练
for epoch in range(1, opt.nepoch):   #range(1,200)
    print('======>>>>> Online epoch: #%d <<<<<======' % (epoch))
    torch.cuda.synchronize()  #等设备上所有流中的核心完成，为下面计时做准备

    if opt.pro_train == 1:#true  训练阶段
        timer = time.time()  #  1639215265.8108509 开始计时
        print('======>>>>> training <<<<<======')
        print('frame_number: %d' %(opt.pad*2+1))  #3
        print('processing file %s:' %opt.model_doc)  #st_gcn
        print('learning rate %f' % (lr))
        mean_error = train(opt, actions, train_dataloader, model, criterion, optimizer_all)
        timer = time.time() - timer #1639226887.0862799-1639215265.8108509
        timer = timer / len(train_data) #3119504   len(Fusion)，跳到该类的__len__子函数
        print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))  #ms 学习一个样本的时间

    #每个epoch，先训练，再测试
    if opt.pro_test == 1: #true
        timer = time.time()
        print('======>>>>> test<<<<<======')
        print('frame_number: %d' %(opt.pad*2+1))
        print('processing file %s:' %opt.model_doc)
        mean_error = val(opt, actions, test_dataloader, model, criterion) #不用优化器
        timer = time.time() - timer
        timer = timer / len(test_data)
        print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))

        if opt.save_out_type == 'xyz': #true
            data_threshold = mean_error['xyz']

        elif opt.save_out_type == 'post': #pose_refine
            data_threshold = mean_error['post']

        #每个epoch在测试后看要不要存更好的模型
        if opt.save_model and data_threshold < opt.previous_best_threshold: # 1 and 当前平均误差33.79<先前最低误差(初始设为inf)，此时保存新模型，删除旧模型，先设置新模型的名称
            opt.previous_st_gcn_name = save_model(opt.previous_st_gcn_name, opt.save_dir, epoch, opt.save_out_type, data_threshold, model['st_gcn'], 'st_gcn')

            if opt.post_refine:
                opt.previous_post_refine_name = save_model(opt.previous_post_refine_name, opt.save_dir, epoch, opt.save_out_type,
                                                      data_threshold, model['post_refine'], 'post_refine')
            opt.previous_best_threshold = data_threshold  #将当前误差设为新的最佳阈值


    if epoch % opt.large_decay_epoch == 0:  #当前epoch是否为5的倍数
        for param_group in optimizer_all.param_groups: #优化器中的参数组合
            param_group['lr'] *= 0.5  #学习率变为之前的0.5
            lr *= 0.5 #这里是为了print时

    else:
        for param_group in optimizer_all.param_groups:
            param_group['lr'] *= opt.lr_decay  #每个epoch按照0.95衰减
            lr *= opt.lr_decay






