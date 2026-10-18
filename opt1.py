import argparse #./python3.6/argparse.py
import os
import math

#可调的参数放到一个py文件中，修改方便，否则要各个地方修改，易出错。
class opts():  #（）可以空，可以无括号，可以（父类）
    def __init__(self): #()也可有其它参数。类内每个def括号内都有self，等同于opts类。
        self.parser = argparse.ArgumentParser() #添加self.类内属性；  运行ArgumentParser类中的__init__函数（有点复杂）
   #每个参数的作用，修改help？
    def init(self):
        #=====数据的可选参数=====
        self.parser.add_argument('--dataset', type=str, default='h36m', help='dataset') #按顺序赋值，可直接写值，顺序排列也无法区分是哪个参数时必须写参数名， 参数只写要改变的和没有赋值的
        self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, metavar='NAME',
                            help='2D pose use gt/cpn_ft_h36m_dbb')
        self.parser.add_argument('--data_augmentation', type=bool, default=True, help='training-set flip to augment') #True大写
        self.parser.add_argument('--test_augmentation', type=bool, default=True, help='flip and fuse the output result')
        self.parser.add_argument('--crop_uv', type=int, default=0, help='if crop_uv to center and do normalization')
        self.parser.add_argument('--root_path', type=str, default='/home/chuanjiang/Projects/3view_306weights-notcn2022_1205_annotated/STGCN/data/dataset/', help='dataset root path')
        self.parser.add_argument('--cal_uvd', type=bool, default=True, help='calculate uvd error as well')
        self.parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST', help='actions to train/test on, separated by comma, or * for all') #取*表示在所有action上训练和测试，如果不为*，取几种动作的str，表示在这几种动作上操作,在load_data_hm36的L16和L109
        self.parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor (semi-supervised)')
        self.parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
        self.parser.add_argument('-s', '--stride', default=1, type=int, metavar='N', help='chunk size to use during training')
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False, help='if reverse the video to augment data') #反向视频来增强 generator.py L43

        # =====网络训练测试的可选参数=====
        self.parser.add_argument('--pro_train', type=int, default=0, help='if start train process')
        self.parser.add_argument('--pro_test', type=int, default=1, help='if start test process')
        self.parser.add_argument('--nepoch', type=int, default=200, help='number of epochs')
        self.parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
        self.parser.add_argument('--learning_rate', type=float, default=5e-4)
        self.parser.add_argument('--large_decay_epoch', type=int, default=5, help='give a large lr decay after how manys epochs')
        self.parser.add_argument('--sym_penalty', type=int, default=1, help='if add sym penalty add on train_multi')
        self.parser.add_argument('--co_diff', type=float, default=0)
        self.parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
        self.parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer for SGD')
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='SGD or Adam')
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float, metavar='LR', help='learning rate decay per epoch')
        self.parser.add_argument('--save_skl', type=bool, default=False)

        #=====模型的可选参数=====
        self.parser.add_argument('--pad', type=int, default=1)
        self.parser.add_argument('--show_protocol2', action='store_true') #无default,store_true:终端写入--show_protocpol2时，则show_protocol2=true，没写入则为false； store_false：则相反
        self.parser.add_argument('--model_doc', type=str, default='st_gcn', help='current model document name')
        self.parser.add_argument('--layout', type=str, default='hm36_gt', help='dataset used')
        self.parser.add_argument('--strategy', type=str, default='spatial', help='dataset used')
        self.parser.add_argument('--save_model', type=int, default=0, help='if save model')
        self.parser.add_argument('--save_out_type', type=str, default='xyz', help='xyz/uvd/post/time')
        self.parser.add_argument('--post_refine', action='store_true', help='if use post_refine model')
        self.parser.add_argument('--stgcn_reload', type=int, default=1, help='if continue from last time')
        self.parser.add_argument('--post_refine_reload', type=int, default=0, help='if continue from last time')
        self.parser.add_argument('--previous_dir', type=str,default='/home/chuanjiang/Projects/3view_306weights-notcn2022_1205_annotated/STGCN/results/3_frame/st_gcn/no_pose_refine/cpn/', help='previous output folder')
        self.parser.add_argument('--stgcn_model', type=str, default='model_st_gcn_75_eva_xyz_3055.pth', help='model name')
        self.parser.add_argument('--post_refine_model', type=str, default='model_post_refine_10_eva_post_4870.pth', help='model name')
        self.parser.add_argument('--n_joints', type=int, default=17, help='number of joints, 16 for human body 21 for hand pose')
        self.parser.add_argument('--out_joints', type=int, default=17, help='number of joints, 16 for human body 21 for hand pose')
        self.parser.add_argument('--out_all', type=bool, default=True, help='output 1 frame or all frames')
        self.parser.add_argument('--in_channels', type=int, default=2, help='expected input channels here 2')
        self.parser.add_argument('--out_channels', type=int, default=3, help='expected input channels here 2')
        self.parser.add_argument('-previous_best_threshold', type=float, default=math.inf, help='threshold data:reg_RGB_3D/reg_3D_3D')
        self.parser.add_argument('-previous_st_gcn_name', type=str, default='', help='save last saved model name')
        self.parser.add_argument('-previous_post_refine_name', type=str, default='', help='save last saved model name')
        self.parser.add_argument('--vis_keypoints', type=bool, default=True, help='if plot 3d poses')

    def parse(self):
        self.init() #self.init()，表示运行该函数
        self.opt = self.parser.parse_args()#parse_args是解析parser，得到{Namespace类}：(batchSize=256, dataset='h36m', keypoints='cpn_ft_h36m_dbb'等)
        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)  if not name.startswith('_'))#for取一个元素，if判断一次，符合的执行for前括号里的
        # startswith是pycharm内置函数， dir(Namespace)见笔记dir()函数;for xx in list， （name,getattr(...)）;
        #getattr(Namespace类，名称)返回该名称的值; dict括号对整体，循环完后dict转为词典。 最终结果：{'actions': '*', 'batchSize': 256, 'cal_uvd': True，...}

        #设置训练的pth模型和log存放的路径
        self.opt.save_dir = './results/'+'%d_frame/'%(self.opt.pad*2+1) +self.opt.model_doc + '/'+ \
        '%spose_refine/'%('' if self.opt.post_refine else 'no_')  #Namespace.xx可在命名空间中添加xx属性；  str的连接，直接+，加引号;%d后用%（），中间不用空格，括号里的结果替换%d（整数）;%s表示str
        # ‘’表示空str; xx if xx  else xx; 结果：'./results/3_frame/st_gcn/pose_refine/'   路径要斜线开始和收尾 ./表示到main py文件的path

        if self.opt.dataset == 'h36m': #true    if可省略else
            self.opt.subjects_train = 'S1,S5,S6,S7,S8'
            self.opt.subjects_test = 'S9,S11'

        if self.opt.keypoints == 'cpn_ft_h36m_dbb':
            self.opt.save_dir += 'cpn/'  #如使用CPN网络预测的2d姿态,路径后加cpn/
            self.opt.layout = 'hm36_gt'

        elif self.opt.keypoints == 'gt': #用二维真值  elif
            self.opt.save_dir += 'gt/'
            self.opt.layout = 'hm36_gt' #无论是否用真值2D，layout都是hm36_gt(三维真值)

        if not os.path.exists(self.opt.save_dir): #系统中是否有这个文件路径
            os.makedirs(self.opt.save_dir) #创建文件
        file_name = os.path.join(self.opt.save_dir, 'opt.txt')  #两个str连到一起, 得到opt.txt的路径

        with open(file_name, 'wt') as opt_file: #见笔记， open内置函数，创建了TxetIOWrapper类; w和wt一样，r和rt一样，t是默认参数，可省略;类Unix平台的换行符是\n，
                                                # windows平台是\r\n，python采用\n。 加t，python读取文本时自动把\r\n转换成\n.
            opt_file.write('==> Args:\n')    #write内置函数
            for k, v in sorted(args.items()):#items内置函数  dict.items()=>{dict_items:46}([('actions', '*'), ('batchSize', 256), ... ,('workers', 6)]);
                                             #sorted内置函数，排序，应该是按照首字母，结果是{list:46}：[('actions', '*'), ('batchSize', 256), ... , ('workers', 6)]
                                             #每个元素是元组(tuple); for取1元素，两个值赋给k和v
                opt_file.write('%s: %s\n' % (str(k), str(v))) #两个%s，     %（）括号内两个替换的值逗号隔开;转换为str型
            opt_file.write('==> Args:\n') #最后加此行，表示结束

        print(self.opt) #Namespace可直接print到终端上
        return self.opt #return比def要缩进一步
