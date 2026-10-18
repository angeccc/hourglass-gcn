import numpy as np


class ChunkedGenerator:
    """ refined from https://github.com/facebookresearch/VideoPose3D
    """

    def __init__(self, batch_size, cameras, poses_3d, poses_2d,#opt.batchsize//stride=256 训练时一个batch输入帧数//stride
        #cameras:{ '(S1,Directions1)':array(5532,9),..., '(S8,act30)':array } 共150个视频;   poses_3d:array(5532,9)换成(5532,17,3);    poses_2d:换成(5532,17,2)
                 chunk_length=1, pad=0, causal_shift=0,#chunk_length=opt.stride=1,chunk_length:输入/输出隔多少帧取一个样例，一般为1，所以batchsize要除以stride; pad=1;
                                                                # causal_shift:使用因果卷积时的非对称填充偏移,0(0或“pad”)
                 shuffle=False, random_seed=1234,          #  shuffle:初始值为false，调用ChunkedGenerator时没有更改值
                 augment=False, reverse_aug=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, out_all=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d)) #assert后or,满足其一，否则print逗号后的（num,num）
        assert cameras is None or len(cameras) == len(poses_2d)      #150=150

        pairs = []
        self.saved_index = {}
        start_index = 0
#为了更好理解这些参数，假设chunk_length=3,poses_2d[key].shape[0]=602。则n_chunks=602+3-1//3=202
#offset=(202x3-602)//2=2,表示偏移.  202个块，200个是整的，602左右各加2个，凑3的倍数
#bounds：从0到202，乘3，从0到606间隔3，再减2， 从-2到604
#aug: 203个False
#reverse:203个false
#keys：203个  ‘sub’,'action'
#pairs: (sub,action) -2~601  1~604  False  False;   训练/测试样本的标签: -2, 1是起始和结束的帧编号
#end_index=0+602=602
        for key in poses_2d.keys(): #tuple('S1','Directions 1')
            assert poses_3d is None or poses_2d[key].shape[0] == poses_3d[key].shape[0]  #5532=5532
            n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length  #5532+1-1)//1=5532
            offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2  #5532x1-5532)//2=0
            bounds = np.arange(n_chunks + 1) * chunk_length - offset    #加1是为了冗余一个，才方便定义pairs; np.arange(5533),从0到5532的一维ndarray, 再乘1，减去0
            augment_vector = np.full(len(bounds - 1), False, dtype=bool) #array(0-5532都是false); 元素-1,len不变，可能括号外减1?; np.full(维度,填充的值,值的类型)
            reverse_augment_vector = np.full(len(bounds - 1), False, dtype=bool) #同上
            keys = np.tile(np.array(key).reshape([1,2]), (len(bounds - 1), 1))#将key变为array .reshape([1,2])确认维度;tile(array,(5533，1)); keys :ndarray(5533,2)
            pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, reverse_augment_vector)) # bounds[:-1]:ndarray(0-5531);  bounds[1:]:ndarray(1-5532);
               #aug:ndarray(5533个False);   rev:ndarray(5533个False);   zip后， 各自最左维取最小值5532，剩下的维度组合为tuple,keys第二维还是ndarray,一维剩下的维度是元素值，为int或bool型;
               #list后:[ (array(['S','Directions1']), 0, 1, false, false ), ( ),....,( ) ], 共5532个元素
            if reverse_aug: #false  不反向视频以数据增强
                pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, ~reverse_augment_vector))
            if augment: #true 水平翻转，增强数据  test时为False
                if reverse_aug:  #false
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, ~reverse_augment_vector))
                else:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, reverse_augment_vector)) #~取反，变为True

            end_index = start_index + poses_3d[key].shape[0] #0+5532=5532
            self.saved_index[key] = [start_index, end_index] #存储每个视频的起始结束帧index,   end_index最终为1559752,测试集中为543344
            start_index = start_index + poses_3d[key].shape[0] #5532下个循环作为新的start_index
        #循环后pairs共311万9504个元素

        #创建缓存区
        if cameras is not None: #true
            self.batch_cam = np.empty((batch_size, cameras[key].shape[-1]))  #np.empty( (256,9) )=ndarray(256,9).接近0的随机数
        if poses_3d is not None: #true
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1])) #256,1,17,3,
        self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[key].shape[-2], poses_2d[key].shape[-1])) #256,3,17,2

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size #每个epoch的batch数如有余数，批次数少1，所以加256，如整除，+256批次数多1，所以-1,
                                                                       # 3119504+256-1//256=12186；test时为2123
        self.batch_size = batch_size #256
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle #false
        self.pad = pad   #1
        self.causal_shift = causal_shift  #0
        self.endless = endless   #false
        self.state = None


        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left #[4, 5, 6, 11, 12, 13]
        self.kps_right = kps_right #[1, 2, 3, 14, 15, 16]
        self.joints_left = joints_left #[1, 2, 3, 14, 15, 16]
        self.joints_right = joints_right
        self.out_all = out_all  #true

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state


    def get_batch1(self, seq_i, start_3d, end_3d, flip, reverse):

        subject, action = seq_i #seq:ndarray:(2,)        xx , xx
        seq_name = (subject, action)
        start_2d = start_3d - self.pad - self.causal_shift  #1811-1-0=1810
        end_2d = end_3d + self.pad - self.causal_shift #1812+1-0=1813

        # 2D poses
        seq_2d = self.poses_2d[seq_name].copy() #{ndarray:(4644,17,2)}.copy  观察数据发现，seq_2d变，poses_2d中对应的不变，是因为多了一个[seq_name]吗?
        low_2d = max(start_2d, 0) #1810  max( , )   序列边界。
        high_2d = min(end_2d, seq_2d.shape[0]) #min(1813,4644)
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0: #pad_left/right序列扩充
            self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge') #假设seq_2d[0:1]为array:(1,17,2)
                                               # ( (2,0),(0,0),(0,0) ),三个维度的前扩展和后扩展， edge表示根据边缘值填充，其它扩展见笔记,从1，17，2，得到3，17，2
        else:
            self.batch_2d = seq_2d[low_2d:high_2d] # 3,17,2

        if flip: #true/false
            self.batch_2d[ :, :, 0] *= -1 #y坐标不变，x坐标变相反数，绕y轴对称
            self.batch_2d[ :, self.kps_left + self.kps_right] = self.batch_2d[ :, #左右节点序号调换，起到一种镜像效果。
                                                                  self.kps_right + self.kps_left]
        if reverse:#false
            self.batch_2d = self.batch_2d[::-1].copy()#第一维度倒序  https://blog.csdn.net/qq_28944169/article/details/88072738

        # 3D poses
        if self.poses_3d is not None:#true
            seq_3d = self.poses_3d[seq_name].copy() #4644,17,3
            if self.out_all:  #true
                low_3d = low_2d
                high_3d = high_2d
                pad_left_3d = pad_left_2d
                pad_right_3d = pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d

            if pad_left_3d != 0 or pad_right_3d != 0: #false
                self.batch_3d = np.pad(seq_3d[low_3d:high_3d],
                                          ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            else:
                self.batch_3d = seq_3d[low_3d:high_3d] #1810:1813

            if flip: #true
                self.batch_3d[ :, :, 0] *= -1 #和2d的一样
                self.batch_3d[ :, self.joints_left + self.joints_right] = self.batch_3d[ :, self.joints_right + self.joints_left]

            if reverse:
                self.batch_3d = self.batch_3d[::-1].copy()

        if self.cameras is not None:
            seq_cam = self.cameras[seq_name].copy()   #4644,9
            low_cam = max(start_3d, 0)
            high_cam = min(end_3d, seq_cam.shape[0])
            pad_left_cam = low_cam - start_3d
            pad_right_cam = end_3d - high_cam
            if pad_left_cam != 0 or pad_right_cam != 0: #false
                self.batch_cam = np.pad(seq_cam[low_cam:high_cam],
                                          ((pad_left_cam, pad_right_cam), (0, 0), (0, 0)), 'edge')
            else:
                self.batch_cam = seq_cam[low_cam:high_cam]  #1811：1812 中间帧的 (1,9)

            if flip:
                self.batch_cam[:, 2] *= -1 #相机中心的x坐标
                self.batch_cam[:, 7] *= -1 #切向畸变的x坐标

        if self.poses_3d is None and self.cameras is None: #false
            return None, None, self.batch_2d.copy(), action, subject
        elif self.poses_3d is not None and self.cameras is None:#false
            return np.zeros(9), self.batch_3d.copy(), self.batch_2d.copy(),action, subject
        elif self.poses_3d is None: #false
            return self.batch_cam, None, self.batch_2d.copy(),action, subject
        else:
            return self.batch_cam, self.batch_3d.copy(), self.batch_2d.copy(), action, subject #是因为?加.copy
