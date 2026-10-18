import torch.utils.data as data
import numpy as np
from data.common.camera import *
from data.common.utils import deterministic_random
from data.common.generator import ChunkedGenerator

#prepare_data:对三维和二维数据做归一化、坐标转换等预处理，然后fetch得到（sub，act，i）的相参、2d、3d pose,然后Chunk对所有帧建立pairs属性和index，初始化三种256x...的缓存
class Fusion(data.Dataset):
    def __init__(self, opt, dataset, root_path, train=True):
        self.data_type = opt.dataset   #'h36m'
        self.train = train #True
        self.keypoints_name = opt.keypoints #'cpn_ft_h36m_dbb'
        self.root_path = root_path
        self.train_list = opt.subjects_train.split(',')  #  'S1,S5,S6,S7,S8'.split(',')=['S1', 'S5', 'S6', 'S7', 'S8']
        self.test_list = opt.subjects_test.split(',')    #  ['S9','s11']
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')   #None  是否有新动作
        self.downsample = opt.downsample  #下采样  1
        self.subset = opt.subset      #1  子集
        self.stride = opt.stride      #1  步幅
        self.crop_uv = opt.crop_uv    #0  标准化二维姿态 shuffle
        self.test_aug = opt.test_augmentation  #测试集数据增强 True
        self.pad = opt.pad   #1
        if self.train:
            self.keypoints = self.prepare_data(dataset, self.train_list)  #得到估计的二维图像uv坐标归一化到（-1，1）
            self.cameras_train, self.poses_train_3d, self.poses_train_2d, self.cameras_train1, self.poses_train_3d1, self.poses_train_2d1 = self.fetch(dataset,
                                              self.train_list, subset=self.subset)

            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, self.cameras_train1, self.poses_train_3d1, self.poses_train_2d1, self.stride,
                                              pad=self.pad, augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left, joints_right=self.joints_right, out_all=opt.out_all)

            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.keypoints = self.prepare_data(dataset, self.test_list)
            self.cameras_test, self.poses_test_3d, self.poses_test_2d, self.cameras_test1, self.poses_test_3d1, self.poses_test_2d1 = self.fetch(dataset,
                                              self.test_list, subset=self.subset)

            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, self.cameras_test1, self.poses_test_3d1, self.poses_test_2d1, pad=self.pad,
                                              augment=False, kps_left=self.kps_left, kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all)

            self.key_index = self.generator.saved_index #{('S9','Directions 1'):[0, 9424],...,'(S11,action):[ ]'},少一个动作，59个元素
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))  #543488

    def prepare_data(self, dataset, folder_list):
        print('Preparing data...')  #print括号里的字符串
        for subject in folder_list:#['S1','S5','S6','S7','S8']
            print('load %s' % subject)
            for action in dataset[subject].keys(): #'Sitting1'; dataset是Human36mDataset类，其父类是MocapDataset。   []转向父类里__getitem__，返回self._data[key]
                anim = dataset[subject][action] #{‘positions’:ndarray(几千，17，3), ‘cameras’:[{},{},{},{}]}

                positions_3d = []
                for cam in anim['cameras']:  #4个相机之一的参数
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation']) #函数未细看, pos_3d：ndarray(3304，17，3),一个相机坐标系下的
                    pos_3d[:, 1:] -= pos_3d[:, :1]  #a-=b等价于a=a-b; 1-16关节都减去0关节三维坐标，相当于0节点为中心，0关节不变以作为轨迹信息，ndarray(3304,16,3)减(3304,1,3),只第二维广播
                    if self.keypoints_name.startswith('sh'): #false
                        pos_3d = np.delete(pos_3d,obj=9,axis=1)# remove neck for sh 2D detection
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d #dict新增positions_3d元素，值为[ndarray,ndarray,ndarray,ndarray];
#上面只改变了dataset,每个action内加了positions_3d元素
        print('Loading 2D detections...')
        keypoints = np.load(self.root_path + 'data_2d_' + self.data_type + '_' + self.keypoints_name + '.npz',allow_pickle=True) #'/home...dataset/data_2d_h36m_cpn_ft_h36m_dbb.npz'
        keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry'] #.item（）后得到{"layout':h36m, 'num_joints':17, 'keypoints_symmetry':[[4,5,6,11,12,13],[1,2,3,14,15,16]]}
        if self.keypoints_name.startswith('sh'): #false
            self.kps_left, self.kps_right = [4,5,6,10,11,12], [1,2,3,13,14,15]
            self.joints_left, self.joints_right = [4,5,6,10,11,12], [1,2,3,13,14,15]
        else:
            self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1]) #前加list可确保是list
            self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right()) #结果和上句一样

        keypoints = keypoints['positions_2d'].item() #{ 'S6':{ 'Directions1':[ndarray(2847,17,2),nd,nd,nd],... 'action30': }, S7,5,11,1,9,8,2,3,S4 }
        for subject in folder_list: #[S1,5,6,7,8]
            assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject) #assert后加逗号，再加str.format(xx),  字符串中有{}，将xx替代{}
            for action in dataset[subject].keys():
                assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)#确认三维数据中的act，在二维中都有，str中二个{}，format( , )
                for cam_idx in range(len(keypoints[subject][action])):   #len(list)  range(4)
                    mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]  #3304
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length #3305>3304,二维帧数>=三维帧数
                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length: #如果大于，则缩短为一样长度
                        keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
                assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])  #4,确认2D和3D视图数一样
#上面确保二维和三维Sx，actionx,帧数和视角数一致
        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]): #cam_ids:0-3  kps:ndarray(3000,17,2)
                    cam = dataset.cameras()[subject][cam_idx] #{ori:, id:,...}
                    if self.crop_uv == 0:  #true
                        kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])  #归一化  ...，表示前面所有维
                    if self.keypoints_name.startswith('sh'):  #false
                        permute_index = [6,2,1,0,3,4,5,7,8,9,13,14,15,12,11,10]
                        kps = kps[:,permute_index,:]
                    keypoints[subject][action][cam_idx] = kps #该句可不要，效果一样
        return keypoints
#估计的二维uv坐标归一化到（-1，1）,为新的keypoints
    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):

        out_poses_3d = {}
        out_poses_3d1 = {}
        out_poses_2d = {}
        out_poses_2d1 = {}
        out_camera_params = {}
        out_camera_params1 = {}

        for subject in subjects:   #['S1','S5','S6','S7','S8']
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None: #None,   false
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break #跳出最里面的for/while循环
                    if not found:
                        continue


                out_poses_2d2 = []
                poses_2d = self.keypoints[subject][action]   #[array,array,array,array];
                for a in range(len(poses_2d[0])): #len(array),得数组最左维的维数1383
                    for b in range(len(poses_2d)): #range(4)
                        out_poses_2d2.append(poses_2d[b][a]) #out_poses_2d2:[第一帧的0(array17,2), 1,2,3，  第二帧的0,1,2,3....]
                out_poses_2d2 = np.array(out_poses_2d2, dtype='float32') #list变ndarray,dtype
                out_poses_2d1[(subject, action)] = out_poses_2d2 #dict加元素，名称为(str,str)，值为ndarray(5532,17,2)

                for i in range(len(poses_2d)):  #range(4)
                    out_poses_2d[(subject, action, i)] = poses_2d[i]


                out_camera_params2 = []
                if subject in dataset.cameras():   #如果训练集中的subject在dataset.cameras的对象(10个)中
                    cams = dataset.cameras()[subject]  #[ {所有相机参数},{},{},{} ]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch' #4=4,确认2维和3维数据视角数一样
                    for j in range(len(poses_2d[0])): #range(1383)
                        for i, cam in enumerate(cams):
                            if 'intrinsic' in cam:  #true
                                out_camera_params[(subject, action, i)] = cam['intrinsic']
                                out_camera_params2.append(cam['intrinsic'])
                    out_camera_params2 = np.array(out_camera_params2, dtype='float32')
                    out_camera_params1[(subject, action)] = out_camera_params2 #5532,9   相机本质矩阵


                out_poses_3d2 = [ ]
                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']            #[array,array,array,array] 分别几千个三维姿态
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'  #4=4
                    for i in range(len(poses_3d)):  # range(4)
                        out_poses_3d[(subject, action, i)] = poses_3d[i]
                    for a in range(len(poses_3d[0])): #类似上面的poses_2d
                        for b in range(len(poses_3d)):
                            out_poses_3d2.append(poses_3d[b][a])
                    out_poses_3d2 = np.array(out_poses_3d2, dtype='float32')
                    out_poses_3d1[(subject, action)] = out_poses_3d2


        if len(out_camera_params) == 0:  #false,(sub,act,i),len(dict)
            out_camera_params = None
        if len(out_poses_3d) == 0:    #false,600
            out_poses_3d = None

        stride = self.downsample #1
        if subset < 1:  #1,false  不采用数据的子集，使用全部
            for key in out_poses_2d.keys():
                n_frames = int(round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:stride]
        elif stride > 1: #false  不对训练数据进行下采样

            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d, out_camera_params1, out_poses_3d1, out_poses_2d1

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):  #执行dataloader.py的_worker_loop函数中L106,  每次从batch_indices(256个数)中取一个index:1636087
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]
        cam, gt_3D, input_2D, action, subject = self.generator.get_batch1(seq_name, start_3d, end_3d, flip, reverse)
        if self.train == False and self.test_aug: #测试阶段且测试增强为true

            _, _, input_2D_aug, _, _ = self.generator.get_batch1(seq_name, start_3d, end_3d, flip=True, reverse=reverse) # _    flip为true
            input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0) #3,17,2先扩展维度，在第一维，变成1,3,17,2,
                                                                                                                # 再把两个按第一维concatenate，变成2,3,17,2
        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        scale = np.float(1.0)

        return cam, gt_3D, input_2D_update, action, subject, scale, bb_box, start_3d








