from tqdm import tqdm
from utils.utils1 import *
import data.common.eval_cal as eval_cal
import torch
import data.visual_image.vis as vis

def step(split, opt, actions, dataLoader, model, criterion, optimizer=None):
#dataLoader里的sampler下的data_source: cameras_train:dict(S,act,i),600个array(0-8)； 还有keypoints,10对象/30动作/0，1，2，3/几千17x2; generator：里面有batch_2d,batch_3d,batch_cam，都是ndarray，存储着随机初始化的256x....数据。
                       #还有cameras，同cameras_train。还有pairs。还有poses_2d,dict(sub，act，i),600,每个几千帧数据。还有类似的poses_3d;
                       #还有poses_train,同poses_3d;poses_train_2d,同poses_2d
          #里的batch_sampler在sampler的基础上多了batch-size和drop-last
          #里的dataset是Fusion类，内容和data_source一样


    num_data_all = 0 #样本的总数量
    loss_all_sum = {'loss_gt': AccumLoss(), 'loss_diff': AccumLoss(), 'loss_sum': AccumLoss()} #utils.utils1.AccumLoss

    mean_error = {'xyz': 0.0, 'post': 0.0} #两种平均误差,表示是否加入post_refine细化网络的误差,损失和误差不同
    error_sum = AccumLoss()  #误差和

    if opt.dataset == 'h36m': #true
        limb_center = [2, 5, 11, 14] if opt.keypoints.startswith('sh') else [2, 5, 12, 15] #四肢中心[2, 5, 12, 15]
        limb_terminal = [3, 6, 12, 15] if opt.keypoints.startswith('sh') else [3, 6, 13, 16] #四肢末端[3,6,13,16]
        action_error_sum = define_error_list(actions)  #{ 'act1':{'p1':AcccumLoss(), 'p2':Accum()},  ... ,'act15': }
        action_error_sum_post_out = define_error_list(actions)  #同上
        joints_left = [4, 5, 6, 10, 11, 12] if opt.keypoints.startswith('sh') else [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 13, 14, 15] if opt.keypoints.startswith('sh') else [1, 2, 3, 14, 15, 16]

    criterion_mse = criterion['MSE']  #这里赋值是为了下面应用时不加中括号
    model_st_gcn = model['st_gcn']
    model_post_refine = model['post_refine']

    if split == 'train':  #true
        model_st_gcn.train() #st_gcn每一层的training模式设为true
        if opt.out_all: #true   是否输出和输入同样的帧数
            out_all_frame = True
        else:
            out_all_frame = False
    else:
        model_st_gcn.eval()  #验证模式
        out_all_frame = True #测试,一定输出和输入同样的帧数
    torch.cuda.synchronize()  #？啥用

    # load data
    batches_largest_jerror_sample_inf = []
    batches_largest_jerror = []
    for i, data in enumerate(tqdm(dataLoader, 0)):  # for in会跳到tqdm/std.py L1129 for obj in  iterable,  得到的obj是list:7   i从0到12186
        if split == 'train': #true
            if optimizer is None: #false
                print("error! No Optimizer")
            else:
                optimizer_all = optimizer #Adam

        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, start_3d = data  #右边为list，  Tensor(256,1,9);  T(256,3,17,3);   T(256,3,17,2);  (act1,...,act256)
                                                                            #(Sx,...,Sx);   T(256个1)；    T(256个[0,0,1,1])

        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box]) #cpu类型数据转成GPU型,[]=[], 对应位置的元素分别赋值

        N = input_2D.size(0) #256
        num_data_all += N    #0+256=256

        out_target = gt_3D.clone().view(N, -1, opt.n_joints, opt.out_channels) #clone见笔记  [256,3,17,3].view(256,-1,17,3)=256,3,17,3  第四维表示三维坐标
        out_target[:, :, 0] = 0   #第一关节的三维坐标变0
        gt_3D = gt_3D.view(N, -1, opt.n_joints, opt.out_channels).type(torch.cuda.FloatTensor) #[256,3,17,3].view(256,-1,17,3)=256,3,17,3,三维真值除0节点不为0，
                                                                                             # 其它和out_target一样    L92已设置了cuda,这里重复了
        if out_target.size(1) > 1:  #3>1 ,true
            out_target_single = out_target[:, opt.pad].unsqueeze(1) #  [256,17,3].unsqueeze增加维度[256, 1, 17, 3]
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D

#前向传播
        if opt.test_augmentation and split =='test': # 测试阶段且测试增强 true
            input_2D, output_3D = input_augmentation(input_2D, model_st_gcn, joints_left, joints_right) #测试 256,2,3,17,1;  256,3,1,17,1
        else: #二维输入变成[256,2,3,17,1],便于输入网络
            input_2D = input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(torch.cuda.FloatTensor)#[256,3,17,2].view(256,-1,17,2,1)
            # =[256，3，17，2，1].permute()=256，2，3，17，1,将之前的按照0，3，1，2，4重新排列  L92已设置了cuda,这里重复了
            output_3D = model_st_gcn(input_2D, out_all_frame) #out_all_frame:true; output_3D:[256, 3, 3, 17, 1],第三维表示3帧;   Model父类是Module,加括号会运行模型,
                                                              #先跳到Module类中L471的__call__(),self为Model
#对网络输出的3d结果的处理
        output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.n_joints, opt.out_channels) #.permute变成[256,3,17,1,3],view(256,-1,17,3)变成[256,3,17,3];第2维表示3帧    test:256,1,17,3    可用不同的大写字母来表示每维度的含义
        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1), opt.n_joints, opt.out_channels) #scale变成256,1,1
                                                                                                            # ,1.repeat(1,3,17,3)=256,3,17,3, tensor*是dian cheng
        if output_3D.size(1) > 1: #帧数3>1,true
            output_3D_single = output_3D[:, opt.pad].unsqueeze(1) #取中间帧，得到[256,17,3].unsq=[256,1,17,3]
        else:
            output_3D_single = output_3D

        plot_largest_error_poses = 0
        if split =='test' and opt.vis_keypoints == True: #test and true时画图
            if plot_largest_error_poses:
                samples_joints_error = torch.norm(out_target_single-output_3D_single, dim=len(out_target_single.shape)-1 ) #512,1,17
                joint_largest_error, flatten_indices = torch.max(samples_joints_error)



                origin_idx = np.unravel_index(flatten_indices, (N,1,17))
                batches_largest_jerror.append(joint_largest_error)
            #        topn_joint_errors, batch_indices = torch.topk(batches_largest_jerror.view(samples_joints_error.numel()), 1)
            #        batches_largest_jerror_sample_inf.append((i,(origin_idx),joint_largest_error,subject[origin_idx[0]],action[origin_idx[0]],start_3d[origin_idx[0]],out_target_single,output_3D_single))

            else:
                for batch_i in range(N): #for each frame plot
                    if action[batch_i]=='SittingDown' and start_3d[batch_i]== 10487: #SittingDown 11727
                        vis.visualize_3d_gt_pred(out_target_single, output_3D_single, kind="human36m", batch_index=batch_i, subject=subject, action=action, start_3d=start_3d)

        if split == 'train': #true
            pred_out = output_3D #[256,3,17,3],第2维表示3帧
        elif split == 'test':
            pred_out = output_3D

        # 从uvd计算相机坐标系下的3d姿态xyz    uv表示图像平面坐标，d表示深度
        input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints, 2)  #[256,2,3,17,1].permute=[256,3,17,2,1].view(256,-1,17,2)=[256,3,17,2]
        if opt.crop_uv:  #0, 即false
            pred_uv = back_to_ori_uv(input_2D, bb_box) #与边界框裁剪有关
        else:
            pred_uv = input_2D #[256,3,17,2], 估计的二维姿态,3帧uv

        uvd = torch.cat((pred_uv[:, opt.pad, :, :].unsqueeze(1), output_3D_single[:, :, :, 2].unsqueeze(-1)), -1) #取中间帧[256,17,2].unsque=[256,1,17,2],
                                # stgcn网络输出的三维姿态的中间帧的深度值output_3D_single，叠加后变成[256,1,17,3]; cat(grad=false, grad=true)得到的tensor的grad为true
        xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)  #xyz:[256, 1, 17, 3]

#对细化的姿态计算mpjpe损失和对称损失(中间帧)
        if opt.post_refine: #false?
            post_out = model_post_refine(output_3D_single, xyz)  #两种三维姿态的加权和
            loss_sym = eval_cal.sym_penalty(opt.dataset, opt.keypoints, post_out) #骨长对称损失
            loss_post_refine = eval_cal.mpjpe(post_out, out_target_single) + 0.01*loss_sym #refine是否可以对三帧都优化
        else:
            loss_post_refine = 0

#st_gcn3维预测的mpjpe损失
        loss_gt = eval_cal.mpjpe(pred_out, out_target) #第2维表示3帧
#st_gcn的对称损失
        if opt.sym_penalty:#true
            loss_gt += 0.01 * eval_cal.sym_penalty(opt.dataset, opt.keypoints, pred_out) #对称损失的权重为0.01

        if opt.pad == 0 or split == 'test' or opt.out_all==True:  #多视角不会加该损失，该损失在时间帧上有意义。
            loss_diff = torch.zeros(1).cuda() #tensor  0
        else:
            weight_diff = 4 * torch.ones(output_3D[:, :-1, :].size()).cuda() #(256,2,17,3),全为4
            weight_diff[:, :, limb_center] = 2.5 #2,5,12,15 四肢中心权重变2.5
            weight_diff[:, :, limb_terminal] = 1#3,6,13,16四肢终端权重变1
            diff = (output_3D[:, 1:] - output_3D[:, :-1]) * weight_diff  #网络输出的二三帧减去一二帧，再乘以权重
            loss_diff = criterion_mse(diff, Variable(torch.zeros(diff.size()), requires_grad=False).cuda())

        loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine
        #几种损失的存储
        loss_all_sum['loss_gt'].update(loss_gt.detach().cpu().numpy() * N, N) #loss_gt是tensor(一个数)，detach使require_grad变False。 变cpu和adrray,N是batchsize,update是AcumLoss的子函数
        loss_all_sum['loss_diff'].update(loss_diff.detach().cpu().numpy() * N, N)
        loss_all_sum['loss_sum'].update(loss.detach().cpu().numpy() * N, N) #总损失

        #反向传播
        if split == 'train': #true
            optimizer_all.zero_grad() #Adam下的params_groups中有59个Parameter类。   在pytorch中，梯度值是累加的,所以每个batch,梯度值要清零。
            loss.backward() #根据当前batch下的预测值、真值、损失函数，基于链式法则，计算各层参数的梯度
            optimizer_all.step() #Adam中存了参数的当前值v，v=v-a*梯度,a为学习率。新参数值参与下一个batch的前向传播。
            #计算训练误差
            pred_out[:, :, 0, :] = 0 #256,3,17,3 ,0节点为0； 估计的零关节是趋于0的，但有小偏差，可人为改为0，再算误差
            joint_error = eval_cal.mpjpe(pred_out, out_target).item()  #误差;    tensor.item()取一维tensor中的值
            error_sum.update(joint_error*N, N) #更新误差的存储

        elif split == 'test':
            pred_out[:, :, 0, :] = 0 #256,3,17,3
            action_error_sum = eval_cal.test_calculation(pred_out, out_target, action, action_error_sum, opt.dataset, show_protocol2=opt.show_protocol2)#{Dire
                                                                                                 #ctions:{p1:AccumLoss, p2:AccumLoss}, Discussion:{ , },...} 15种动作

            if opt.post_refine:
                post_out[:, :, 0, :] = 0
                action_error_sum_post_out = eval_cal.test_calculation(post_out, out_target, action, action_error_sum_post_out, opt.dataset, #使用细化后的三维姿态post_out来计算误差
                                                             show_protocol2=opt.show_protocol2)

    if split == 'train': #1个epoch(12186个batch)结束后执行
        mean_error['xyz'] = error_sum.avg
        print('loss gt of 1 sample: %f mm' % (loss_all_sum['loss_gt'].avg *1000))  #st-gcn网络的所有train样本的(mpjpe+对称loss)的平均
        print('loss diff of 1 sample: %f' % (loss_all_sum['loss_diff'].avg *1000)) #.......diff损失的.....
        print('loss of 1 sample: %f' % (loss_all_sum['loss_sum'].avg *1000))     #......所有损失项的......

        print('mean joint error: %f' % (mean_error['xyz']*1000)) #训练集的mpjpe的平均

    elif split == 'test':
        if not opt.post_refine:
            mean_error_all = print_error(opt.dataset, action_error_sum, opt.show_protocol2)
            mean_error['xyz'] = mean_error_all

        elif opt.post_refine:
            print('-----post out')
            mean_error_all = print_error(opt.dataset, action_error_sum_post_out,  opt.show_protocol2)
            mean_error['post'] = mean_error_all

    return mean_error


def train(opt, actions, train_loader, model, criterion, optimizer):
    return step('train',  opt, actions, train_loader, model, criterion, optimizer)


def val(opt, actions, val_loader, model, criterion):
    return step('test',  opt, actions, val_loader, model, criterion)

def input_augmentation(input_2D, model_st_gcn, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape #256,2,3,17,2
    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) #N, C, T, J , M-->256,2,3,17,1
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) #N, C, T, J , M  256,2,3,17,1

    output_3D_flip = model_st_gcn(input_2D_flip, out_all_frame=True) #256,3,3,17,1
    output_3D_flip[:, 0] *= -1 #第三维是帧数，x坐标变相反数。由flip的2d姿态中估计的3d姿态,镜像翻转回去，应该和非flip二维姿态估计的3d姿态是相似的，取两者的均值作为最终的输出。
    output_3D_flip[:, :, :, joints_left + joints_right] = output_3D_flip[:, :, :, joints_right + joints_left]
    output_3D_non_flip = model_st_gcn(input_2D_non_flip, out_all_frame=True)

    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    input_2D = input_2D_non_flip

    return input_2D, output_3D