import torch
from torch.autograd import Variable  #torch.autograd.variable.Variable
import numpy as np
import os

class AccumLoss(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val #val:当前batch的mpjpe的和
        self.sum += val #sum:val的累加
        self.count += n #count:样本数量的累加
        self.avg = self.sum / self.count #avg：现在和此前所有样本的平均mpjpe

def get_varialbe(split, target):

    num = len(target)  #5
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor) #对target中tensor执行Variable，只把requires_grad变为False,
                       #variable得到的是否和原tensor共享内存？ contiguous见笔记,type设置新属性，cpu变GPU.
            var.append(temp)
    else:
        with torch.no_grad(): #之后的内容不进行反向传播
            for i in range(num):
                temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor) #
                var.append(temp)
    return var


def get_uvd2xyz(uvd, gt_3D, cam): #gt_3D:3d真值的中间帧[256,1,17,3]   cam:256,1,9
    N, T, V,_ = uvd.size()  #256,1,17
    dec_out_all = uvd.view(-1, T, V, 3).clone()  # [256,1,17,3].clone,没变
    root = gt_3D[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1).clone()#取0节点[256,1,3],1代表帧，.unsq=[256,1,1,3]，复制17份[256,1,17,3]
    enc_in_all = uvd[:, :, :, :2].view(-1, T, V, 2).clone()  # [256,1,17,2].view=[256,1,17,2]     取uv

    cam_f_all = cam[..., :2].view(-1,1,1,2).repeat(1,T,V,1) #焦距坐标     [256,2].view=[256,1,1,2].repeat=[256,1,17,2]
    cam_c_all = cam[..., 2:4].view(-1,1,1,2).repeat(1,T,V,1)#相机中心坐标    [256,9][2:4]=[256,2].view=[256,1,1,2].repeat=[256,1,17,2]

    z_global = dec_out_all[:, :, :, 2]#网络预测的z  [256,1,17]
    z_global[:, :, 0] = root[:, :, 0, 2] #三维真值的0节点深度坐标[256,1]替代预测的0节点的深度坐标
    z_global[:, :, 1:] = dec_out_all[:, :, 1:, 2] + root[:, :, 1:, 2]  #预测的1-16节点的深度坐标=预测的1-16关节的深度坐标+三维真值的0节点深度坐标,test时??? #标签以0关节为原点。所以网络预测的也是，因此17个节点的z要加真值的0节点深度
    z_global = z_global.unsqueeze(-1)  #[256,1,17,1]
    
    uv = enc_in_all - cam_c_all#[256,1,17,2]  归一化（1，1）过的预测的2d坐标-归一化过的相机中心坐标，得到图像物理坐标。相机中心坐标很接近0，w和h不完全相同，相机中心不完全在图像中心位置。
    xy = uv * z_global.repeat(1, 1, 1, 2) / cam_f_all  #.repeat=[256,1,17,2]/  =[256,1,17,2]  得到的xy是Xc,Yc    归一化过的焦距坐标？？结合相机模型

    xyz_global = torch.cat((xy, z_global), -1)  #[256,1,17,3]
    xyz_offset = (xyz_global - xyz_global[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1))#[256,1,3].unsqu=[256,1,1,3].repeat=[256,1,17,3]，减0节点坐标，变成以0节点为中心
    return xyz_offset

def print_error(data_type,action_error_sum,show_protocol2=False):


    if data_type =='h36m':
        mean_error =  print_error_action(action_error_sum, show_protocol2)  #{'action1':{'p1':Accumloss, 'p2':Accumloss},...'action15':  }    true
    elif data_type.startswith('STB'):
        mean_error = print_error_directly(action_error_sum)

    return mean_error

def print_error_directly(action_error_sum):

    error = action_error_sum.avg * 1000.0
    print('Error:%f mm' % (error))
    return error



def print_error_action(action_error_sum, show_protocol2=False):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}
    if show_protocol2 :


        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm")) # ===Action=== ==p#1 mm== =p#2 mm=    0:  1:是编号，分别占12，10，8个字位，空格也算，不足的用=填充
        for action,value in action_error_sum.items(): #value:{p1:  p2: }
            print("{0:<12} ".format(action), end="")
            for j in range(1,3): # 1, 2

                mean_error_each['p'+str(j)] = action_error_sum[action]['p'+str(j)].avg * 1000.0
                mean_error_all['p'+str(j)].update(mean_error_each['p'+str(j)], 1)

            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'],mean_error_each['p2']))
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, mean_error_all['p2'].avg))
    else:

        print("{0:=^12} {1:=^6}".format("p#1 Action", "mm"))
        for action,value in action_error_sum.items():
            print("{0:<12} ".format(action), end="")  #结束符

            mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
            print("{0:>6.2f}".format(mean_error_each['p1']))  #%-6.2f  -为左对齐     2f表示保留小数点后两位
            mean_error_all['p1'].update(mean_error_each['p1'], 1)


        print("{0:<12} {1:>6.2f}".format("Average", mean_error_all['p1'].avg))
    return mean_error_all['p1'].avg


def print_error_xyz(action_error_sum_xyz):
    mean_error_xyz_sum = np.zeros([3])
    print("{0:=^12} {1:=^6} {2:=^6} {3:=^6}".format("p#1 Action", "x", "y","z"))
    for action, value in action_error_sum_xyz.items():
        print("{0:<12} ".format(action), end="")
        mean_error_xyz = np.array(action_error_sum_xyz[action][1:4]) /action_error_sum_xyz[action][0] * 1000.0
        mean_error_xyz_sum += mean_error_xyz
        print("{0:>6.2f} {1:>6.2f} {2:>6.2f}".format(mean_error_xyz[0],mean_error_xyz[1],mean_error_xyz[2]))
    mean_error_xyz_sum/= float(len(action_error_sum_xyz))
    print("{0:<12} {1:>6.2f}{2:>6.2f}{3:>6.2f}".format("Average", mean_error_xyz_sum[0], mean_error_xyz_sum[1], mean_error_xyz_sum[2]))


def get_loss_sum(pre_list,value_list,num_data):
    """

    :param pre_list: [loss_sum1,loss_sum2,...]
    :param value_list: [loss_value_1,loss_value_2...]
    :param num_data: number of data in this batch
    :return:
    """
    num_list = len(pre_list)
    for i in range(num_list):
        pre_list[i] = pre_list[i] + value_list[i].detach() * num_data
    return pre_list


def save_model(previous_st_gcn_name, save_dir,epoch, save_out_type, data_threshold, model, model_name):
    if os.path.exists(previous_st_gcn_name): # true
        os.remove(previous_st_gcn_name)  #删除文件

    torch.save(model.state_dict(), #'./results/3_frame/st_gcn/no_pose_refine/cpn/  model_st_gcn_  1_eva_  xyz_   3379.pth'
               '%s/model_%s_%d_eva_%s_%d.pth' % (save_dir, model_name, epoch, save_out_type, data_threshold * 100))  #保存新的模型pth文件
    previous_name = '%s/model_%s_%d_eva_%s_%d.pth' % (save_dir, model_name,epoch, save_out_type, data_threshold * 100) #把新文件的名称作为新的前名称
    return previous_name


def define_error_list(actions):  #actions:[act1,...,act15]
    error_sum = {}
    error_sum.update({actions[i]: {'p1':AccumLoss(), 'p2':AccumLoss()} for i in range(len(actions))})
    return error_sum

def back_to_ori_uv(cropped_uv,bb_box):
    N, T, V,_ = cropped_uv.size() #cropped_uv: input_2d(256,3,17,2)  bb_box:tensor(256,4):(0,0,1,1)
    uv = (cropped_uv+1)*(bb_box[:, 2:].view(N, 1, 1, 2)/2.0)+bb_box[:, 0:2].view(N, 1, 1, 2) #256,2).view=256,1,1,2)/2
    return uv





























