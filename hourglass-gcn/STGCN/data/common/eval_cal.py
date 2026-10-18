import torch
import numpy as np



def mpjpe(predicted, target):
    assert predicted.shape == target.shape #确认tensor的维度一样 torch.Size([256, 3, 17, 3])
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1)) #dim=4-1  norm对第四维计算F2范数,见笔记   torch.mean见笔记



def test_calculation(predicted, target, action, error_sum, data_type, show_protocol2=False):

    if data_type == 'h36m': #true
        error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
        if show_protocol2:#true
            error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    elif data_type.startswith('STB'):
        error_sum = mjmpe_directly(predicted, target, error_sum)

    return error_sum

def mjmpe_directly(predicted, target,action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1),dim=len(target.shape) - 2)
    # action_error_sum[0] += num
    # action_error_sum[1] += num * torch.mean(dist).item()
    action_error_sum.update(torch.mean(dist).item() * num, num)
    return action_error_sum


def mjmpe_by_action_subject(predicted, target,action,action_error_sum,subject):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1),dim=len(target.shape) - 2)
    if len(set(list(action))) == 1 and len(set(list(subject))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name][subject][0] += num
        action_error_sum[action_name][subject][1] += num*torch.mean(dist).item()
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name][subject][0] += 1
            action_error_sum[action_name][subject][1] += dist[i].item()
    return action_error_sum

def mpjpe_by_action_p1(predicted, target, action, action_error_sum):

    assert predicted.shape == target.shape #256,3,17,3
    num = predicted.size(0) #256
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2) # dim=3,dim=2   dist:256，3
    #dist = torch.norm(predicted - target, dim=len(target.shape) - 1) #256,3,17

    #if len(set(list(action))) == 1:  #true  先变成list（256个action名）,再set取一个，再len(str)
    per_action_avg = 0
    if per_action_avg: #如果求每动作误差，先改名。但被设为0,即默认不按照动作求误差，所以求误差的语句没写
        end_index = action[0].find(' ')  #directions 1找空格，第11位，从0开始，就是10
        if end_index != -1:  #不是最后一位，说明字符串中间有空格
            action_name = action[0][:end_index] #Directions  去除空格后的内容
        else:
            action_name = action[0]
    else: #每关节误差，与动作无关，随意找了Directions去update所有batch的值，其实应该加一个all
        action_name = 'Directions'
    action_error_sum[action_name]['p1'].update(torch.mean(dist, dim=0).cpu().detach().numpy()*num, num) #mean的结果是3，表示三帧共51个关节点的估计误差？？？,
                                                                                                        # 乘num表示一个batch中误差的和
    return action_error_sum

def mpjpe_by_action_p2(predicted, target,action,action_error_sum):
    """
    Aligned Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1]) #256,3,17,3.reshape(-1,17,3)=768,17,3    3指三维坐标
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1]) #ndarray(768,17,3)
    dist = p_mpjpe(pred, gt) #array(768,)
    if len(set(list(action))) == 1: #action:tuple:256
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        # action_error_sum[action_name][2] += num*np.mean(dist)
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:   # 虽是顺序取样，一个batch中仍可能有不同action
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            # action_error_sum[action_name][2] += dist[i].item()
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)
    return action_error_sum



def p_mpjpe(predicted, target):
    """
    rigid alignment (scale, rotation, and translation)

    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True) #768,1,3   768: 256x三帧  1,3表示xyz每个轴的17节点的平均值
    muY = np.mean(predicted, axis=1, keepdims=True) #768,1,3

    X0 = target - muX  #对768中每一帧17x3,xyz分别减去平均值
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True)) #**表示乘方，即两个X0 array做点乘; 再对结果求和,axis为第二和第三维度,得768,1,1; 再开根号
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True)) #768,1,1

    X0 /= normX  #减去xyz每轴平均值，再除以xyz整体(三轴求和)的标准差。对估计和真值都进行此操作 得768,17,3
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0) #类似tensor中的permute,转置,变为768,3,17,  结果是768,3,3
    U, s, Vt = np.linalg.svd(H) #奇异值分解 768,3,3;  768,3; 768,3,3
    V = Vt.transpose(0, 2, 1) #768,3,3
    R = np.matmul(V, U.transpose(0, 2, 1)) #768,3,3

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1)) #det求行列式,结果为array(768,)扩展维度,(768,1) sign(x)根据x中每元素大于等于小于0，返回1,0,-1  结果768,1
    V[:, :, -1] *= sign_detR #v的最后一维的三个数乘(-1,0,1)
    s[:, -1] *= sign_detR.flatten() #flatten使从768,1到768, s的3乘(-1,0,1)
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation 768,3,3

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2) #sum得768,1，再 768,1,1

    a = tr * normX / normY  # Scale 768,1,1
    t = muX - a * np.matmul(muY, R)  # Translation 768,1,3 *768,1,3   t:768,1,3

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t #768,17,3

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1),axis=len(target.shape) - 2)


def sym_penalty(dataset, keypoints, pred_out):
    loss_sym = 0
    if dataset == 'h36m': #true
        if keypoints.startswith('sh'): #false
            left_bone =  [(0, 4), (4, 5), (5, 6), (8, 10), (10, 11), (11, 12)]
            right_bone = [(0, 1), (1, 2), (2, 3), (8, 13), (13, 14), (14, 15)]
        else:
            left_bone =  [(0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13)]
            right_bone = [(0, 1), (1, 2), (2, 3), (8, 14), (14, 15), (15, 16)] #如：(0,4)与(0,1)是对称的骨块
        for (i_left, j_left), (i_right, j_right) in zip(left_bone, right_bone):  #zip
            left_part = pred_out[:, :, i_left] - pred_out[:, :, j_left] #骨长
            right_part = pred_out[:, :, i_right] - pred_out[:, :, j_right]
            loss_sym += torch.mean(torch.norm(left_part, dim=- 1) - torch.norm(right_part, dim=- 1)) #左骨长-右骨长，[256,1]-[256,1],再mean求平均数，所有对loss求和
    elif dataset.startswith('STB'):
        loss_sym = 0
    return loss_sym

def bone_length_penalty(dataset,keypoints,pred_out):
    """
    get penalty for the consistency of the bone length in sequences
    :return:
    """
    loss_bone = 0
    if pred_out.size(1) == 1:
        return 0
    if dataset == 'h36m' and keypoints is not 'sh_ft_h36m':
        bone_id = [(0,4),(4,5),(5,6),(8,11),(11,12),(12,13),(0,1),(1,2),(2,3),(8,14),(14,15),(15,16)
            ,(0,7),(7,8),(8,9),(9,10)]
        for (i,j) in bone_id:
            bone = torch.norm(pred_out[:,:,i]-pred_out[:,:,j],dim=-1)#N*T
            loss_bone += torch.sum(torch.var(bone,dim=1))#N

    return loss_bone






