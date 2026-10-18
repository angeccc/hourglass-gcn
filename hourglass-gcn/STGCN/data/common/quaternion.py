import torch
#文件名quaternion是四元数的意思

def qrot(q, v): #四元数和旋转之间的原理见学习记录https://blog.csdn.net/candycat1992/article/details/41254799
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4  #q:3304,17,4
    assert v.shape[-1] == 3  #v:3304,17,3
    assert q.shape[:-1] == v.shape[:-1] #torch.Size([3304,17])

    qvec = q[..., 1:] #(3304,17,3)
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)  #len(torch.Size(x,x,x))=3,dim=2,qvec第三维的三元素与v对应位置做交叉积，计算见笔记。
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return (v + 2 * (q[..., :1] * uv + uuv)) #最外面加了括号3304，17，1与3304，17，3tensor相乘,广播，对应点乘。


def qinverse(q, inplace=False): #q: tensor:(4,)
    # 假设这个四元数已经被标准化
    if inplace: #false
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1] #tensor[ ]索引; 逗号前是...表示从最高维到倒数第二维。逗号后跟数组一样，取第一维，得到的结果不是一个元素值，也是Tensor(1,)
        xyz = q[..., 1:] #tensor([-0.1501, -0.7552,  0.6223])
        return torch.cat((w, -xyz), dim=len(q.shape) - 1) #Tensors.shape={Size:1}取4; len(Size)=1  -Tensor,元素值变负数  torch.cat在第0维将元素串联，变Tensor(4,)