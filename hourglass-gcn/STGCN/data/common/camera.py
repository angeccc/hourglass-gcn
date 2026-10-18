import numpy as np
import torch

from data.common.utils import wrap
from data.common.quaternion import qrot, qinverse


def normalize_screen_coordinates(X, w, h): #X:ndarray（2，）； ndarray(2843,17,2)         h:1002  w:1000
    assert X.shape[-1] == 2 #ndarray.shape=tuple,tuple[]
    return X / w * 2 - [1, h / w] #推导见论文背面  ndarray减list

def image_coordinates(X, w, h):
    assert X.shape[-1] == 2
    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2

def world_to_camera(X, R, t):  # R:ndarray(4),旋转参数;     T:ndarray(3) 平移参数
    Rt = wrap(qinverse, R) #R第一元素不变，后三个变为相反数，则得到Rt
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) #先shape操作，X.shape=tuple(3304,17,3),再[:-1]=tuple（3304，17），再*拆分参数变成3304和17  np.tile(Rt,(3304,17,1))对array复制，最低维复制一次，指最里面的中括号里的内容整体复制一倍（等于没变），第二个中括号内整体复制17倍并仍放在第二级中括号内，第三级维度复制3304倍，第三级中括号......,得到ndarray(3304,17,4);  wrap有三个参数，t是位移矩阵，通过广播的原理，将每个关节的三维坐标减去它
    #wrap的结果:ndarray(3304，17，3),一个相机参数下的三维坐标
def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t #非类中定义的函数一般最后都return语句，类中也可以有

def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1,
                           keepdim=True)
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c


def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)

    return f * XX + c