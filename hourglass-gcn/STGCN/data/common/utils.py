import torch
import numpy as np
import hashlib


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    # Convert input types where applicable
    args = list(args) #这里是打包参数，将R赋值给*args，虽不是多个参数，但也会把它打包成元组，所以args变成了tuple,元素是R; list把tuple变成list; 第二次wrap:args:( ndarray(3304,17,4),nd(3304,17,3) ); list操作变成[ ndarray,ndarray ]
    for i, arg in enumerate(args): #i:0 arg:array(4,);
        if type(arg) == np.ndarray: #true  type（xx），完整的类型
            args[i] = torch.from_numpy(arg) #将arg从ndarray变成Tensor类
            if unsqueeze:#False
                args[i] = args[i].unsqueeze(0)

    result = func(*args) #result:{tensor:(4,)}; func:qinverse函数，*拆分参数，把元组或列表中的元素拆开，分别赋给多个参数。这里只赋给qinverse的q参数;  第二次:func指代qrot函数,args:[tensor,tensor] 拆为tensor，tensor  返回3304，17，3
    if isinstance(result, tuple): #False   判断result是不是tuple类
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor: #True
        if unsqueeze: #False
            result = result.squeeze(0)
        return result.numpy() #Tensor.numpy(),Tensor变ndarray
    else:
        return result


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value