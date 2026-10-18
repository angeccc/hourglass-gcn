import torch
import torch.nn as nn

class per_node_dot(nn.Module):
    def __init__(self,
                out_channels, #128; 128; 256; 512; 512
                node_num): #51; 51; 51; 15; 15
        super().__init__()
        self.per_node_weight = nn.Parameter(torch.randn(6, out_channels, 1, node_num)) #先randn创建并随机初始化tensor(0,1正态分布),rand()为0，1均匀分布，再Parameter将
                                                                                       #之变为可训练的参数，randn初始化每次都一样。
    def forward(self, x, A):
        x = self.per_node_weight.mul(x)  #(6，128，1，51).mul(256,6,128,1,51)对应位置点乘，广播原理。
        x = torch.einsum('nkctv,kvw->nctw', (x, A))  #结果：256，128，1，51
        return x.contiguous(), A #为啥contiguous
