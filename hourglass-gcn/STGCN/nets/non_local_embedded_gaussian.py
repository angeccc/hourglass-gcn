import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3] #2  assert声明，不符合就报错

        self.dimension = dimension #2
        self.sub_sample = sub_sample #False

        self.in_channels = in_channels #512
        self.inter_channels = inter_channels #None

        if self.inter_channels is None:  #  把is换成==也行，None is None
            self.inter_channels = in_channels // 2 #256
            if self.inter_channels == 0: #False
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2: #True
            conv_nd = nn.Conv2d #没加括号，只把该类赋给conv_nd，不会运行该类
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2)) #最大池化，2x2框中取最大
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, #512,256
                         kernel_size=1, stride=1, padding=0)

        if bn_layer: #True
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, #256,512
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels) #512
            )
            nn.init.constant_(self.W[1].weight, 0) #sequential[1]取第二个，{BatchNorm2D}，有weight属性,   .weight得到{Parameter类}，里面的data就是Tensor,用0来填充它。
            nn.init.constant_(self.W[1].bias, 0)  #有bias属性,用0来填充它???????????????
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, #512，256
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, #512，256
                           kernel_size=1, stride=1, padding=0)

        if sub_sample: #False
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)  #256,512,3,17

        g_x = self.g(x).view(batch_size, self.inter_channels, -1) # (256,256,3,17).view(256,256,-1)
        g_x = g_x.permute(0, 2, 1) #256,51,256

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1) #(256,256,3,17).view(256,256,-1)
        theta_x = theta_x.permute(0, 2, 1)#256,51,256

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1) #(256,256,3,17).view(256,256,-1)  这三部分并列，都以x为输入，不是前者的输出作为后者输入

        f = torch.matmul(theta_x, phi_x) #256,51,51 见笔记
        f_div_C = F.softmax(f, dim=-1)   #256,51,51 见笔记  functional里定义了各种函数，如池化，dropout,各种激活函数，linear，bn,交叉熵，常见损失函数，采样策略

        y = torch.matmul(f_div_C, g_x)   #256,51,256
        y = y.permute(0, 2, 1).contiguous()  #256,256,51
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) #view(256,256,3,17)  *如何把torch.Size([3,17])变成3，17？
        W_y = self.W(y) #256,512,3,17 到网络输出时3才可能表示三维，前面都是三帧
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True): #512， sub_sample=False
        super(NONLocalBlock2D, self).__init__(in_channels, inter_channels=inter_channels, dimension=2, sub_sample=sub_sample, bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


if __name__ == '__main__':
    import torch

    for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
        img = torch.zeros(2, 3, 20)
        net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.zeros(2, 3, 20, 20)
        net = NONLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.randn(2, 3, 8, 20, 20)
        net = NONLocalBlock3D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())


