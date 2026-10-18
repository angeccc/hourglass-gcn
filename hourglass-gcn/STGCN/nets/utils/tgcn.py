import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size #6
        self.conv = nn.Conv2d(
            in_channels,                    #2;  128;  128;  256;  512
            out_channels * kernel_size,     #(128x6=768);  128x6;  256x6;  512x6;  512x6
            kernel_size=(t_kernel_size, 1), #(1,1) 卷积核尺寸
            padding=(t_padding, 0),         #(0,0) 是否边界扩充
            stride=(t_stride, 1),           #(1,1) 卷积的步长
            dilation=(t_dilation, 1),       #(1,1) 膨胀卷积
            bias=bias)                      #True  卷积后+偏置

    def forward(self, x):
        x = self.conv(x) #(256,2,1,51) 二维卷积的输入格式：batchsize,输入通道数，xx,xx   卷积核尺寸axb,沿第三维a个参数，沿第四维b个参数卷积
        n, kc, t, v = x.size() #256,768,1,51;
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)  #[256, 6, 128, 1, 51]
        return x.contiguous() #为何要加contiguous
