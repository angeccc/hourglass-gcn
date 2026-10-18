import torch
import torch.nn as nn

from torch.autograd import Variable

inter_channels = [128, 256]
fc_out = inter_channels[1]
fc_unit = 1024

class post_refine(nn.Module):

    def __init__(self, opt):
        super().__init__()
        out_seqlen = 1
        fc_in = opt.out_channels*2*out_seqlen*opt.n_joints  #3x2x1x17=102
        fc_out = opt.in_channels * opt.n_joints    #2x17
        self.post_refine = nn.Sequential(
            nn.Linear(fc_in, fc_unit),    #102,1024   全连接层，参数为1024x102,bias为1024, 结果为(256,1024)
            nn.ReLU(),
            nn.Dropout(0.5, inplace=True), #参数多，dropout大一些
            nn.Linear(fc_unit, fc_out), #1024,34
            nn.Sigmoid())

    def forward(self, x, x_1):
        N, T, V, _ = x.size()   #256,1,17,3
        x_in = torch.cat((x, x_1), -1)  #256,1,17,6
        x_in = x_in.view(N, -1)  #256,102   flatten为102

        score = self.post_refine(x_in).view(N, T, V, 2) #256，34.view=256,1,17,2
        score_cm = Variable(torch.ones(score.size()), requires_grad=False).cuda() - score #torch.ones(n,m,..) 这里括号中是torch.size(),两种都可以; Variable见笔记,
                                                                                          #因为score的requires_grad为true，所以score_cm的从false变为true
        x_out = x.clone() #256,1,17,3
        x_out[:, :, :, :2] = score * x[:, :, :, :2] + score_cm * x_1[:, :, :, :2] #只对前两维的姿态细化，两种三维姿态的加权和，总权值为1,第三维仍取中间帧的估计值

        return x_out