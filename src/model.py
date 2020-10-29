import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from guided_filter_pytorch.guided_filter import GuidedFilter
    
import settings


class FSBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y).view(b, c, 1, 1)
        y2 = self.fc2(y).view(b, c, 1, 1)
        y3 = self.fc3(y).view(b, c, 1, 1)
        return x*y1,x*y2,x*y3

class NoSEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()

    def forward(self, x):
        return x


SE = NoSEBlock

class ConvGRU(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_xz = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xr = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xn = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_hz = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hr = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hn = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        z = F.sigmoid(self.conv_xz(x))
        f = F.tanh(self.conv_xn(x))
        h = z * f 
        h = self.relu(h)
        return h

RecUnit = ConvGRU
class MDMTN(nn.Module):
    def __init__(self):
        super().__init__()
        channel = settings.channel
        self.fs = FSBlock(channel)
        self.rnns = nn.ModuleList(
            [RecUnit(3, channel, 3, 1)] + 
            [RecUnit(channel, channel, 3, 2 ** 0)] +
            [RecUnit(channel, channel, 3, 2 ** 1)] +
            [RecUnit(channel, channel, 3, 2 ** 2)] +
            [RecUnit(channel, channel, 3, 2 ** 3)] 
        )
        channel1=channel
        self.dec1 = nn.Sequential(
            nn.Conv2d(channel1, channel1, 3, padding=1),
            SE(channel1, 6),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel1, 1, 1),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(channel1, channel1, 3, padding=1),
            SE(channel1, 6),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel1, 1, 1),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(channel1,channel1, 3, padding=1),
            SE(channel1, 6),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel1, 1, 1),
        )
      

    def forward(self, x):
       
        b,c,h,w=x.shape
        ori = x  
        for rnn in self.rnns:
            x = rnn(x)
   
        x1,x2,x3=self.fs(x)
        x1 = self.dec1(x1)
        x2 = self.dec2(x2)
        x3 = self.dec3(x3)
        x=torch.cat((x1,x2,x3), 1)
  
        return x


if __name__ == '__main__':
    ts = torch.Tensor(16, 3, 64, 64)
    vr = Variable(ts)
    net = MDMTN()
    print(net)
    oups = net(vr)
    for oup in oups:
        print(oup.size())

